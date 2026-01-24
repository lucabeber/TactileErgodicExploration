"""
Density Estimator Module for Tactile Ergodic Exploration

This module provides a reusable interface for estimating density distributions on point clouds
using Gaussian Process Regression (GPR). It can be used by different exploration methods
(HEDAC, SMC, Bayesian Optimization, etc.) to maintain consistent density estimation.

Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>
"""

import numpy as np
import torch
import gpytorch
from pointcloud_utils import normalize_mat, get_border_indices
from gpr_on_point_cloud import rbf_manifold_kernel, GPROnPointCloud


class DensityEstimator:
    """
    A class for estimating density distributions on point clouds using GPR.

    This estimator maintains a GP model that is trained on trajectory samples
    and can predict density distributions over the entire point cloud.
    """

    def __init__(self, pcloud, target_density, gp_params=None, dtype=torch.float32):
        """
        Initialize the density estimator.

        Args:
            pcloud: Point cloud object with vertices attribute
            target_density: Initial target density distribution (u0)
            gp_params: Dictionary with GP hyperparameters:
                - l: length scale (default: 0.010)
                - sigma: signal variance (default: 1.0)
                - n_eig: number of eigenfunctions (default: 500)
            dtype: PyTorch dtype for tensors (default: torch.float32)
        """
        self.pcloud = pcloud
        self.vertices = pcloud.vertices
        self.target_density = target_density
        self.dtype = dtype

        # Set default GP parameters
        if gp_params is None:
            gp_params = {}
        self.l = gp_params.get('l', 0.010)
        self.sigma = gp_params.get('sigma', 1.0)
        self.n_eig = gp_params.get('n_eig', 500)

        # Initialize the manifold kernel
        print(f"Initializing GP kernel (l={self.l}, sigma={self.sigma}, n_eig={self.n_eig})...")
        self.km = rbf_manifold_kernel(self.vertices, self.l, self.sigma, self.n_eig)

        # Initialize the "ground truth" model (used to generate observations)
        train_x = torch.tensor(self.vertices, dtype=self.dtype)
        train_y = torch.tensor(self.target_density, dtype=self.dtype)

        self.likelihood_real = gpytorch.likelihoods.GaussianLikelihood()
        self.model_real = GPROnPointCloud(
            train_x, train_y, self.likelihood_real, self.km, self.vertices
        )

        # Set to evaluation mode
        self.model_real.eval()
        self.likelihood_real.eval()

        # Get initial prediction
        with torch.no_grad():
            observed_pred = self.likelihood_real(self.model_real(train_x))
        self.initial_mean = observed_pred.mean.cpu().numpy()

        print("Density estimator initialized")

    def sample_density_at_point(self, point):
        """
        Sample the target density at a given point.

        Args:
            point: 3D coordinates as numpy array (shape: (3,))

        Returns:
            Sampled density value
        """
        sample_point = torch.tensor(point, dtype=self.dtype).reshape(1, -1)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            prediction = self.likelihood_real(self.model_real(sample_point))

        return prediction.mean.cpu().numpy().item()

    def sample_density_at_points(self, points):
        """
        Sample the target density at multiple points (batched version).

        Args:
            points: 3D coordinates as numpy array (shape: (n_points, 3))

        Returns:
            Sampled density values as numpy array (shape: (n_points,))
        """
        sample_points = torch.tensor(points, dtype=self.dtype)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            prediction = self.likelihood_real(self.model_real(sample_points))

        return prediction.mean.cpu().numpy()

    def create_online_estimator(self, initial_trajectory, initial_samples):
        """
        Create an online GPR estimator that updates as new samples are collected.

        Args:
            initial_trajectory: Initial trajectory points (shape: (n_samples, 3))
            initial_samples: Corresponding density samples (shape: (n_samples,))

        Returns:
            OnlineGPEstimator object
        """
        return OnlineGPEstimator(
            self.vertices,
            self.km,
            initial_trajectory,
            initial_samples,
            dtype=self.dtype
        )

    def get_initial_mean(self):
        """Get the initial mean prediction (ground truth)."""
        return self.initial_mean.copy()


class OnlineGPEstimator:
    """
    Online GP estimator that updates as new trajectory samples are collected.

    This estimator is used during exploration to maintain an evolving estimate
    of the density distribution based on the agent's trajectory.
    """

    def __init__(self, vertices, kernel_matrix, initial_trajectory, initial_samples, dtype=torch.float32):
        """
        Initialize online GP estimator.

        Args:
            vertices: Point cloud vertices
            kernel_matrix: Pre-computed kernel matrix from rbf_manifold_kernel
            initial_trajectory: Initial trajectory points
            initial_samples: Initial density samples
            dtype: PyTorch dtype
        """
        self.vertices = vertices
        self.km = kernel_matrix
        self.dtype = dtype

        # Initialize training data
        self.train_x = torch.tensor(initial_trajectory, dtype=self.dtype)
        self.train_y = torch.tensor(initial_samples, dtype=self.dtype)

        # Initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = GPROnPointCloud(
            self.train_x, self.train_y, self.likelihood, self.km, self.vertices
        )

        # Set to training mode initially (will train when updated)
        self.model.train()
        self.likelihood.train()

        # Set to evaluation mode for predictions
        self.model.eval()
        self.likelihood.eval()

    def update(self, new_trajectory, new_samples):
        """
        Update the GP model with new trajectory samples.

        Args:
            new_trajectory: New trajectory points (shape: (n_new, 3))
            new_samples: Corresponding new density samples (shape: (n_new,))
        """
        # Concatenate new data with existing data
        new_x = torch.tensor(new_trajectory, dtype=self.dtype)
        new_y = torch.tensor(new_samples, dtype=self.dtype)

        self.train_x = torch.cat([self.train_x, new_x], dim=0)
        self.train_y = torch.cat([self.train_y, new_y], dim=0)

        # Update model with new training data
        self.model.train()
        self.likelihood.train()
        self.model.set_train_data(self.train_x, self.train_y, strict=False)

        # Switch back to evaluation mode
        self.model.eval()
        self.likelihood.eval()

    def replace_training_data(self, trajectory, samples):
        """
        Replace (not concatenate) the GP model's training data.

        This matches the original HEDAC behavior where the entire trajectory
        is resampled and used as training data at each update.

        Args:
            trajectory: Trajectory points (shape: (n_points, 3))
            samples: Corresponding density samples (shape: (n_points,))
        """
        # Replace training data
        self.train_x = torch.tensor(trajectory, dtype=self.dtype)
        self.train_y = torch.tensor(samples, dtype=self.dtype)

        # Update model with new training data
        self.model.train()
        self.likelihood.train()
        self.model.set_train_data(self.train_x, self.train_y, strict=False)

        # Switch back to evaluation mode
        self.model.eval()
        self.likelihood.eval()

    def predict(self, test_points=None, return_variance=True):
        """
        Predict density at test points.

        Args:
            test_points: Points to predict at. If None, uses all vertices.
            return_variance: Whether to return variance along with mean

        Returns:
            If return_variance=True: (mean, variance) as numpy arrays
            If return_variance=False: mean as numpy array
        """
        if test_points is None:
            test_x = torch.tensor(self.vertices, dtype=self.dtype)
        else:
            test_x = torch.tensor(test_points, dtype=self.dtype)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(test_x))

        mean = observed_pred.mean.cpu().numpy()

        if return_variance:
            variance = observed_pred.variance.cpu().numpy()
            return mean, variance
        else:
            return mean

    def get_goal_density(self, exploit_alpha, nb_boundary_neighbors=40):
        """
        Compute goal density combining mean and variance (exploration vs exploitation).

        Args:
            exploit_alpha: Weight for exploitation (0=pure exploration, 1=pure exploitation)
            nb_boundary_neighbors: Number of neighbors for border detection

        Returns:
            Normalized goal density array
        """
        mean, variance = self.predict(return_variance=True)

        # Normalize mean and variance
        mean_norm = normalize_mat(np.maximum(mean - np.mean(mean), 0))
        var_norm = normalize_mat(variance)

        # Optionally zero out border regions
        # border_indices = get_border_indices(self.vertices, nb_boundary_neighbors)

        # Combine mean and variance
        goal_density = exploit_alpha * mean_norm + (1 - exploit_alpha) * var_norm
        goal_density = normalize_mat(goal_density)

        return goal_density


def create_density_estimator(pcloud, target_density=None, gp_params=None, dtype=torch.float32):
    """
    Factory function to create a density estimator.

    Args:
        pcloud: Point cloud object
        target_density: Target density distribution (if None, uses boundary as target)
        gp_params: GP hyperparameters dictionary
        dtype: PyTorch dtype

    Returns:
        DensityEstimator instance
    """
    if target_density is None:
        # Default: use boundary as target
        from pointcloud import Pointcloud
        pcd_helper = Pointcloud(pcloud.vertices)
        target_density = np.zeros(len(pcloud.vertices))
        target_density[pcd_helper.is_boundary_arr] = 1.0

    return DensityEstimator(pcloud, target_density, gp_params, dtype)
