"""
Exploration with GP Estimation Mixin

This module provides a mixin class that adds online GP density estimation
to any exploration algorithm. It can be combined with exploration classes
via multiple inheritance.

Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>
"""

from abc import abstractmethod

import numpy as np
import torch

from ..estimation.density_estimator import create_density_estimator
from ..utils.pointcloud_utils import normalize_mat


class ExplorationWithEstimation:
    """
    Mixin for adding GP density estimation to exploration algorithms.

    This is not a standalone class - it must be combined with a PointcloudExploration
    subclass using multiple inheritance. It adds:
    - GP density estimator creation
    - Online density estimation
    - Sample collection and GP updates
    - Target density computation (mean + variance)

    Example usage:
        class HEDACEstimation(PointcloudHEDAC, ExplorationWithEstimation):
            def compute_goal_density(self, gp_mean, gp_variance, coverage):
                return normalize_mat(gp_mean + self.exploit_alpha * gp_variance)
    """

    def __init__(self, *args, exploit_alpha=0.6, gp_update_interval=50, gp_params=None, **kwargs):
        """
        Initialize estimation mixin parameters.

        Args:
            exploit_alpha: Balance between exploration (0) and exploitation (1)
                         Higher values favor exploring uncertain areas
            gp_update_interval: How often to update GP model (every N timesteps)
            gp_params: GP hyperparameters dictionary (l, sigma, n_eig)
            *args, **kwargs: Passed to parent class (PointcloudExploration subclass)
        """
        # Call parent class __init__ (the exploration class)
        super().__init__(*args, **kwargs)

        # Estimation-specific parameters
        self.exploit_alpha = exploit_alpha
        self.gp_update_interval = gp_update_interval

        # Set default GP parameters if not provided
        if gp_params is None:
            gp_params = {"l": 0.010, "sigma": 1.0, "n_eig": 500}
        self.gp_params = gp_params

        # Will be set during setup
        self.density_estimator = None
        self.online_estimator = None

        # Tracking arrays for estimation
        self.goal_density_arr = None
        self.estimated_density_arr = None

    def setup_gp_estimator(self):
        """
        Initialize GP density estimator.

        This creates the density estimator using the point cloud's color/texture
        data (u0 attribute) as the ground truth density to learn.
        """
        if self.verbose:
            print("\n" + "=" * 50)
            print("Setting up GP Density Estimator")
            print("=" * 50)

        self.density_estimator = create_density_estimator(
            self.pcloud,
            self.pcloud.u0,  # Use red channel as ground truth
            self.gp_params,
            dtype=torch.float32,
        )

        if self.verbose:
            print("GP estimator initialized")

    @abstractmethod
    def compute_goal_density(self, gp_mean, gp_variance, coverage):
        """
        Compute goal density from GP estimates and current coverage.

        This method defines how the algorithm combines GP predictions into
        a target distribution. Must be implemented by subclasses.

        Args:
            gp_mean: GP mean prediction (N,)
            gp_variance: GP variance prediction (N,)
            coverage: Current coverage array (N,)

        Returns:
            np.ndarray: Goal density (N,) to use for exploration

        Example implementations:
            # Exploration-exploitation balance
            return normalize_mat(gp_mean + exploit_alpha * gp_variance)

            # Pure exploitation (follow mean)
            return normalize_mat(gp_mean)

            # Adaptive based on coverage
            low_coverage = coverage < coverage.mean()
            goal = gp_mean.copy()
            goal[low_coverage] += exploit_alpha * gp_variance[low_coverage]
            return normalize_mat(goal)
        """
        raise NotImplementedError("Subclasses must implement compute_goal_density()")

    def explore(self):
        """
        Override explore() to add GP sampling and updates.

        This method wraps the base exploration loop with:
        1. GP estimator setup
        2. Initial sampling
        3. Periodic GP updates during exploration
        4. Goal density recomputation from GP predictions
        """
        # Setup point cloud and algorithm (from PointcloudExploration)
        self.setup_point_cloud()
        self.setup_algorithm()

        # Setup GP estimator
        self.setup_gp_estimator()

        # Initialize agent
        self.initialize_agent()

        if self.verbose:
            print("\n" + "=" * 50)
            print(f"Running {self.__class__.__name__} Exploration")
            print("=" * 50)

        # Initialize estimation tracking arrays
        self.goal_density_arr = np.zeros((len(self.pcloud.vertices), self.timesteps))
        self.estimated_density_arr = np.zeros_like(self.goal_density_arr)

        # Sample initial density at agent's starting position
        initial_sample = self.density_estimator.sample_density_at_point(self.agent.x)

        # Create online estimator
        self.online_estimator = self.density_estimator.create_online_estimator(
            initial_trajectory=self.agent.x.reshape(1, -1),
            initial_samples=np.array([initial_sample]),
        )

        # Get initial GP prediction
        gp_mean = self.online_estimator.predict(return_variance=False)
        gp_mean = normalize_mat(gp_mean)

        # Compute initial goal density (just mean, no variance yet)
        goal_density = gp_mean.copy()

        # Initialize agent time counter
        self.agent.t = 0

        # Main exploration loop with GP updates
        for t in range(self.timesteps):
            # Update GP and recompute goal density periodically
            if t > 0 and t % self.gp_update_interval == 0:
                # Sample densities along trajectory (every 5th point for efficiency)
                trajectory_samples = self.agent.x_arr[:t:5, :]
                density_samples = self.density_estimator.sample_density_at_points(
                    trajectory_samples
                )

                # Replace training data (more efficient than incremental updates)
                self.online_estimator.replace_training_data(
                    trajectory_samples, density_samples
                )

                # Get GP prediction with variance
                gp_mean, gp_variance = self.online_estimator.predict(return_variance=True)
                gp_mean = normalize_mat(gp_mean)
                gp_variance = normalize_mat(gp_variance)

                # Compute new goal density (algorithm-specific)
                # Pass coverage if the algorithm tracks it
                if hasattr(self, 'coverage'):
                    goal_density = self.compute_goal_density(gp_mean, gp_variance, self.coverage)
                else:
                    goal_density = self.compute_goal_density(gp_mean, gp_variance, None)

                # Store estimated density
                self.estimated_density_arr[:, t] = gp_mean

                if self.verbose and t % 100 == 0:
                    print(f"GP update at step {t}")

            # Run algorithm-specific exploration step
            step_results = self.run_exploration_step(t, goal_density)

            # Store goal density
            self.goal_density_arr[:, t] = goal_density

            # Progress update
            if self.verbose and (t + 1) % 100 == 0:
                print(f"Step {t+1}/{self.timesteps}")

        if self.verbose:
            print("\n" + "=" * 50)
            print("Exploration Complete!")
            print("=" * 50)

        # Return results (trajectory + algorithm-specific data + estimation data)
        results = {
            "trajectory": self.agent.x_arr[:self.agent.t, :],  # Shape: (timesteps, 3)
            "pcloud": self.pcloud,
            "goal_density": self.goal_density_arr,
            "estimated_density": self.estimated_density_arr,
        }

        return results
