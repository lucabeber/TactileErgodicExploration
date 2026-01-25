"""
HEDAC Exploration with GP Density Estimation

This module implements HEDAC exploration with online GP-based density estimation.
It combines PointcloudHEDAC exploration with ExplorationWithEstimation mixin.

Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>
"""

import numpy as np

from .hedac import PointcloudHEDAC
from .estimation_mixin import ExplorationWithEstimation
from ..utils.pointcloud_utils import normalize_mat


class HEDACEstimation(PointcloudHEDAC, ExplorationWithEstimation):
    """
    HEDAC exploration with online GP density estimation.

    This class combines:
    - PointcloudHEDAC: Heat equation-based exploration algorithm
    - ExplorationWithEstimation: GP-based target density estimation

    The agent explores to match a target distribution that is learned online
    via Gaussian Process regression. The target balances exploitation (GP mean)
    and exploration (GP variance) using the exploit_alpha parameter.

    Attributes inherited from PointcloudHEDAC:
        - Heat equation solver and Laplacian matrices
        - Coverage tracking
        - HEDAC-specific parameters (alpha, source_strength, etc.)

    Attributes inherited from ExplorationWithEstimation:
        - GP density estimator
        - Online estimator for incremental updates
        - exploit_alpha for exploration-exploitation balance
        - gp_update_interval for controlling update frequency
    """

    def __init__(
        self,
        obj_name="bun270_X",
        voxel_size=0.002,
        timesteps=1500,
        alpha=100,
        source_strength=1,
        nb_max_neighbors=500,
        nb_minimum_neighbors=20,
        nb_boundary_neighbors=40,
        agent_start_index=1500,
        exploit_alpha=0.6,
        gp_update_interval=50,
        gp_params=None,
        **kwargs
    ):
        """
        Initialize HEDAC exploration with GP estimation.

        Args:
            obj_name: Point cloud object name
            voxel_size: Voxel size for downsampling
            timesteps: Number of exploration steps
            alpha: Heat equation parameter (controls diffusion rate)
            source_strength: Strength of heat source term
            nb_max_neighbors: Maximum neighbors for k-NN queries
            nb_minimum_neighbors: Minimum neighbors required
            nb_boundary_neighbors: Neighbors for boundary detection
            agent_start_index: Vertex index where agent starts
            exploit_alpha: Balance between exploration (0) and exploitation (1)
            gp_update_interval: How often to update GP model (every N timesteps)
            gp_params: GP hyperparameters dictionary (l, sigma, n_eig)
            **kwargs: Additional arguments for base classes
        """
        # Initialize PointcloudHEDAC first (which calls PointcloudExploration)
        PointcloudHEDAC.__init__(
            self,
            obj_name=obj_name,
            voxel_size=voxel_size,
            timesteps=timesteps,
            alpha=alpha,
            source_strength=source_strength,
            nb_max_neighbors=nb_max_neighbors,
            nb_minimum_neighbors=nb_minimum_neighbors,
            nb_boundary_neighbors=nb_boundary_neighbors,
            agent_start_index=agent_start_index,
            **kwargs
        )

        # Initialize ExplorationWithEstimation attributes separately
        # (can't call __init__ because it would conflict with PointcloudExploration)
        self.exploit_alpha = exploit_alpha
        self.gp_update_interval = gp_update_interval

        # Set default GP parameters if not provided
        if gp_params is None:
            gp_params = {"l": 0.010, "sigma": 1.0, "n_eig": 500}
        self.gp_params = gp_params

        self.density_estimator = None
        self.online_estimator = None
        self.goal_density_arr = None
        self.estimated_density_arr = None

    def compute_goal_density(self, gp_mean, gp_variance, coverage):
        """
        Compute HEDAC goal density from GP estimates.

        HEDAC uses exploration-exploitation balance with mean-centering:
        goal = exploit_alpha * mean_centered + (1 - exploit_alpha) * variance

        This encourages the agent to explore areas with high uncertainty
        (high variance) while also exploiting known high-density regions
        (high mean).

        Args:
            gp_mean: GP mean prediction (N,)
            gp_variance: GP variance prediction (N,)
            coverage: Current coverage array (N,) - not used by HEDAC

        Returns:
            np.ndarray: Normalized goal density (N,)
        """
        # Mean-center the GP mean and clip negative values
        mean_centered = np.maximum(gp_mean - np.mean(gp_mean), 0)
        mean_norm = normalize_mat(mean_centered)
        var_norm = normalize_mat(gp_variance)

        # Combine with exploit_alpha weighting
        goal = self.exploit_alpha * mean_norm + (1 - self.exploit_alpha) * var_norm
        return normalize_mat(goal)

    def explore(self):
        """
        Run HEDAC exploration with GP estimation.

        This calls the mixin's explore() to add HEDAC-specific results
        to the output dictionary.

        Returns:
            dict: Results including trajectory, coverage, heat, GP estimates, etc.
        """
        # Call ExplorationWithEstimation.explore() explicitly
        results = ExplorationWithEstimation.explore(self)

        # Add HEDAC-specific results (if they exist)
        if hasattr(self, 'heat_arr') and self.heat_arr is not None:
            results.update({
                "heat": self.heat_arr,
                "coverage": self.coverage_arr,
                "time": self.time_arr,
                "speed": self.speed_arr,
            })

            if self.verbose:
                print(f"Average speed: {self.speed_arr.mean():.3e}")
                print(f"Average solve time: {self.time_arr.mean():.3e}s")

        return results
