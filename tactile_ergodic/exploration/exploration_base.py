"""
Base class for exploration algorithms with GP-based density estimation.

This module provides a common base class that handles point cloud setup
and GP density estimation, shared across different exploration methods
(HEDAC, SMC, BO, etc.).

Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>
"""

import numpy as np
import torch
from abc import ABC, abstractmethod

from ..estimation.density_estimator import create_density_estimator
from ..utils.pointcloud_utils import process_point_cloud


class ExplorationWithGP(ABC):
    """
    Abstract base class for exploration algorithms with GP density estimation.

    This class provides common functionality for:
    - Point cloud loading and processing
    - GP-based density estimation setup
    - Configuration management

    Subclasses must implement the explore() method with their specific
    exploration strategy.
    """

    def __init__(
        self,
        voxel_size=0.002,
        exploit_alpha=0.6,
        gp_update_interval=50,
    ):
        """
        Initialize base exploration parameters.

        Args:
            voxel_size: Voxel size for point cloud downsampling
            exploit_alpha: Balance between exploration (0) and exploitation (1)
            gp_update_interval: How often to update GP model (every N timesteps)
        """
        self.voxel_size = voxel_size
        self.exploit_alpha = exploit_alpha
        self.gp_update_interval = gp_update_interval

        # These will be set during setup
        self.pcloud = None
        self.density_estimator = None

    def setup_point_cloud(self, filename, additional_params=None):
        """
        Load and setup the point cloud for exploration.

        Args:
            filename: Path to point cloud file
            additional_params: Dictionary with additional parameters specific
                             to the exploration method (e.g., alpha for HEDAC)
        """
        # Create parameter object for process_point_cloud
        class param:
            pass

        param.voxel_size = self.voxel_size

        # Add any method-specific parameters
        if additional_params is not None:
            for key, value in additional_params.items():
                setattr(param, key, value)

        self.pcloud = process_point_cloud(filename, param)
        print(f"Point cloud processed: {len(self.pcloud.vertices)} points")

        # Call hook for subclass-specific setup
        self._setup_point_cloud_hook()

    def _setup_point_cloud_hook(self):
        """
        Hook for subclasses to add additional point cloud setup.

        Override this method in subclasses to add method-specific
        point cloud processing (e.g., Laplacian matrices for HEDAC,
        UV parameterization for SMC).
        """
        pass

    def setup_density_estimator(self, gp_params=None):
        """
        Setup the GP-based density estimator.

        Args:
            gp_params: Dictionary with GP hyperparameters (l, sigma, n_eig)
        """
        if gp_params is None:
            gp_params = {"l": 0.010, "sigma": 1.0, "n_eig": 500}

        print("\n" + "=" * 50)
        print("Initializing Density Estimator")
        print("=" * 50)

        self.density_estimator = create_density_estimator(
            self.pcloud, self.pcloud.u0, gp_params, dtype=torch.float32
        )

        print("=" * 50 + "\n")

    @abstractmethod
    def explore(self, *args, **kwargs):
        """
        Perform exploration using the specific algorithm.

        This method must be implemented by subclasses with their
        specific exploration strategy (HEDAC, SMC, BO, etc.).

        Returns:
            Results dictionary or tuple containing exploration outcomes
        """
        raise NotImplementedError("Subclasses must implement explore()")
