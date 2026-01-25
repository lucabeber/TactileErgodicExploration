"""Exploration algorithm base classes and controllers."""

# Base classes
from .pointcloud_exploration import PointcloudExploration
from .exploration_base import ExplorationWithGP  # Legacy, kept for compatibility
from .estimation_mixin import ExplorationWithEstimation

# Pure exploration algorithms (no GP estimation)
from .hedac import PointcloudHEDAC

# Exploration with GP estimation
from .hedac_estimation import HEDACEstimation

# Laplacian SMC components
from .laplacian_smc import (
    LaplacianSMCController,
    compute_laplacian_eigenpairs,
    normalize_density_from_coeffs,
)

# UV SMC components
from .ergodic_control_uv import (
    ErgodicControlUV,
    compute_uv_parameterization_pca,
    create_uv_interpolator,
    setup_fourier_basis,
)

__all__ = [
    # Base classes
    "PointcloudExploration",
    "ExplorationWithGP",  # Legacy
    "ExplorationWithEstimation",
    # Pure exploration
    "PointcloudHEDAC",
    # Exploration with estimation
    "HEDACEstimation",
    # Laplacian SMC
    "LaplacianSMCController",
    "compute_laplacian_eigenpairs",
    "normalize_density_from_coeffs",
    # UV SMC
    "ErgodicControlUV",
    "compute_uv_parameterization_pca",
    "create_uv_interpolator",
    "setup_fourier_basis",
]
