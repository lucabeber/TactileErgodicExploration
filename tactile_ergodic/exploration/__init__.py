"""Exploration algorithm base classes and controllers."""

from .exploration_base import ExplorationWithGP
from .laplacian_smc import (
    LaplacianSMCController,
    compute_laplacian_eigenpairs,
    normalize_density_from_coeffs,
)
from .ergodic_control_uv import (
    ErgodicControlUV,
    compute_uv_parameterization_pca,
    create_uv_interpolator,
    setup_fourier_basis,
)

__all__ = [
    "ExplorationWithGP",
    "LaplacianSMCController",
    "compute_laplacian_eigenpairs",
    "normalize_density_from_coeffs",
    "ErgodicControlUV",
    "compute_uv_parameterization_pca",
    "create_uv_interpolator",
    "setup_fourier_basis",
]
