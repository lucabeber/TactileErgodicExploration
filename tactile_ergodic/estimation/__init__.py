"""Density estimation modules using Gaussian Processes."""

from .density_estimator import DensityEstimator, OnlineGPEstimator
from .gpr_on_point_cloud import GPROnPointCloud, rbf_manifold_kernel

__all__ = [
    "DensityEstimator",
    "OnlineGPEstimator",
    "GPROnPointCloud",
    "rbf_manifold_kernel",
]
