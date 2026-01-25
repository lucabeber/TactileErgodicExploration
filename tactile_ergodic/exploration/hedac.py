"""
HEDAC (Heat Equation Driven Autonomous Coverage) Exploration

This module implements pure HEDAC exploration without GP estimation.
The target distribution comes from the point cloud's color/texture data (red channel).

Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>
"""

import time

import numpy as np
import robust_laplacian
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from .pointcloud_exploration import PointcloudExploration
from ..utils import Pointcloud, PointcloudScalarDiffusion
from ..utils.pointcloud_utils import (
    calculate_dt,
    get_pcloud_neighbors,
    get_gradient,
    normalize_mat,
)


class PointcloudHEDAC(PointcloudExploration):
    """
    Heat Equation Driven Autonomous Coverage (HEDAC) exploration.

    HEDAC uses a heat diffusion equation on the point cloud manifold to generate
    exploration gradients. The agent follows the heat gradient to achieve coverage
    that matches a target distribution.

    Key components:
    - Heat equation solver (Laplacian + mass matrix)
    - Coverage tracking
    - Source term computation (goal - coverage)
    - Gradient computation on manifold
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
        **kwargs
    ):
        """
        Initialize HEDAC exploration.

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
            **kwargs: Additional arguments for PointcloudExploration
        """
        super().__init__(
            obj_name=obj_name,
            voxel_size=voxel_size,
            timesteps=timesteps,
            agent_type="first_order",  # HEDAC uses first-order agent
            agent_start_index=agent_start_index,
            **kwargs
        )

        # HEDAC-specific parameters
        self.alpha = alpha
        self.source_strength = source_strength
        self.nb_max_neighbors = nb_max_neighbors
        self.nb_minimum_neighbors = nb_minimum_neighbors
        self.nb_boundary_neighbors = nb_boundary_neighbors

        # HEDAC-specific attributes (set during setup)
        self.pcd_helper = None
        self.scalar_diffusion_solver = None
        self.boundary_normals = None

        # Tracking arrays
        self.coverage = None
        self.coverage_arr = None
        self.heat_arr = None
        self.speed_arr = None
        self.time_arr = None
        self.ut = None  # Current heat field
        self.goal_density = None

    def setup_point_cloud(self, additional_params=None):
        """Setup point cloud with HEDAC-specific parameters."""
        # Add alpha parameter for dt calculation
        params = {'alpha': self.alpha}
        if additional_params:
            params.update(additional_params)
        super().setup_point_cloud(additional_params=params)

    def setup_algorithm(self):
        """
        HEDAC-specific setup: heat equation solver and Laplacian matrices.
        """
        # Calculate dt and h for heat equation
        self.pcloud.dt, self.pcloud.h = calculate_dt(self.pcloud.vertices, self.alpha)

        if self.verbose:
            print(f"dt: {self.pcloud.dt:.3e}, h: {self.pcloud.h:.3e}, s: {self.voxel_size:.3e}")

        # Setup point cloud helper
        self.pcd_helper = Pointcloud(self.pcloud.vertices)

        # Get boundary normals (this also sets is_boundary_arr attribute)
        self.boundary_normals = self.pcd_helper.get_boundary_normals()

        # Setup heat equation solver
        self.scalar_diffusion_solver = PointcloudScalarDiffusion(pcloud=self.pcd_helper)

        # Setup Laplacian matrices
        self.pcloud.C, self.pcloud.M = robust_laplacian.point_cloud_laplacian(
            self.pcloud.vertices, n_neighbors=self.nb_boundary_neighbors
        )
        A = csc_matrix(self.pcloud.M + self.pcloud.dt * self.pcloud.C)
        self.pcloud.A_factorized = splu(A)

        # Initialize goal density from red channel
        self.goal_density = normalize_mat(self.pcloud.u0.copy())

        if self.verbose:
            print(
                f"Goal density from red channel: "
                f"min={self.goal_density.min():.3f}, max={self.goal_density.max():.3f}"
            )

        # Initialize tracking arrays
        self.coverage_arr = np.zeros((len(self.pcloud.vertices), self.timesteps))
        self.heat_arr = np.zeros_like(self.coverage_arr)
        self.speed_arr = np.zeros(self.timesteps)
        self.time_arr = np.zeros(self.timesteps)

        # Initialize coverage and heat field
        self.coverage = np.zeros_like(self.goal_density)
        self.ut = np.array(self.goal_density)

    def run_exploration_step(self, t, goal_density=None):
        """
        Execute one HEDAC exploration step.

        Args:
            t: Current timestep
            goal_density: Target distribution (if None, uses self.goal_density)

        Returns:
            dict: Step results with coverage and heat values
        """
        # Use provided goal density or default
        if goal_density is None:
            goal_density = self.goal_density

        # Get neighbors around agent
        dists, neighbor_ids, neighbor_coords = get_pcloud_neighbors(
            self.pcloud.pcd_tree,
            self.pcloud.vertices,
            np.copy(self.agent.x),
            self.agent.radius,
            self.nb_max_neighbors,
            self.nb_minimum_neighbors,
        )

        # Update coverage using Gaussian kernel
        kernel_vals = np.exp(-(1 / self.agent.radius) * dists**2)
        self.coverage[neighbor_ids] += kernel_vals

        # Compute gradient direction using closer neighbors
        neighbor_ids = neighbor_ids[:self.nb_minimum_neighbors]
        dists = dists[:self.nb_minimum_neighbors]
        neighbor_coords = self.pcloud.vertices[neighbor_ids, :]

        # Normalize coverage and compute source term
        coverage_density = normalize_mat(self.coverage)
        source = np.maximum(goal_density - coverage_density, 0) ** 2
        source = normalize_mat(source)

        # Solve heat equation
        start_time = time.time()
        self.ut = self.pcloud.A_factorized.solve(self.pcloud.M @ self.ut)
        self.time_arr[t] = time.time() - start_time

        # Add source term and enforce boundary conditions
        self.ut += self.source_strength * source
        self.ut[self.pcd_helper.is_boundary_arr] = 0

        # Compute gradient field
        self.scalar_diffusion_solver.get_gradient(self.ut)

        # Project agent to surface before computing gradient
        (self.agent.x,) = get_gradient(
            np.copy(self.agent.x),
            neighbor_coords,
            neighbor_ids,
            self.ut,
        )

        # Get gradient at agent location
        gradient = np.mean(
            self.scalar_diffusion_solver.gradient_ut_3d[neighbor_ids[:10]], axis=0
        )

        # Track speed before update
        prev_x = np.copy(self.agent.x)
        self.agent.update(gradient)
        displacement = self.agent.x - prev_x
        self.speed_arr[t] = np.linalg.norm(displacement)

        # Project agent back to surface after update (critical for staying on manifold!)
        dists_proj, neighbor_ids_proj, neighbor_coords_proj = get_pcloud_neighbors(
            self.pcloud.pcd_tree,
            self.pcloud.vertices,
            np.copy(self.agent.x),
            self.agent.radius,
            self.nb_max_neighbors,
            self.nb_minimum_neighbors,
        )
        neighbor_ids_proj = neighbor_ids_proj[:self.nb_minimum_neighbors]
        neighbor_coords_proj = self.pcloud.vertices[neighbor_ids_proj, :]

        (self.agent.x,) = get_gradient(
            np.copy(self.agent.x),
            neighbor_coords_proj,
            neighbor_ids_proj,
            self.ut,
        )

        # Store results
        self.coverage_arr[:, t] = self.coverage
        self.heat_arr[:, t] = self.ut

        # Progress update with more detail
        if self.verbose and (t + 1) % 100 == 0:
            print(
                f"Step {t+1}/{self.timesteps}, speed: {self.speed_arr[t]:.3e}, "
                f"solve time: {self.time_arr[t]:.3e}s"
            )

        return {
            "coverage": self.coverage.copy(),
            "heat": self.ut.copy(),
        }

    def explore(self, goal_density=None):
        """
        Run HEDAC exploration.

        Args:
            goal_density: Optional target distribution (uses red channel if None)

        Returns:
            dict: Results including trajectory, coverage, heat, etc.
        """
        # Run base exploration
        results = super().explore(goal_density=goal_density)

        # Add HEDAC-specific results
        results.update({
            "heat": self.heat_arr,
            "coverage": self.coverage_arr,
            "time": self.time_arr,
            "goal_density": self.goal_density,
            "speed": self.speed_arr,
        })

        if self.verbose:
            print(f"Average speed: {self.speed_arr.mean():.3e}")
            print(f"Average solve time: {self.time_arr.mean():.3e}s")

        return results
