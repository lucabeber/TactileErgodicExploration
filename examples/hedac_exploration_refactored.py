"""
HEDAC (Heat Equation Driven Autonomous Coverage) Exploration - Refactored Version

This version uses the density_estimator module for GP-based density estimation,
making the code more modular and reusable across different exploration methods.

Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>
"""

import numpy as np

np.set_printoptions(formatter={"float": lambda x: "{0:0.3e}".format(x)})

import time

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)
torch.set_default_device(device)

import robust_laplacian
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

import sys
sys.path.insert(0, '..')

from tactile_ergodic.utils import config
from tactile_ergodic.exploration import ExplorationWithGP
from tactile_ergodic.utils.plotting_utils import *
from tactile_ergodic.utils import Pointcloud, PointcloudScalarDiffusion, FirstOrderAgent, SecondOrderAgent
from tactile_ergodic.utils.pointcloud_utils import *


class HEDAC(ExplorationWithGP):
    """
    Heat Equation Driven Autonomous Coverage (HEDAC) exploration algorithm.

    This class encapsulates the HEDAC algorithm with GP-based density estimation.
    Inherits common GP and point cloud setup from ExplorationWithGP.
    """

    def __init__(
        self,
        exploit_alpha=0.6,
        timesteps=1500,
        alpha=100,
        voxel_size=0.002,
        source_strength=1,
        nb_max_neighbors=500,
        nb_minimum_neighbors=20,
        nb_boundary_neighbors=40,
        gp_update_interval=50,
    ):
        """
        Initialize HEDAC parameters.

        Args:
            exploit_alpha: Balance between exploration (0) and exploitation (1)
            timesteps: Number of exploration timesteps
            alpha: Heat equation parameter
            voxel_size: Voxel size for point cloud downsampling
            source_strength: Strength of heat equation source term
            nb_max_neighbors: Maximum number of neighbors to consider
            nb_minimum_neighbors: Minimum number of neighbors required
            nb_boundary_neighbors: Number of neighbors for boundary detection
            gp_update_interval: How often to update GP model (every N timesteps)
        """
        # Initialize base class
        super().__init__(
            voxel_size=voxel_size,
            exploit_alpha=exploit_alpha,
            gp_update_interval=gp_update_interval,
        )

        # HEDAC-specific parameters
        self.timesteps = timesteps
        self.alpha = alpha
        self.agent_radius = 2.5 * voxel_size
        self.max_velocity = 0.1 * voxel_size * 2
        self.max_acceleration = 1.0 * self.max_velocity * 2
        self.source_strength = source_strength
        self.nb_max_neighbors = nb_max_neighbors
        self.nb_minimum_neighbors = nb_minimum_neighbors
        self.nb_boundary_neighbors = nb_boundary_neighbors

        # HEDAC-specific attributes (set during setup)
        self.pcd_helper = None
        self.scalar_diffusion_solver = None

    def _setup_point_cloud_hook(self):
        """
        HEDAC-specific point cloud setup: Laplacian matrices and heat equation solver.
        """
        # Calculate dt and h for heat equation (HEDAC-specific)
        from pointcloud_utils import calculate_dt
        self.pcloud.dt, self.pcloud.h = calculate_dt(self.pcloud.vertices, self.alpha)
        print(f"dt: {self.pcloud.dt:.3e}, h: {self.pcloud.h:.3e}, s: {self.voxel_size:.3e}")

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

    def explore(self, agent):
        """
        Perform HEDAC exploration with online GP density estimation.

        Args:
            agent: Virtual agent for exploration

        Returns:
            tuple: (trajectory, heat_array, coverage_array, time_array,
                    goal_density_array, estimated_density_array)
        """
        # Initialize arrays
        coverage_arr = np.zeros((len(self.pcloud.vertices), self.timesteps))
        heat_arr = np.zeros_like(coverage_arr)
        goal_density_arr = np.zeros_like(coverage_arr)
        estimated_density_arr = np.zeros_like(coverage_arr)
        speed_arr = np.zeros(self.timesteps)

        # Sample initial density at agent's starting position
        initial_sample = self.density_estimator.sample_density_at_point(agent.x)

        # Create online estimator
        online_estimator = self.density_estimator.create_online_estimator(
            initial_trajectory=agent.x.reshape(1, -1),
            initial_samples=np.array([initial_sample]),
        )

        # Get initial goal density (use only mean, not mean+variance yet)
        # This matches the original HEDAC behavior
        mean_tmp = online_estimator.predict(return_variance=False)
        mean_tmp = normalize_mat(mean_tmp)
        goal_density = mean_tmp.copy()  # Initial goal is just the mean

        # Initialize coverage
        coverage = np.zeros_like(goal_density)
        ut = np.array(goal_density)
        time_arr = np.zeros(self.timesteps)

        agent.t = 0

        # Main exploration loop
        for t in range(self.timesteps):
            # Get neighbors
            dists, neighbor_ids, neighbor_coords = get_pcloud_neighbors(
                self.pcloud.pcd_tree,
                self.pcloud.vertices,
                np.copy(agent.x),
                agent.radius,
                self.nb_max_neighbors,
                self.nb_minimum_neighbors,
            )

            # Update coverage
            kernel_vals = np.exp(-(1 / agent.radius) * dists**2)
            coverage[neighbor_ids] += kernel_vals

            # Compute gradient direction
            neighbor_ids = neighbor_ids[: self.nb_minimum_neighbors]
            dists = dists[: self.nb_minimum_neighbors]
            neighbor_coords = self.pcloud.vertices[neighbor_ids, :]

            coverage_density = normalize_mat(coverage)
            source = np.maximum(goal_density - coverage_density, 0) ** 2
            source = normalize_mat(source)

            start_time = time.time()
            ut = self.pcloud.A_factorized.solve(self.pcloud.M @ ut)
            time_arr[t] = time.time() - start_time

            ut += self.source_strength * source
            ut[self.pcd_helper.is_boundary_arr] = 0
            self.scalar_diffusion_solver.get_gradient(ut)

            # Get gradient at agent location and project agent back to surface
            (agent.x,) = get_gradient(
                np.copy(agent.x),
                neighbor_coords,
                neighbor_ids,
                ut,
            )

            gradient = np.mean(
                self.scalar_diffusion_solver.gradient_ut_3d[neighbor_ids[:10]], axis=0
            )

            # Track speed
            prev_x = np.copy(agent.x)
            agent.update(gradient)
            displacement = agent.x - prev_x
            speed_arr[t] = np.linalg.norm(displacement)

            # Store data
            coverage_arr[..., t] = coverage
            heat_arr[..., t] = np.copy(ut)
            goal_density_arr[..., t] = goal_density
            estimated_density_arr[..., t] = mean_tmp

            # Update GP estimate periodically
            if t % self.gp_update_interval == 0 and t > 0:
                print(f"Time step: {t}/{self.timesteps}")

                # Sample densities along trajectory (batched for speed)
                trajectory_samples = agent.x_arr[:t:5, :]
                density_samples = self.density_estimator.sample_density_at_points(
                    trajectory_samples
                )

                # Replace training data to match original HEDAC behavior
                online_estimator.replace_training_data(trajectory_samples, density_samples)

                # Update goal density
                goal_density = online_estimator.get_goal_density(
                    exploit_alpha=self.exploit_alpha,
                    nb_boundary_neighbors=self.nb_boundary_neighbors,
                )

                mean_tmp = online_estimator.predict(return_variance=False)
                mean_tmp = normalize_mat(mean_tmp)

        # Print speed statistics
        print("\n" + "=" * 50)
        print("3D Speed Statistics:")
        print("=" * 50)
        print(f"Min speed:     {np.min(speed_arr):.6e}")
        print(f"Max speed:     {np.max(speed_arr):.6e}")
        print(f"Average speed: {np.mean(speed_arr):.6e}")
        print(f"Median speed:  {np.median(speed_arr):.6e}")
        print(f"Std dev:       {np.std(speed_arr):.6e}")
        print("=" * 50 + "\n")

        return (
            agent.x_arr,
            heat_arr,
            coverage_arr,
            time_arr,
            goal_density_arr,
            estimated_density_arr,
        )


if __name__ == "__main__":
    # Select the object to explore
    obj_name = "bun270_X"  # Stanford bunny with X projected as the target
    # obj_name = "plate_shapes"  # random IKEA plate with hand-drawn shapes
    # obj_name = "cup_X"

    # Initialize HEDAC algorithm
    hedac = HEDAC(
        exploit_alpha=0.6,
        timesteps=1500,
        alpha=100,
        voxel_size=0.002,
        source_strength=1,
        nb_max_neighbors=500,
        nb_minimum_neighbors=20,
        nb_boundary_neighbors=40,
        gp_update_interval=50,
    )

    # Setup point cloud and density estimator
    filename = config.get_point_cloud_path(f"{obj_name}.ply")
    hedac.setup_point_cloud(filename)

    gp_params = {"l": 0.010, "sigma": 1.0, "n_eig": 500}
    hedac.setup_density_estimator(gp_params)

    # Initialize agent
    agent = FirstOrderAgent(
        x=np.zeros(3),
        dim_t=hedac.timesteps,
        max_velocity=hedac.max_velocity,
    )
    agent.x = hedac.pcloud.vertices[1500]
    agent.radius = hedac.agent_radius

    # Run exploration
    print("Starting HEDAC exploration...")
    x_arr, heat_arr, coverage_arr, time_arr, goal_arr, estimated_density_arr = (
        hedac.explore(agent)
    )

    # Visualize results
    plots = visualize_point_cloud(
        hedac.pcloud.vertices,
        colors=estimated_density_arr[..., -1],
        is_show_plot=False,
        point_size=5,
    )
    fig = visualize_trajectory(x_arr[:, :], plots, color="black")
    fig.show("browser")

    # Generate animations
    print("\nGenerating animated visualizations...")
    goal_html_path = config.get_animation_path(f"hedac_refactored_goal_{obj_name}.html")
    animate_trajectory_pcloud(
        x_arr=x_arr,
        vertices=hedac.pcloud.vertices,
        color_frames=goal_arr,
        timesteps=hedac.timesteps,
        save_path=str(goal_html_path),
    )
    print(f"Goal density animation saved to {goal_html_path}")

    est_html_path = config.get_animation_path(
        f"hedac_refactored_estimated_{obj_name}.html"
    )
    animate_trajectory_pcloud(
        x_arr=x_arr,
        vertices=hedac.pcloud.vertices,
        color_frames=estimated_density_arr,
        timesteps=hedac.timesteps,
        save_path=str(est_html_path),
    )
    print(f"Estimated density animation saved to {est_html_path}")

    # Plot distribution evolution
    plot_distribution_evolution_column_auto(
        vertices=hedac.pcloud.vertices,
        original_density=hedac.density_estimator.get_initial_mean(),
        estimated_density_arr=estimated_density_arr,
        pdf_name=str(config.get_plot_path(f"hedac_refactored_{obj_name}.pdf")),
        agent_trajectory=x_arr[:, :],
    )
    print("\nExploration complete!")
