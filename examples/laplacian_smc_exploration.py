"""
Laplacian SMC Ergodic Exploration with GP-based Density Estimation

This script combines:
1. Laplacian SMC ergodic control (from laplacian_smc.py)
2. GP-based density estimation (from density_estimator.py)
3. Active learning framework (similar to HEDAC and UV-SMC)

The controller operates directly on the 3D surface using Laplacian eigenfunctions
for computational efficiency, while the GP learns the target distribution online.

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

import sys
sys.path.insert(0, '..')

from tactile_ergodic.utils import config
from tactile_ergodic.exploration import ExplorationWithGP
from tactile_ergodic.exploration.laplacian_smc import (
    LaplacianSMCController,
    compute_laplacian_eigenpairs,
    normalize_density_from_coeffs,
)
from tactile_ergodic.utils.plotting_utils import *
from tactile_ergodic.utils.pointcloud_utils import *


class LaplacianSMCExplorerGP(ExplorationWithGP):
    """
    Laplacian SMC-based ergodic exploration with GP density estimation.

    This class combines Laplacian eigenfunction-based ergodic control
    with online GP learning to actively explore unknown density distributions
    on point clouds. Inherits common GP and point cloud setup from ExplorationWithGP.
    """

    def __init__(
        self,
        nbData=1200,
        num_eigen=200,
        dt=0.05,
        u_max=0.1,
        rho_u=1e-3,
        knn=30,
        voxel_size=0.002,
        exploit_alpha=0.6,
        gp_update_interval=50,
        lambda_weight="mezic",
        alpha_fixed=None,
    ):
        """
        Initialize Laplacian SMC exploration parameters.

        Args:
            nbData: Number of timesteps for exploration
            num_eigen: Number of Laplacian eigenfunctions to use
            dt: Time step for SMC controller
            u_max: Maximum speed
            rho_u: Control regularization parameter
            knn: Number of neighbors for local tangent plane estimation
            voxel_size: Voxel size for point cloud downsampling
            exploit_alpha: Balance between exploration (0) and exploitation (1)
            gp_update_interval: How often to update GP model (every N timesteps)
            lambda_weight: Eigenvalue weighting scheme ("exp", "mezic", or "smc")
            alpha_fixed: Fixed alpha for exponential forgetting (None for 1/(t+1))
        """
        # Initialize base class
        super().__init__(
            voxel_size=voxel_size,
            exploit_alpha=exploit_alpha,
            gp_update_interval=gp_update_interval,
        )

        # Laplacian SMC-specific parameters
        self.nbData = nbData
        self.num_eigen = num_eigen
        self.dt = dt
        self.u_max = u_max
        self.rho_u = rho_u
        self.knn = knn
        self.lambda_weight = lambda_weight
        self.alpha_fixed = alpha_fixed

        # Laplacian SMC-specific attributes (set during setup)
        self.lambdas = None
        self.eigenvecs = None
        self.M = None
        self.controller = None
        self.online_estimator = None

    def _setup_point_cloud_hook(self):
        """
        Laplacian SMC-specific point cloud setup: compute Laplacian eigenpairs.
        """
        print(f"Computing {self.num_eigen} Laplacian eigenpairs...")
        self.lambdas, self.eigenvecs, self.M = compute_laplacian_eigenpairs(
            self.pcloud.vertices, self.num_eigen
        )
        print(f"Eigenvalues range: [{self.lambdas[0]:.6e}, {self.lambdas[-1]:.6e}]")

    def explore(self, x0=None, seed=0):
        """
        Perform Laplacian SMC exploration with online GP density estimation.

        This method runs Laplacian SMC control with periodic GP updates to adapt
        the target distribution based on learned information.

        Args:
            x0: Initial position in 3D. If None, randomly selected from point cloud
            seed: Random seed for initial position selection

        Returns:
            dict: Exploration results containing trajectories and statistics
        """
        rng = np.random.default_rng(seed)

        # Select initial position
        if x0 is None:
            x0 = self.pcloud.vertices[rng.integers(len(self.pcloud.vertices))]
        else:
            x0 = np.asarray(x0, dtype=float).reshape(3)

        # Sample initial density at starting position
        initial_sample = self.density_estimator.sample_density_at_point(x0)

        # Create online estimator
        self.online_estimator = self.density_estimator.create_online_estimator(
            initial_trajectory=x0.reshape(1, -1),
            initial_samples=np.array([initial_sample]),
        )

        # Get initial goal density (use only mean initially)
        mean_tmp = self.online_estimator.predict(return_variance=False)
        mean_tmp = normalize_mat(mean_tmp)
        goal_density = mean_tmp.copy()

        # Storage arrays for trajectory and densities over time
        trajectory_3d = np.zeros((self.nbData + 1, 3), dtype=float)
        trajectory_3d[0] = x0
        goal_density_arr = np.zeros((len(self.pcloud.vertices), self.nbData + 1))
        estimated_density_arr = np.zeros((len(self.pcloud.vertices), self.nbData + 1))
        goal_density_arr[:, 0] = goal_density
        estimated_density_arr[:, 0] = mean_tmp

        # Track controller metrics
        ergodic_metric_arr = np.zeros(self.nbData + 1)
        speed_arr = np.zeros(self.nbData + 1)

        print("Starting Laplacian SMC exploration with GP updates...")

        # Run in chunks to allow periodic GP updates
        chunk_size = self.gp_update_interval
        num_chunks = (self.nbData + chunk_size - 1) // chunk_size

        current_x = x0.copy()

        for chunk_idx in range(num_chunks):
            # Determine timesteps for this chunk
            start_t = chunk_idx * chunk_size
            end_t = min((chunk_idx + 1) * chunk_size, self.nbData)
            chunk_steps = end_t - start_t

            if chunk_idx > 0:
                print(f"\nTime step: {start_t}/{self.nbData}")

            # Create controller with current goal density
            controller = LaplacianSMCController(
                points_N3=self.pcloud.vertices,
                M_NN=self.M,
                F_NK=self.eigenvecs,
                lambdas_K=self.lambdas,
                phi_N=goal_density,
                dt=self.dt,
                u_max=self.u_max,
                rho_u=self.rho_u,
                knn=self.knn,
                alpha=self.alpha_fixed,
                lambda_weight=self.lambda_weight,
            )

            # Run controller for this chunk
            for t_offset in range(chunk_steps):
                t = start_t + t_offset + 1  # +1 because t=0 is initial position

                x_next, u, dbg = controller.step(current_x)

                # Store results
                trajectory_3d[t] = x_next
                goal_density_arr[:, t] = goal_density
                estimated_density_arr[:, t] = mean_tmp
                ergodic_metric_arr[t] = dbg["E"]
                speed_arr[t] = dbg["u_norm"]

                current_x = x_next

            # Update GP with trajectory so far (if not the last chunk)
            if chunk_idx < num_chunks - 1:
                # Sample densities along trajectory (every 5th point)
                trajectory_samples = trajectory_3d[: end_t + 1 : 5, :]
                density_samples = self.density_estimator.sample_density_at_points(
                    trajectory_samples
                )

                # Replace training data
                self.online_estimator.replace_training_data(
                    trajectory_samples, density_samples
                )

                # Update goal density for next chunk
                goal_density = self.online_estimator.get_goal_density(
                    exploit_alpha=self.exploit_alpha,
                    nb_boundary_neighbors=40,
                )

                mean_tmp = self.online_estimator.predict(return_variance=False)
                mean_tmp = normalize_mat(mean_tmp)

        # Final GP update with full trajectory
        print("\nFinal update: Computing final GP predictions...")
        trajectory_samples = trajectory_3d[::5, :]
        density_samples = self.density_estimator.sample_density_at_points(
            trajectory_samples
        )
        self.online_estimator.replace_training_data(trajectory_samples, density_samples)

        # Get final predictions
        mean_final, var_final = self.online_estimator.predict(return_variance=True)
        mean_final = normalize_mat(mean_final)
        var_final = normalize_mat(var_final)

        # Final goal density
        goal_density_final = self.online_estimator.get_goal_density(
            exploit_alpha=self.exploit_alpha,
            nb_boundary_neighbors=40,
        )

        # Print statistics
        print("\n" + "=" * 50)
        print("Laplacian SMC Exploration Statistics:")
        print("=" * 50)
        print(f"Final ergodic metric:  {ergodic_metric_arr[-1]:.6e}")
        print(f"Min speed:             {np.min(speed_arr[1:]):.6e}")
        print(f"Max speed:             {np.max(speed_arr):.6e}")
        print(f"Average speed:         {np.mean(speed_arr[1:]):.6e}")
        print(f"Median speed:          {np.median(speed_arr[1:]):.6e}")
        print(f"Std dev:               {np.std(speed_arr[1:]):.6e}")
        print("=" * 50 + "\n")

        # Build results dictionary
        results = {
            "trajectory_3d": trajectory_3d,
            "goal_density_arr": goal_density_arr,
            "estimated_density_arr": estimated_density_arr,
            "ergodic_metric": ergodic_metric_arr,
            "speed": speed_arr,
            "goal_density_final": goal_density_final,
            "estimated_mean_final": mean_final,
            "estimated_var_final": var_final,
        }

        print(f"\nExploration complete: {self.nbData} timesteps")
        print(f"Final ergodic metric: {results['ergodic_metric'][-1]:.6e}")
        print(f"Mean speed: {results['speed'][1:].mean():.6e}")

        return results


if __name__ == "__main__":
    # Select the object to explore
    obj_name = "bun270_X"  # Stanford bunny with X projected as the target
    # obj_name = "plate_shapes"  # random IKEA plate with hand-drawn shapes

    # Initialize Laplacian SMC explorer
    lap_smc_explorer = LaplacianSMCExplorerGP(
        nbData=1200,
        num_eigen=200,
        dt=0.05,
        u_max=0.1,
        rho_u=1e-3,
        knn=30,
        voxel_size=0.002,
        exploit_alpha=0.6,
        gp_update_interval=50,
        lambda_weight="mezic",
        alpha_fixed=None,
    )

    # Setup point cloud and density estimator
    filename = config.get_point_cloud_path(f"{obj_name}.ply")
    lap_smc_explorer.setup_point_cloud(filename)

    gp_params = {"l": 0.010, "sigma": 1.0, "n_eig": 500}
    lap_smc_explorer.setup_density_estimator(gp_params)

    # Run exploration
    results = lap_smc_explorer.explore(x0=None, seed=0)

    # Visualize results
    print("\nGenerating visualizations...")

    # Plot final trajectory on 3D point cloud
    trajectory_3d = results["trajectory_3d"]

    plots = visualize_point_cloud(
        lap_smc_explorer.pcloud.vertices,
        colors=results["estimated_mean_final"],
        is_show_plot=False,
        point_size=5,
    )
    fig = visualize_trajectory(trajectory_3d, plots, color="black")
    fig.show("browser")

    # Generate animations (showing density evolution over time)
    print("\nGenerating animated visualizations...")

    # Animation for goal density (evolving over time)
    goal_html_path = config.get_animation_path(f"laplacian_smc_goal_{obj_name}.html")
    animate_trajectory_pcloud(
        x_arr=trajectory_3d,
        vertices=lap_smc_explorer.pcloud.vertices,
        color_frames=results["goal_density_arr"],
        timesteps=lap_smc_explorer.nbData + 1,
        save_path=str(goal_html_path),
    )
    print(f"Goal density animation saved to {goal_html_path}")

    # Animation for estimated density (evolving over time)
    est_html_path = config.get_animation_path(
        f"laplacian_smc_estimated_{obj_name}.html"
    )
    animate_trajectory_pcloud(
        x_arr=trajectory_3d,
        vertices=lap_smc_explorer.pcloud.vertices,
        color_frames=results["estimated_density_arr"],
        timesteps=lap_smc_explorer.nbData + 1,
        save_path=str(est_html_path),
    )
    print(f"Estimated density animation saved to {est_html_path}")

    # Plot distribution evolution
    print("\nCreating distribution evolution plot...")
    plot_distribution_evolution_column_auto(
        vertices=lap_smc_explorer.pcloud.vertices,
        original_density=lap_smc_explorer.density_estimator.get_initial_mean(),
        estimated_density_arr=results["estimated_density_arr"],
        pdf_name=str(
            config.get_plot_path(f"laplacian_smc_distribution_evolution_{obj_name}.pdf")
        ),
        agent_trajectory=trajectory_3d,
    )

    print("\nLaplacian SMC exploration complete!")
