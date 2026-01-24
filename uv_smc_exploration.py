"""
SMC (Sliding Mode Control) Ergodic Exploration with GP-based Density Estimation

This script combines:
1. SMC ergodic control in UV-parameterized space (from pcloud_uv_smc_common.py)
2. GP-based density estimation (from density_estimator.py)
3. Active learning framework (similar to HEDAC)

The SMC controller operates in 2D UV space for computational efficiency,
while the GP learns the 3D target distribution online through exploration.

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

import config
from exploration_base import ExplorationWithGP
from plotting_utils import *
from pointcloud_utils import *
from pcloud_uv_smc_common import (
    ErgodicControlUV,
    compute_uv_parameterization_pca,
    create_uv_interpolator,
    project_uv_to_surface,
    setup_fourier_basis,
)


class SMCExplorerGP(ExplorationWithGP):
    """
    SMC-based ergodic exploration with GP density estimation.

    This class combines SMC control in UV space with online GP learning
    to actively explore unknown density distributions on point clouds.
    Inherits common GP and point cloud setup from ExplorationWithGP.
    """

    def __init__(
        self,
        nbData=500,
        nbFct=50,
        nbRes=100,
        dt=1.0,
        u_max=1e-2,
        voxel_size=0.002,
        exploit_alpha=0.6,
        gp_update_interval=50,
    ):
        """
        Initialize SMC exploration parameters.

        Args:
            nbData: Number of timesteps for exploration
            nbFct: Number of Fourier basis functions along x and y
            nbRes: Resolution of the grid for distribution
            dt: Time step for SMC controller
            u_max: Maximum speed in UV space
            voxel_size: Voxel size for point cloud downsampling
            exploit_alpha: Balance between exploration (0) and exploitation (1)
            gp_update_interval: How often to update GP model (every N timesteps)
        """
        # Initialize base class
        super().__init__(
            voxel_size=voxel_size,
            exploit_alpha=exploit_alpha,
            gp_update_interval=gp_update_interval,
        )

        # SMC-specific parameters
        self.nbData = nbData
        self.nbFct = nbFct
        self.nbVar = 2  # UV is 2D
        self.nbRes = nbRes
        self.sp = (self.nbVar + 1) / 2  # Sobolev norm parameter
        self.dt = dt
        self.u_max = u_max
        self.u_max_3d = 1 * voxel_size * 2  # Limit 3D speed based on voxel size
        self.u_norm_reg = 1e-8  # Regularizer
        self.xlim = [0, 1]  # Domain limit for each dimension

        # SMC-specific attributes (set during setup)
        self.uv_coords = None
        self.points_3d = None
        self.linear_interp = None
        self.nearest_interp = None
        self.fourier_basis = None
        self.online_estimator = None

    def _setup_point_cloud_hook(self):
        """
        SMC-specific point cloud setup: UV parameterization and Fourier basis.
        """
        # Compute UV parameterization
        print("Computing UV parameterization...")
        self.uv_coords, self.points_3d = compute_uv_parameterization_pca(
            self.pcloud.vertices
        )
        print(
            f"UV coordinates: range [{self.uv_coords.min():.3f}, {self.uv_coords.max():.3f}]"
        )

        # Create UV to 3D interpolators
        print("Creating UV to 3D interpolators...")
        self.linear_interp, self.nearest_interp = create_uv_interpolator(
            self.uv_coords, self.points_3d
        )

        # Setup Fourier basis
        print("Setting up Fourier basis...")
        self.fourier_basis = setup_fourier_basis(
            self.nbFct, self.nbVar, self.nbRes, self.xlim, self.sp
        )

    def compute_w_hat_from_gp(self, goal_density):
        """
        Compute Fourier coefficients from GP-predicted goal density.

        Args:
            goal_density: GP-predicted density on point cloud vertices

        Returns:
            w_hat: Fourier coefficients for the target distribution
        """
        # Interpolate goal density to UV grid
        from scipy.interpolate import griddata

        xm1d = self.fourier_basis["xm1d"]
        arg1 = self.fourier_basis["arg1"]
        arg2 = self.fourier_basis["arg2"]
        L = self.fourier_basis["L"]

        # Create grid points in UV space
        grid_u, grid_v = np.meshgrid(xm1d, xm1d)
        grid_points = np.column_stack([grid_u.flatten(), grid_v.flatten()])

        # Interpolate goal density from 3D points to UV grid
        g_dist = griddata(
            self.uv_coords, goal_density, grid_points, method="linear", fill_value=0.0
        )

        # Handle NaN values (points outside convex hull)
        nan_mask = np.isnan(g_dist)
        if np.any(nan_mask):
            g_dist_nearest = griddata(
                self.uv_coords, goal_density, grid_points, method="nearest"
            )
            g_dist[nan_mask] = g_dist_nearest[nan_mask]

        # Ensure non-negative and normalize
        g_dist = np.maximum(g_dist, 0)
        g_dist = g_dist * self.nbRes**self.nbVar / (np.sum(g_dist) + 1e-10)

        # Compute Fourier coefficients from the gridded distribution
        # arg1 and arg2 are (nbFct^2, nbRes^2) matrices
        phi_inv = np.cos(arg1) * np.cos(arg2) / L**self.nbVar / self.nbRes**self.nbVar
        w_hat = phi_inv @ g_dist

        return w_hat

    def explore(self, x0_uv=None):
        """
        Perform SMC exploration with online GP density estimation.

        This method runs SMC control with periodic GP updates to adapt
        the target distribution based on learned information.

        Args:
            x0_uv: Initial position in UV space [u, v]. If None, uses center [0.4, 0.6]

        Returns:
            dict: Exploration results containing trajectories and statistics
        """
        if x0_uv is None:
            x0_uv = [0.4, 0.6]

        # Project initial UV point to 3D
        x0_3d = project_uv_to_surface(x0_uv, self.linear_interp, self.nearest_interp)

        # Sample initial density at starting position
        initial_sample = self.density_estimator.sample_density_at_point(x0_3d)

        # Create online estimator
        self.online_estimator = self.density_estimator.create_online_estimator(
            initial_trajectory=x0_3d.reshape(1, -1),
            initial_samples=np.array([initial_sample]),
        )

        # Get initial goal density (use only mean initially)
        mean_tmp = self.online_estimator.predict(return_variance=False)
        mean_tmp = normalize_mat(mean_tmp)
        goal_density = mean_tmp.copy()

        # Compute initial Fourier coefficients
        w_hat = self.compute_w_hat_from_gp(goal_density)

        # Storage arrays for trajectory and densities over time
        trajectory_3d_list = []
        goal_density_arr = np.zeros((len(self.pcloud.vertices), self.nbData))
        estimated_density_arr = np.zeros((len(self.pcloud.vertices), self.nbData))

        print("Starting SMC exploration with GP updates...")

        # Run SMC controller in chunks to allow periodic GP updates
        chunk_size = self.gp_update_interval
        num_chunks = (self.nbData + chunk_size - 1) // chunk_size

        current_x_uv = np.array(x0_uv)
        all_trajectories_uv = []
        all_trajectories_3d = []
        all_reconstruction_errors = []
        all_speeds_3d = []

        for chunk_idx in range(num_chunks):
            # Determine timesteps for this chunk
            start_t = chunk_idx * chunk_size
            end_t = min((chunk_idx + 1) * chunk_size, self.nbData)
            chunk_steps = end_t - start_t

            if chunk_idx > 0:
                print(f"\nTime step: {start_t}/{self.nbData}")

            # Run SMC for this chunk
            smc = ErgodicControlUV(
                x0=current_x_uv,
                nbData=chunk_steps,
                nbFct=self.nbFct,
                nbVar=self.nbVar,
                nbRes=self.nbRes,
                dt=self.dt,
                u_max=self.u_max,
                u_max_3d=self.u_max_3d,
                u_norm_reg=self.u_norm_reg,
                xlim=self.xlim,
                fourier_basis=self.fourier_basis,
                w_hat=w_hat,
                linear_interp=self.linear_interp,
                nearest_interp=self.nearest_interp,
                phim=self.fourier_basis["phim"],
                enable_3d_speed_control=True,
                debug=False,
            )

            chunk_results = smc.run()

            # Store chunk results
            all_trajectories_uv.append(chunk_results["trajectory_uv"])
            all_trajectories_3d.append(chunk_results["trajectory_3d"])
            all_reconstruction_errors.append(chunk_results["reconstruction_error"])
            all_speeds_3d.append(chunk_results["speed_3d"])

            # Store goal and estimated densities for this chunk (repeat for each timestep)
            for t_offset in range(chunk_steps):
                goal_density_arr[:, start_t + t_offset] = goal_density
                estimated_density_arr[:, start_t + t_offset] = mean_tmp

            # Update current position for next chunk
            current_x_uv = chunk_results["trajectory_uv"][:, -1]

            # Collect trajectory for GP update
            chunk_traj_3d = chunk_results["trajectory_3d"].T  # (chunk_steps, 3)
            trajectory_3d_list.append(chunk_traj_3d)

            # Update GP with trajectory so far (if not the last chunk)
            if chunk_idx < num_chunks - 1:
                # Concatenate all trajectory chunks so far
                full_trajectory = np.vstack(trajectory_3d_list)

                # Sample densities along trajectory (every 5th point)
                trajectory_samples = full_trajectory[::5, :]
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

                # Recompute Fourier coefficients with updated goal
                w_hat = self.compute_w_hat_from_gp(goal_density)

        # Concatenate all chunks
        trajectory_uv = np.hstack(all_trajectories_uv)
        trajectory_3d = np.hstack(all_trajectories_3d)
        reconstruction_error = np.concatenate(all_reconstruction_errors)
        speed_3d = np.concatenate(all_speeds_3d)

        # Final GP update with full trajectory
        print("\nFinal update: Computing final GP predictions...")
        full_trajectory = np.vstack(trajectory_3d_list)
        trajectory_samples = full_trajectory[::5, :]
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

        # Build results dictionary
        results = {
            "trajectory_uv": trajectory_uv,
            "trajectory_3d": trajectory_3d,
            "reconstruction_error": reconstruction_error,
            "speed_3d": speed_3d,
            "goal_density_arr": goal_density_arr,
            "estimated_density_arr": estimated_density_arr,
            "goal_density_final": goal_density_final,
            "estimated_mean_final": mean_final,
            "estimated_var_final": var_final,
        }

        print(f"\nExploration complete: {self.nbData} timesteps")
        print(f"Final reconstruction error: {results['reconstruction_error'][-1]:.6e}")
        print(f"Mean 3D speed: {results['speed_3d'].mean():.6e}")

        return results


if __name__ == "__main__":
    # Select the object to explore
    obj_name = "bun270_X"  # Stanford bunny with X projected as the target
    # obj_name = "plate_shapes"  # random IKEA plate with hand-drawn shapes

    # Initialize SMC explorer
    smc_explorer = SMCExplorerGP(
        nbData=500,
        nbFct=50,
        nbRes=100,
        dt=1.0,
        u_max=1e-2,
        voxel_size=0.002,
        exploit_alpha=0.6,
        gp_update_interval=50,
    )

    # Setup point cloud and density estimator
    filename = config.get_point_cloud_path(f"{obj_name}.ply")
    smc_explorer.setup_point_cloud(filename)

    gp_params = {"l": 0.010, "sigma": 1.0, "n_eig": 500}
    smc_explorer.setup_density_estimator(gp_params)

    # Run exploration
    results = smc_explorer.explore(x0_uv=[0.4, 0.6])

    # Visualize results
    print("\nGenerating visualizations...")

    # Plot final trajectory on 3D point cloud
    trajectory_3d = results["trajectory_3d"].T  # (N, 3)

    plots = visualize_point_cloud(
        smc_explorer.pcloud.vertices,
        colors=results["estimated_mean_final"],
        is_show_plot=False,
        point_size=5,
    )
    fig = visualize_trajectory(trajectory_3d, plots, color="black")
    fig.show("browser")

    # Generate animations (showing density evolution over time)
    print("\nGenerating animated visualizations...")

    # Animation for goal density (evolving over time)
    goal_html_path = config.get_animation_path(f"smc_goal_{obj_name}.html")
    animate_trajectory_pcloud(
        x_arr=trajectory_3d,
        vertices=smc_explorer.pcloud.vertices,
        color_frames=results["goal_density_arr"],
        timesteps=smc_explorer.nbData,
        save_path=str(goal_html_path),
    )
    print(f"Goal density animation saved to {goal_html_path}")

    # Animation for estimated density (evolving over time)
    est_html_path = config.get_animation_path(f"smc_estimated_{obj_name}.html")
    animate_trajectory_pcloud(
        x_arr=trajectory_3d,
        vertices=smc_explorer.pcloud.vertices,
        color_frames=results["estimated_density_arr"],
        timesteps=smc_explorer.nbData,
        save_path=str(est_html_path),
    )
    print(f"Estimated density animation saved to {est_html_path}")

    # Plot distribution evolution
    print("\nCreating distribution evolution plot...")
    plot_distribution_evolution_column_auto(
        vertices=smc_explorer.pcloud.vertices,
        original_density=smc_explorer.density_estimator.get_initial_mean(),
        estimated_density_arr=results["estimated_density_arr"],
        pdf_name=str(
            config.get_plot_path(f"smc_distribution_evolution_{obj_name}.pdf")
        ),
        agent_trajectory=trajectory_3d,
    )

    print("\nSMC exploration complete!")
