"""
SMC-based Ergodic Exploration with GPR

This script combines:
1. SMC (Sliding Mode Control) ergodic controller from pointcloud_smc.py
2. GPR (Gaussian Process Regression) density estimation from ergodic_exploration.py
3. Visualization functions from ergodic_exploration.py

The SMC controller works in 2D UV parameterized space for computational efficiency,
while maintaining the GPR-based active learning framework.
"""

import time

import gpytorch
import numpy as np
import open3d as o3d
import torch
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, griddata

from gpr_on_point_cloud import *
from plotting_utils import *
from pointcloud_utils import *


# Helper functions from pointcloud_smc
# =====================================
def hadamard_matrix(n: int) -> np.ndarray:
    """Constructs a Hadamard matrix of size n."""
    if n == 1:
        return np.array([[1]])

    half_size = n // 2
    h_half = hadamard_matrix(half_size)

    h = np.empty((n, n), dtype=int)
    h[:half_size, :half_size] = h_half
    h[half_size:, :half_size] = h_half
    h[:half_size:, half_size:] = h_half
    h[half_size:, half_size:] = -h_half

    return h


def compute_uv_parameterization_pca(points):
    """Compute a 2D UV parameterization using PCA projection."""
    centroid = points.mean(axis=0)
    centered = points - centroid

    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    uv_coords = centered @ eigenvectors[:, :2]

    uv_min = uv_coords.min(axis=0)
    uv_max = uv_coords.max(axis=0)
    uv_coords = (uv_coords - uv_min) / (uv_max - uv_min)

    return uv_coords, eigenvectors[:, :2], centroid, (uv_min, uv_max)


def create_uv_interpolator(uv_coords, points):
    """Create pre-built interpolators for fast UV to 3D projection."""
    linear_interp = LinearNDInterpolator(uv_coords, points)
    nearest_interp = NearestNDInterpolator(uv_coords, points)
    return linear_interp, nearest_interp


def project_uv_to_surface(uv_point, linear_interp, nearest_interp):
    """Project a 2D UV point back to the 3D surface."""
    point_3d = linear_interp(uv_point)
    if np.any(np.isnan(point_3d)):
        point_3d = nearest_interp(uv_point)
    return point_3d


def smc_ergodic_exploration_with_gpr(
    pcloud, param, model_real, likelihood_real, km, uv_coords, uv_params
):
    """
    Perform SMC-based ergodic exploration with GPR estimation.

    Args:
        pcloud: Point cloud object with vertices
        param: Parameters object
        model_real: GPR model for ground truth target
        likelihood_real: GPR likelihood
        km: Kernel matrix for point cloud
        uv_coords: UV coordinates of point cloud vertices
        uv_params: Tuple of (eigenvectors, centroid, uv_bounds)

    Returns:
        tuple: (trajectory, goal_density_arr, estimated_density_arr)
    """
    # Unpack UV parameters
    eigenvectors, centroid, uv_bounds = uv_params

    # Initialize arrays to store results
    goal_density_arr = []
    estimated_density_arr = []
    trajectory = []

    # SMC parameters
    nbFct = param.nbFct
    nbVar = 2  # UV space is 2D
    sp = (nbVar + 1) / 2
    dt = param.dt
    xlim = [0, 1]
    L = (xlim[1] - xlim[0]) * 2
    om = 2 * np.pi / L
    u_max = param.u_max
    u_norm_reg = 1e-3
    nbRes = param.nbRes

    # Fourier basis setup
    rg = np.arange(0, nbFct, dtype=float)
    KX = np.zeros((nbVar, nbFct, nbFct))
    KX[0, :, :], KX[1, :, :] = np.meshgrid(rg, rg)
    Lambda = np.array(KX[0, :].flatten() ** 2 + KX[1, :].flatten() ** 2 + 1).T ** (-sp)

    xm1d = np.linspace(xlim[0], xlim[1], nbRes)
    xm = np.zeros((nbVar, nbRes, nbRes))
    xm[0, :, :], xm[1, :, :] = np.meshgrid(xm1d, xm1d)

    arg1 = (
        KX[0, :, :].flatten().T[:, np.newaxis]
        @ xm[0, :, :].flatten()[:, np.newaxis].T
        * om
    )
    arg2 = (
        KX[1, :, :].flatten().T[:, np.newaxis]
        @ xm[1, :, :].flatten()[:, np.newaxis].T
        * om
    )
    phim = np.cos(arg1) * np.cos(arg2) * 2**nbVar

    xx, yy = np.meshgrid(np.arange(1, nbFct + 1), np.arange(1, nbFct + 1))
    hk = np.concatenate(([1], 2 * np.ones(nbFct)))
    HK = hk[xx.flatten() - 1] * hk[yy.flatten() - 1]
    phim = phim * np.tile(HK, (nbRes**nbVar, 1)).T

    # Precompute interpolators for UV to 3D projection
    linear_interp, nearest_interp = create_uv_interpolator(uv_coords, pcloud.vertices)

    # Initialize GPR with first sample
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Project initial position from 3D to UV
    current_pos_3d = np.array(param.x0[:3])
    centered_pos = current_pos_3d - centroid
    uv_min, uv_max = uv_bounds
    current_pos_uv = centered_pos @ eigenvectors
    current_pos_uv = (current_pos_uv - uv_min) / (uv_max - uv_min)
    current_pos_uv = np.clip(current_pos_uv, xlim[0], xlim[1])

    sample_points = torch.tensor(current_pos_3d, dtype=torch.float32).reshape(1, -1)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        gpr_original_density = likelihood_real(model_real(sample_points))

    density_sample = gpr_original_density.mean
    train_x = sample_points.clone()
    train_y = density_sample.clone()

    # Initialize GPR model for online learning
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPROnPointCloud(train_x, train_y, likelihood, km, pcloud.vertices)

    model.eval()
    likelihood.eval()

    test_x = torch.tensor(pcloud.vertices, dtype=torch.float32, device=torch_device)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))

    goal_density = observed_pred.mean.cpu().numpy()
    goal_density = normalize_mat(goal_density)
    mean_tmp = normalize_mat(observed_pred.mean.cpu().numpy())

    print(f"Initial goal density shape: {goal_density.shape}")
    print(f"Starting SMC-based exploration with GPR updates...")

    # Compute initial w_hat from goal density (do this once at start and when GP updates)
    def compute_w_hat(goal_density):
        """Compute Fourier coefficients from goal density."""
        grid_u, grid_v = np.meshgrid(xm1d, xm1d)
        grid_points = np.column_stack([grid_u.flatten(), grid_v.flatten()])

        g_dist = griddata(
            uv_coords, goal_density, grid_points, method="linear", fill_value=0.0
        )

        nan_mask = np.isnan(g_dist)
        if np.any(nan_mask):
            g_dist_nearest = griddata(
                uv_coords, goal_density, grid_points, method="nearest"
            )
            g_dist[nan_mask] = g_dist_nearest[nan_mask]

        g_dist = np.maximum(g_dist, 0)
        g_dist = g_dist * nbRes**nbVar / (np.sum(g_dist) + 1e-10)

        phi_inv = np.cos(arg1) * np.cos(arg2) / L**nbVar / nbRes**nbVar
        return phi_inv @ g_dist

    # Compute initial w_hat
    w_hat = compute_w_hat(goal_density)

    # Start SMC control loop
    start_time = time.time()
    x_uv = current_pos_uv.copy()  # Current position in UV space
    wt = np.zeros(nbFct**nbVar)
    previous_3d_pos = (
        current_pos_3d.copy()
    )  # Track 3D position for velocity calculation

    timestep = 0
    while timestep < param.timesteps:
        # Update goal density with GPR every N timesteps
        if timestep % param.gp_update_freq == 0 and timestep > 0:
            print(f"\nTimestep {timestep}/{param.timesteps}")

            # Update GPR with trajectory so far
            traj_array = np.array(trajectory[:timestep:5])
            sample_points = torch.tensor(
                traj_array, dtype=torch.float32, device=torch_device
            )
            print(f"Sample points shape: {sample_points.shape}")

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                gpr_original_density = likelihood_real(model_real(sample_points))

            density_sample = gpr_original_density.mean
            train_x = sample_points.clone()
            train_y = density_sample.clone()

            model.train()
            likelihood.train()
            model.set_train_data(train_x, train_y, strict=False)
            model.eval()
            likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = likelihood(model(test_x))

            var_tmp = normalize_mat(observed_pred.variance.cpu().numpy())
            mean_tmp = normalize_mat(observed_pred.mean.cpu().numpy())

            # Exploration-exploitation tradeoff
            goal_density = param.exploit_alpha * normalize_mat(
                np.maximum(mean_tmp - np.mean(mean_tmp), 0)
            ) + (1 - param.exploit_alpha) * normalize_mat(var_tmp)
            goal_density = normalize_mat(goal_density)

            print(
                f"Updated goal density - min: {goal_density.min():.6f}, max: {goal_density.max():.6f}"
            )

            # Recompute w_hat only when GP updates (not every timestep!)
            w_hat = compute_w_hat(goal_density)

        # SMC control step
        angle = x_uv[:, np.newaxis] * rg * om
        phi1 = np.cos(angle) / L
        dphi1 = -np.sin(angle) * np.tile(rg * om, (nbVar, 1)) / L

        phix = phi1[0, xx - 1].flatten()
        phiy = phi1[1, yy - 1].flatten()
        dphix = dphi1[0, xx - 1].flatten()
        dphiy = dphi1[1, yy - 1].flatten()

        dphi = np.vstack([[dphix * phiy], [phix * dphiy]]).T

        wt = wt + (phix * phiy).T
        w = wt / (timestep + 1)

        # Controller with constrained velocity norm
        u = -dphi.T @ (Lambda * (w - w_hat))
        u = u * u_max / (np.linalg.norm(u) + u_norm_reg)

        x_uv = x_uv + (u * dt)
        x_uv = np.clip(x_uv, xlim[0], xlim[1])

        # Project UV position to 3D surface
        point_3d = project_uv_to_surface(x_uv, linear_interp, nearest_interp)
        # Ensure point_3d is a 1D array of shape (3,)
        if point_3d.ndim > 1:
            point_3d = point_3d.flatten()
        trajectory.append(point_3d.copy())

        # Calculate actual 3D velocity for monitoring
        if timestep > 0:
            displacement_3d = np.linalg.norm(point_3d - previous_3d_pos)
            velocity_3d = displacement_3d / dt  # Actual 3D velocity
        else:
            velocity_3d = 0.0

        previous_3d_pos = point_3d.copy()

        # Store density arrays
        if timestep % param.gp_update_freq == 0:
            goal_density_arr.append(goal_density.copy())
            estimated_density_arr.append(mean_tmp.copy())

        # Print progress with 3D velocity monitoring
        if timestep % 50 == 0:
            elapsed = time.time() - start_time
            print(
                f"Timestep {timestep}/{param.timesteps} - Elapsed: {elapsed:.2f}s - 3D velocity: {velocity_3d:.6f} m/s (target: {param.hedac_max_velocity:.6f})"
            )

        timestep += 1

    total_time = time.time() - start_time
    print(f"\nTotal exploration time: {total_time:.2f}s")

    trajectory = np.array(trajectory)
    goal_density_arr = np.array(goal_density_arr)
    estimated_density_arr = np.array(estimated_density_arr)

    return trajectory, goal_density_arr, estimated_density_arr


def main():
    # Parameters
    class param:
        pass

    # Object selection
    # object_name = "bun270_X"
    object_name = "plate_shapes"
    print(f"Object: {object_name}")

    # Point cloud processing parameters
    param.voxel_size = 0.002
    param.nb_boundary_neighbors = 40
    param.alpha = 100

    # Load and process point cloud
    point_cloud_dir = "point_clouds/"
    filename = f"{point_cloud_dir}{object_name}.ply"
    pcloud = process_point_cloud(filename, param)

    print(f"Point cloud vertices: {len(pcloud.vertices)}")

    # Compute UV parameterization
    print("Computing UV parameterization...")
    uv_coords, eigenvectors, centroid, uv_bounds = compute_uv_parameterization_pca(
        pcloud.vertices
    )
    print(
        f"UV coordinates computed: range [{uv_coords.min():.3f}, {uv_coords.max():.3f}]"
    )

    # GPR parameters
    l = 0.010
    sigma = 1.0
    n_eig = 500
    km = rbf_manifold_kernel(pcloud.vertices, l, sigma, n_eig)

    # Set up ground truth GPR model
    train_x = torch.tensor(pcloud.vertices, dtype=torch.float32)
    train_y = torch.tensor(pcloud.u0, dtype=torch.float32)

    likelihood_real = gpytorch.likelihoods.GaussianLikelihood()
    model_real = GPROnPointCloud(train_x, train_y, likelihood_real, km, pcloud.vertices)

    model_real.train()
    likelihood_real.train()
    model_real.eval()
    likelihood_real.eval()

    # SMC exploration parameters - match HEDAC for fair comparison
    param.timesteps = 1500  # Same as HEDAC (ergodic_exploration.py:263)
    param.gp_update_freq = 50  # Update GP every 50 timesteps (same as HEDAC)
    param.exploit_alpha = 0.6  # Same exploration-exploitation tradeoff as HEDAC

    # SMC-specific parameters
    param.dt = 1e-2  # Time step for SMC controller
    param.nbFct = 8  # Number of Fourier basis functions
    param.nbRes = 100  # Resolution for Fourier reconstruction

    # Speed matching with HEDAC
    # HEDAC: max_velocity = 0.1 * voxel_size * 2 = 0.1 * 0.002 * 2 = 0.0004 m/s
    # We want to match this speed in 3D space, not UV space
    # Since we control in UV space but care about 3D speed, we'll use a higher
    # UV space velocity and let the natural dynamics determine the 3D speed
    # The actual 3D speed will be monitored and reported
    hedac_max_velocity = 0.1 * param.voxel_size * 2  # 0.0004 m/s in 3D
    param.hedac_max_velocity = hedac_max_velocity
    # Use a moderate UV space speed - the 3D projection will determine actual speed
    param.u_max = 5e-1  # UV space speed (will be adjusted if needed)

    print(f"Total timesteps: {param.timesteps}")
    print(f"GP update frequency: every {param.gp_update_freq} timesteps")
    print(f"HEDAC max velocity (3D): {hedac_max_velocity:.6f} m/s")
    print(f"SMC max velocity (UV space): {param.u_max:.6f} units/s")
    print(f"Exploration-exploitation alpha: {param.exploit_alpha}")

    # Set initial position
    param.x0 = np.array([*pcloud.vertices[1500], 0.0, 0.0, 0.0])
    print(f"Initial position: {param.x0[:3]}")

    # Run SMC-based exploration with GPR
    uv_params = (eigenvectors, centroid, uv_bounds)
    trajectory, goal_density_arr, estimated_density_arr = (
        smc_ergodic_exploration_with_gpr(
            pcloud, param, model_real, likelihood_real, km, uv_coords, uv_params
        )
    )

    # Visualization
    print("\nGenerating visualization...")
    plots = visualize_point_cloud(
        pcloud.vertices,
        colors=estimated_density_arr[-1],
        is_show_plot=False,
        point_size=5,
    )
    fig = visualize_trajectory(trajectory, plots, color="black")
    fig.show("browser")

    print("Visualization complete!")

    # Get original density
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred_original = likelihood_real(model_real(train_x))
    original_density = observed_pred_original.mean.cpu().numpy()

    # Plot distribution evolution
    print("\nGenerating distribution evolution plot...")
    plot_distribution_evolution_column_auto(
        vertices=pcloud.vertices,
        original_density=original_density,
        estimated_density_arr=estimated_density_arr.T,
        pdf_name=f"smc_distribution_evolution_{object_name}.pdf",
        agent_trajectory=trajectory,
    )
    print(
        f"Distribution evolution saved to smc_distribution_evolution_{object_name}.pdf"
    )

    # Generate animated visualization
    print("\nGenerating animated visualization...")
    total_traj_steps = len(trajectory)

    # Interpolate density arrays to match trajectory length
    density_frames_est = np.zeros((len(pcloud.vertices), total_traj_steps))

    for i in range(len(estimated_density_arr)):
        start_idx = i * param.gp_update_freq
        end_idx = min((i + 1) * param.gp_update_freq, total_traj_steps)
        density_frames_est[:, start_idx:end_idx] = estimated_density_arr[i : i + 1].T

    est_html_path = f"smc_estimated_density_{object_name}.html"
    animate_trajectory_pcloud(
        x_arr=trajectory,
        vertices=pcloud.vertices,
        color_frames=density_frames_est,
        timesteps=total_traj_steps,
        save_path=est_html_path,
    )
    print(f"Estimated density animation saved to {est_html_path}")
    print("Animation opened in browser!")

    # Save results
    print("\nSaving results...")
    np.savez(
        f"smc_exploration_{object_name}.npz",
        trajectory=trajectory,
        goal_density_arr=goal_density_arr,
        estimated_density_arr=estimated_density_arr,
        vertices=pcloud.vertices,
        original_density=original_density,
        uv_coords=uv_coords,
    )
    print(f"Results saved to smc_exploration_{object_name}.npz")


if __name__ == "__main__":
    main()
