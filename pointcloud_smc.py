"""
Point Cloud Ergodic Control - 2D parameterization with 3D projection

This script:
1. Loads a 3D point cloud (from .npy, .ply, .xyz, or generates from mesh)
2. Computes a 2D parameterization (UV mapping) of the point cloud
3. Solves the ergodic control SMC problem on the 2D domain
4. Projects the 2D trajectory back to the 3D point cloud

Copyright notice follows the original ergodic_control_SMC_2D.py
"""

import os
import tempfile
from pathlib import Path

# Matplotlib writes cache/config files; ensure it uses a writable location in sandboxed runs
MPL_DIR = Path(tempfile.gettempdir()) / "matplotlib-cache"
MPL_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_DIR.as_posix())
os.environ.setdefault("XDG_CACHE_HOME", MPL_DIR.as_posix())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, interpn

from plotting_utils import animate_trajectory_pcloud


# Helper functions
# ===============================
def hadamard_matrix(n: int) -> np.ndarray:
    """
    Constructs a Hadamard matrix of size n.

    Args:
        n (int): The size of the Hadamard matrix.

    Returns:
        np.ndarray: A Hadamard matrix of size n.
    """
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


def load_point_cloud(
    filepath: str = None, mesh_path: str = None, n_samples: int = 5000
):
    """
    Load or generate a 3D point cloud.

    Args:
        filepath: Path to point cloud file (.npy, .ply, .xyz)
        mesh_path: Path to mesh file to sample points from
        n_samples: Number of points to sample from mesh

    Returns:
        points: (N, 3) array of 3D point coordinates
        colors: (N, 4) array of RGBA colors (or None if not available)
    """
    colors = None

    if filepath is not None:
        if filepath.endswith(".npy"):
            points = np.load(filepath)
        elif filepath.endswith(".ply"):
            cloud = trimesh.load(filepath)
            points = np.asarray(cloud.vertices)
            # Try to load colors if available
            if hasattr(cloud.visual, "vertex_colors"):
                colors = np.asarray(cloud.visual.vertex_colors)
        elif filepath.endswith(".xyz"):
            points = np.loadtxt(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
    elif mesh_path is not None:
        # Sample points from mesh surface
        mesh = trimesh.load_mesh(mesh_path)
        points, _ = trimesh.sample.sample_surface(mesh, n_samples)
    else:
        raise ValueError("Either filepath or mesh_path must be provided")

    return points, colors


def compute_uv_parameterization_pca(points):
    """
    Compute a 2D UV parameterization using PCA projection.
    Projects points onto the first two principal components.

    Args:
        points: (N, 3) array of 3D point coordinates

    Returns:
        uv_coords: (N, 2) array of UV coordinates normalized to [0, 1]
        points: (N, 3) array of 3D point positions (unchanged)
    """
    # Center the points
    centroid = points.mean(axis=0)
    centered = points - centroid

    # Compute PCA
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (largest first)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Project onto first two principal components
    uv_coords = centered @ eigenvectors[:, :2]

    # Normalize to [0, 1] domain
    uv_min = uv_coords.min(axis=0)
    uv_max = uv_coords.max(axis=0)
    uv_coords = (uv_coords - uv_min) / (uv_max - uv_min)

    return uv_coords, points


def compute_uv_parameterization_xy(points):
    """
    Compute a 2D UV parameterization using simple XY projection.

    Args:
        points: (N, 3) array of 3D point coordinates

    Returns:
        uv_coords: (N, 2) array of UV coordinates normalized to [0, 1]
        points: (N, 3) array of 3D point positions (unchanged)
    """
    # Simple projection onto XY plane
    uv_coords = points[:, :2].copy()

    # Normalize to [0, 1] domain
    uv_min = uv_coords.min(axis=0)
    uv_max = uv_coords.max(axis=0)
    uv_coords = (uv_coords - uv_min) / (uv_max - uv_min)

    return uv_coords, points


def create_uv_interpolator(uv_coords, points):
    """
    Create pre-built interpolators for fast UV to 3D projection.

    Args:
        uv_coords: (N, 2) array - UV coordinates of points
        points: (N, 3) array - 3D positions of points

    Returns:
        tuple: (linear_interpolator, nearest_interpolator)
    """
    linear_interp = LinearNDInterpolator(uv_coords, points)
    nearest_interp = NearestNDInterpolator(uv_coords, points)
    return linear_interp, nearest_interp


def project_uv_to_surface(uv_point, linear_interp, nearest_interp):
    """
    Project a 2D UV point back to the 3D surface using pre-built interpolators.

    Args:
        uv_point: (2,) array - UV coordinates to project
        linear_interp: Pre-built LinearNDInterpolator
        nearest_interp: Pre-built NearestNDInterpolator

    Returns:
        (3,) array - 3D position on the surface
    """
    # Try linear interpolation first
    point_3d = linear_interp(uv_point)

    # If linear fails (point outside convex hull), use nearest
    if np.any(np.isnan(point_3d)):
        point_3d = nearest_interp(uv_point)

    return point_3d


# Parameters
# ===============================
nbData = 500  # Number of datapoints
nbFct = 8  # Number of basis functions along x and y
nbVar = 2  # Dimension of datapoints (2D parameterization)
nbGaussian = 2  # Number of Gaussians to represent the spatial distribution
sp = (nbVar + 1) / 2  # Sobolev norm parameter
dt = 1e-2  # Time step
xlim = [0, 1]  # Domain limit for each dimension
L = (xlim[1] - xlim[0]) * 2  # Size of [-xlim(2),xlim(2)]
om = 2 * np.pi / L
u_max = 3e0  # Maximum speed allowed
u_norm_reg = 1e-3  # Regularizer to avoid numerical issues
# Maximum allowable speed in 3D (to mirror HEDAC-style limits)
u_max_3d = 5e-1

# Initial point in UV space
x0 = [0.1, 0.3]
nbRes = 100

BASE_DIR = Path(__file__).resolve().parent
# DEFAULT_POINT_CLOUD = BASE_DIR / "point_clouds" / "bun270_X.ply"
DEFAULT_POINT_CLOUD = BASE_DIR / "point_clouds" / "plate_shapes.ply"

# Load point cloud
# ===============================
print("Loading point cloud...")
# Load from plate_shapes.ply file
# points, colors = load_point_cloud(filepath="point_clouds/plate_shapes.ply")
points, colors = load_point_cloud(filepath=str(DEFAULT_POINT_CLOUD))
# Other options:
# points, colors = load_point_cloud(mesh_path="surface.obj", n_samples=5000)
# points, colors = load_point_cloud(filepath="pointcloud.npy")

print(f"Point cloud loaded: {len(points)} points")
if colors is not None:
    print(f"Colors loaded: {colors.shape}, range [{colors.min()}, {colors.max()}]")

# Compute UV parameterization
# ===============================
print("Computing UV parameterization...")
# Option 1: Use PCA-based parameterization (better for arbitrary surfaces)
uv_coords, points_3d = compute_uv_parameterization_pca(points)
# Option 2: Use XY projection (simpler, good for height-field-like surfaces)
# uv_coords, points_3d = compute_uv_parameterization_xy(points)

print(f"UV coordinates computed: range [{uv_coords.min():.3f}, {uv_coords.max():.3f}]")

# Desired spatial distribution on UV domain
# ===============================
# Use red channel of point cloud colors as target distribution
if colors is not None:
    print("Using red channel from point cloud colors as target distribution...")
    # Extract red channel (normalized to [0, 1])
    red_channel = colors[:, 0] / 255.0
else:
    print("No colors found, using default Gaussian mixture distribution...")
    red_channel = None

# Fourier basis functions (for a discretized map)
# ===============================
rg = np.arange(0, nbFct, dtype=float)
KX = np.zeros((nbVar, nbFct, nbFct))
KX[0, :, :], KX[1, :, :] = np.meshgrid(rg, rg)
Lambda = np.array(KX[0, :].flatten() ** 2 + KX[1, :].flatten() ** 2 + 1).T ** (-sp)

xm1d = np.linspace(xlim[0], xlim[1], nbRes)
xm = np.zeros((nbGaussian, nbRes, nbRes))
xm[0, :, :], xm[1, :, :] = np.meshgrid(xm1d, xm1d)

arg1 = (
    KX[0, :, :].flatten().T[:, np.newaxis] @ xm[0, :, :].flatten()[:, np.newaxis].T * om
)
arg2 = (
    KX[1, :, :].flatten().T[:, np.newaxis] @ xm[1, :, :].flatten()[:, np.newaxis].T * om
)
phim = np.cos(arg1) * np.cos(arg2) * 2 ** (nbVar)

xx, yy = np.meshgrid(np.arange(1, nbFct + 1), np.arange(1, nbFct + 1))
hk = np.concatenate(([1], 2 * np.ones(nbFct)))
HK = hk[xx.flatten() - 1] * hk[yy.flatten() - 1]
phim = phim * np.tile(HK, (nbRes**nbVar, 1)).T

# Compute Fourier series coefficients w_hat
# ===============================
if red_channel is not None:
    # Create distribution from red channel values in UV space
    # First, create a gridded distribution by interpolating red channel values
    from scipy.interpolate import griddata as scipy_griddata

    # Create grid points in UV space
    grid_u, grid_v = np.meshgrid(xm1d, xm1d)
    grid_points = np.column_stack([grid_u.flatten(), grid_v.flatten()])

    # Interpolate red channel values onto the grid
    g_dist = scipy_griddata(
        uv_coords, red_channel, grid_points, method="linear", fill_value=0.0
    )

    # Handle NaN values (points outside convex hull)
    nan_mask = np.isnan(g_dist)
    if np.any(nan_mask):
        g_dist_nearest = scipy_griddata(
            uv_coords, red_channel, grid_points, method="nearest"
        )
        g_dist[nan_mask] = g_dist_nearest[nan_mask]

    # Ensure non-negative and normalize
    g_dist = np.maximum(g_dist, 0)
    g_dist = g_dist * nbRes**nbVar / (np.sum(g_dist) + 1e-10)

    # Compute Fourier coefficients from the gridded distribution
    phi_inv = np.cos(arg1) * np.cos(arg2) / L**nbVar / nbRes**nbVar
    w_hat = phi_inv @ g_dist

    print(
        f"Distribution from red channel: min={g_dist.min():.4f}, max={g_dist.max():.4f}, sum={g_dist.sum():.4f}"
    )
else:
    # Fallback to GMM if no colors
    Mu = np.zeros((nbVar, 2))
    Mu[:, 0] = np.array([0.5, 0.7])
    Mu[:, 1] = np.array([0.6, 0.3])

    Sigma = np.zeros((nbVar, nbVar, 2))
    Sigma[:, :, 0] = np.eye(nbVar) * 0.05
    Sigma[:, :, 1] = np.eye(nbVar) * 0.03
    Alpha = np.ones(2) / 2

    op = hadamard_matrix(2 ** (nbVar - 1))
    op = np.array(op)
    kk = KX.reshape(nbVar, nbFct**2) * om

    w_hat = np.zeros(nbFct**nbVar)
    for j in range(2):
        for n in range(op.shape[1]):
            MuTmp = np.diag(op[:, n]) @ Mu[:, j]
            SigmaTmp = np.diag(op[:, n]) @ Sigma[:, :, j] @ np.diag(op[:, n]).T
            cos_term = np.cos(kk.T @ MuTmp)
            exp_term = np.exp(np.diag(-0.5 * kk.T @ SigmaTmp @ kk))
            w_hat = w_hat + Alpha[j] * cos_term * exp_term
    w_hat = w_hat / (L**nbVar) / (op.shape[1])

# Compute desired spatial distribution from w_hat
g = w_hat.T @ phim

# Precompute interpolators for fast UV to 3D projection
# ===============================
print("Precomputing UV to 3D interpolators...")
linear_interp, nearest_interp = create_uv_interpolator(uv_coords, points_3d)

# Ergodic control in UV space
# ===============================
print("Running ergodic control...")
x = np.array(x0)  # Initial position in UV space
prev_3d = project_uv_to_surface(x, linear_interp, nearest_interp)

wt = np.zeros(nbFct**nbVar)
r_x_uv = np.zeros((nbVar, nbData))  # Trajectory in UV space
r_x_3d = np.zeros((3, nbData))  # Trajectory in 3D space
r_g = np.zeros((nbRes**nbVar, nbData))
r_w = np.zeros((nbFct**nbVar, nbData))
r_e = np.zeros((nbData))

for t in range(nbData):
    # Fourier basis functions and derivatives
    angle = x[:, np.newaxis] * rg * om
    phi1 = np.cos(angle) / L
    dphi1 = -np.sin(angle) * np.tile(rg * om, (nbVar, 1)) / L

    # Gradient of basis functions
    phix = phi1[0, xx - 1].flatten()
    phiy = phi1[1, yy - 1].flatten()
    dphix = dphi1[0, xx - 1].flatten()
    dphiy = dphi1[1, yy - 1].flatten()

    dphi = np.vstack([[dphix * phiy], [phix * dphiy]]).T

    # Fourier series coefficients along trajectory
    wt = wt + (phix * phiy).T
    w = wt / (t + 1)

    # Controller with constrained velocity norm (UV) then clamp by 3D speed
    u = -dphi.T @ (Lambda * (w - w_hat))
    u_dir = u / (np.linalg.norm(u) + u_norm_reg)  # direction only
    u_uv = u_dir * u_max  # UV space magnitude

    # Trial UV update
    x_trial = np.clip(x + (u_uv * dt), xlim[0], xlim[1])
    point_trial = project_uv_to_surface(x_trial, linear_interp, nearest_interp)

    # Enforce 3D speed cap
    speed_3d = np.linalg.norm(point_trial - prev_3d) / dt
    if speed_3d > u_max_3d:
        scale = u_max_3d / (speed_3d + 1e-9)
        u_uv = u_uv * scale
        x_trial = np.clip(x + (u_uv * dt), xlim[0], xlim[1])
        point_trial = project_uv_to_surface(x_trial, linear_interp, nearest_interp)

    x = x_trial  # Update position in UV space

    # Clamp to valid domain
    x = np.clip(x, xlim[0], xlim[1])

    # Log data
    r_x_uv[:, t] = x

    # Project UV position to 3D surface using pre-built interpolators
    r_x_3d[:, t] = point_trial
    prev_3d = point_trial

    r_g[:, t] = phim.T @ w
    r_w[:, t] = w
    r_e[t] = np.sum((w - w_hat) ** 2 * Lambda)

print("Ergodic control complete!")
print(f"Final reconstruction error: {r_e[-1]:.6f}")

# Visualization
# ===============================
print("Generating visualizations...")

fig = plt.figure(figsize=(20, 10))

# 1. UV space trajectory
ax1 = plt.subplot(2, 3, 1)
G = np.reshape(g, [nbRes, nbRes])
G = np.where(G > 0, G, 0)
X = np.squeeze(xm[0, :, :])
Y = np.squeeze(xm[1, :, :])
ax1.contourf(X, Y, G, cmap="viridis", levels=20)
ax1.plot(r_x_uv[0, :], r_x_uv[1, :], linestyle="-", color="red", linewidth=2, alpha=0.7)
ax1.plot(
    r_x_uv[0, 0],
    r_x_uv[1, 0],
    marker="o",
    color="white",
    markersize=10,
    markeredgecolor="black",
    markeredgewidth=2,
)
ax1.plot(
    r_x_uv[0, -1],
    r_x_uv[1, -1],
    marker="s",
    color="red",
    markersize=10,
    markeredgecolor="black",
    markeredgewidth=2,
)
ax1.set_aspect("equal", "box")
ax1.set_title("Trajectory in UV Space")
ax1.set_xlabel("U")
ax1.set_ylabel("V")

# 2. Desired Fourier coefficients
ax2 = plt.subplot(2, 3, 2)
ax2.set_title(r"Desired Fourier coefficients $\hat{w}$")
ax2.imshow(np.reshape(w_hat, [nbFct, nbFct]).T, cmap="gray_r")
ax2.set_xticks([])
ax2.set_yticks([])

# 3. Reproduced Fourier coefficients
ax3 = plt.subplot(2, 3, 3)
ax3.set_title(r"Reproduced Fourier coefficients $w$")
ax3.imshow(np.reshape(wt / nbData, [nbFct, nbFct]).T, cmap="gray_r")
ax3.set_xticks([])
ax3.set_yticks([])

# 4. 3D point cloud with trajectory colored by target distribution
ax4 = plt.subplot(2, 3, 4, projection="3d")

# Use red channel if available, otherwise use reconstructed distribution
if red_channel is not None:
    # Use red channel values directly
    color_values = red_channel
    color_label = "Red Channel (Target)"
else:
    # Compute distribution value at each point from reconstructed g
    color_values = np.zeros(len(points_3d))
    for i in range(len(points_3d)):
        uv = uv_coords[i]
        u_idx = int(np.clip(uv[0] * (nbRes - 1), 0, nbRes - 1))
        v_idx = int(np.clip(uv[1] * (nbRes - 1), 0, nbRes - 1))
        color_values[i] = G[v_idx, u_idx]
    # Normalize for better visualization
    color_values = color_values / (color_values.max() + 1e-10)
    color_label = "Target Distribution"

# Plot point cloud colored by target distribution
# Use a colormap that shows low values clearly (not white)
scatter = ax4.scatter(
    points_3d[:, 0],
    points_3d[:, 1],
    points_3d[:, 2],
    c=color_values,
    cmap="plasma",  # or 'viridis', 'turbo', 'jet' - all show low values
    s=1,
    alpha=0.7,
    vmin=0,  # Explicitly set range to see full distribution
    vmax=color_values.max(),
)
cbar = plt.colorbar(scatter, ax=ax4, shrink=0.5, aspect=5)
cbar.set_label(color_label, rotation=270, labelpad=15)

# Plot trajectory on top
ax4.plot(
    r_x_3d[0, :],
    r_x_3d[1, :],
    r_x_3d[2, :],
    color="red",
    linewidth=3,
    label="Trajectory",
    zorder=10,
)
ax4.scatter(
    r_x_3d[0, 0],
    r_x_3d[1, 0],
    r_x_3d[2, 0],
    color="white",
    s=100,
    edgecolors="black",
    linewidths=2,
    label="Start",
    zorder=11,
)
ax4.scatter(
    r_x_3d[0, -1],
    r_x_3d[1, -1],
    r_x_3d[2, -1],
    color="red",
    s=100,
    marker="s",
    edgecolors="black",
    linewidths=2,
    label="End",
    zorder=11,
)
ax4.set_title("Trajectory on 3D Point Cloud (GMM colored)")
ax4.set_xlabel("X")
ax4.set_ylabel("Y")
ax4.set_zlabel("Z")
ax4.legend()

# 5. Reconstruction error over time
ax5 = plt.subplot(2, 3, 5)
ax5.plot(r_e, color="blue", linewidth=2)
ax5.set_title("Reconstruction Error over Time")
ax5.set_xlabel("Iteration")
ax5.set_ylabel("Error")
ax5.grid(True, alpha=0.3)

# 6. UV parameterization visualization
ax6 = plt.subplot(2, 3, 6)
scatter = ax6.scatter(
    uv_coords[:, 0], uv_coords[:, 1], c=points_3d[:, 2], cmap="coolwarm", s=5, alpha=0.5
)
ax6.plot(
    r_x_uv[0, :], r_x_uv[1, :], color="red", linewidth=2, alpha=0.8, label="Trajectory"
)
ax6.set_title("UV Parameterization (colored by Z)")
ax6.set_xlabel("U")
ax6.set_ylabel("V")
ax6.set_aspect("equal", "box")
plt.colorbar(scatter, ax=ax6, label="Z coordinate")
ax6.legend()

# Animated trajectory with tiled target distribution colors
print("Generating animated trajectory...")
color_frames = np.repeat(color_values[:, np.newaxis], nbData, axis=1)
try:
    animate_trajectory_pcloud(
        x_arr=r_x_3d.T,
        vertices=points_3d,
        color_frames=color_frames,
        timesteps=nbData,
        save_path=None,
        is_show=True,
    )
    print("Animation displayed.")
except PermissionError:
    fallback_path = "pointcloud_trajectory_animation.html"
    animate_trajectory_pcloud(
        x_arr=r_x_3d.T,
        vertices=points_3d,
        color_frames=color_frames,
        timesteps=nbData,
        save_path=fallback_path,
        is_show=False,
    )
    print(
        f"Animation could not be opened automatically; saved to '{fallback_path}' instead."
    )

plt.tight_layout()
plt.savefig("pointcloud_ergodic_control_results.png", dpi=150, bbox_inches="tight")
print("Results saved to 'pointcloud_ergodic_control_results.png'")
if matplotlib.get_backend().lower() != "agg":
    plt.show()
else:
    plt.close(fig)

# Save trajectory data
# ===============================
np.savez(
    "pointcloud_trajectory.npz",
    trajectory_uv=r_x_uv,
    trajectory_3d=r_x_3d,
    reconstruction_error=r_e,
    uv_coords=uv_coords,
    points_3d=points_3d,
    distribution=g,
)
print("Trajectory data saved to 'pointcloud_trajectory.npz'")
