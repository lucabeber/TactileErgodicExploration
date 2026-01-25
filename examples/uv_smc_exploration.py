"""
Point Cloud Ergodic Control - 2D parameterization with 3D projection

This script:
1. Loads a 3D point cloud (from .npy, .ply, .xyz, or generates from mesh)
2. Computes a 2D parameterization (UV mapping) of the point cloud
3. Solves the ergodic control SMC problem on the 2D domain
4. Projects the 2D trajectory back to the 3D point cloud

Copyright notice follows the original ergodic_control_SMC_2D.py
"""

import numpy as np
import open3d as o3d

from tactile_ergodic.exploration.ergodic_control_uv import (
    ErgodicControlUV,
    compute_fourier_coefficients_from_red_channel,
    compute_uv_parameterization_pca,
    compute_uv_parameterization_xy,
    create_uv_interpolator,
    project_uv_to_surface,
    setup_fourier_basis,
)
from tactile_ergodic.utils.pointcloud_utils import process_point_cloud
from tactile_ergodic.utils import config

# Parameters
# ===============================
nbData = 500  # Number of datapoints
nbFct = 30  # Number of basis functions along x and y
nbVar = 2  # Dimension of datapoints (2D parameterization)
nbGaussian = 2  # Number of Gaussians to represent the spatial distribution
sp = (nbVar + 1) / 2  # Sobolev norm parameter
dt = 1.0  # Time step
xlim = [0, 1]  # Domain limit for each dimension
u_max = 1e-2  # Maximum speed allowed
u_norm_reg = 1e-8  # Regularizer to avoid numerical issues
# Maximum allowable speed in 3D (set based on voxel size)
voxel_size = 0.002  # Voxel size for downsampling
u_max_3d = 1 * voxel_size * 2  # Limit 3D speed based on voxel size

# Initial point in UV space
x0 = [0.4, 0.6]
nbRes = 100

DEFAULT_POINT_CLOUD = config.get_point_cloud_path("bun270_X.ply")
# DEFAULT_POINT_CLOUD = config.get_point_cloud_path("plate_shapes.ply")

# Load and process point cloud
# ===============================
print("Loading point cloud...")


# Create parameter object for process_point_cloud
class Param:
    def __init__(self):
        self.voxel_size = voxel_size
        self.alpha = 1  # scaling factor for dt calculation


param = Param()

# Use process_point_cloud from pointcloud_utils
pcloud = process_point_cloud(str(DEFAULT_POINT_CLOUD), param)
points = pcloud.vertices
print(f"Point cloud processed: {len(points)} points")

# Compute UV parameterization
# ===============================
print("Computing UV parameterization...")

# Choose one of the following parameterization methods:

# Option 1: PCA - Fast, simple projection (current default)
uv_coords, points_3d = compute_uv_parameterization_pca(points)

# Option 2: XY projection - Simplest, good for height-field-like surfaces
# uv_coords, points_3d = compute_uv_parameterization_xy(points)

# Note: Other parameterization methods (Isomap, LLE, MDS, Spectral) are not yet implemented
# in the tactile_ergodic package. Only PCA and XY methods are currently available.

print(f"UV coordinates computed: range [{uv_coords.min():.3f}, {uv_coords.max():.3f}]")

# Desired spatial distribution on UV domain
# ===============================
# Use red channel from point cloud (already extracted by process_point_cloud)
print("Using red channel from point cloud colors as target distribution...")
red_channel = pcloud.u0  # Already normalized [0, 1]
print(f"Red channel values: range [{red_channel.min():.3f}, {red_channel.max():.3f}]")

# Fourier basis functions (for a discretized map)
# ===============================
fourier_basis = setup_fourier_basis(nbFct, nbVar, nbRes, xlim, sp)
rg = fourier_basis["rg"]
KX = fourier_basis["KX"]
Lambda = fourier_basis["Lambda"]
xm1d = fourier_basis["xm1d"]
xm = fourier_basis["xm"]
arg1 = fourier_basis["arg1"]
arg2 = fourier_basis["arg2"]
phim = fourier_basis["phim"]
xx = fourier_basis["xx"]
yy = fourier_basis["yy"]
om = fourier_basis["om"]
L = fourier_basis["L"]

# Compute Fourier series coefficients w_hat
# ===============================
w_hat, g_dist = compute_fourier_coefficients_from_red_channel(
    uv_coords, red_channel, xm1d, arg1, arg2, nbRes, nbVar, L
)

print(
    f"Distribution from red channel: min={g_dist.min():.4f}, max={g_dist.max():.4f}, sum={g_dist.sum():.4f}"
)


# Compute desired spatial distribution from w_hat
g = w_hat.T @ phim

# Precompute interpolators for fast UV to 3D projection
# ===============================
print("Precomputing UV to 3D interpolators...")
linear_interp, nearest_interp = create_uv_interpolator(uv_coords, points_3d)

# Ergodic control in UV space
# ===============================
ergodic_control = ErgodicControlUV(
    x0=x0,
    nbData=nbData,
    nbFct=nbFct,
    nbVar=nbVar,
    nbRes=nbRes,
    dt=dt,
    u_max=u_max,
    u_max_3d=u_max_3d,
    u_norm_reg=u_norm_reg,
    xlim=xlim,
    fourier_basis=fourier_basis,
    w_hat=w_hat,
    linear_interp=linear_interp,
    nearest_interp=nearest_interp,
    phim=phim,
    enable_3d_speed_control=False,  # Disabled to check if it causes wiggles
    debug=True,
)

results = ergodic_control.run()

# Extract results
r_x_uv = results["trajectory_uv"]
r_x_3d = results["trajectory_3d"]
r_e = results["reconstruction_error"]
r_w = results["fourier_coefficients"]
r_g = results["distribution"]
r_speed_3d = results["speed_3d"]

# Visualization with Plotly
# ===============================
print("Generating visualizations with Plotly...")

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Prepare data
G = np.reshape(g, [nbRes, nbRes])
G = np.where(G > 0, G, 0)
X = np.squeeze(xm[0, :, :])
Y = np.squeeze(xm[1, :, :])
color_values = red_channel

# Create subplots with 2 rows and 3 columns
fig = make_subplots(
    rows=2,
    cols=3,
    subplot_titles=(
        "Trajectory in UV Space",
        "Desired Fourier coefficients w_hat",
        "Reproduced Fourier coefficients w",
        "Trajectory on 3D Point Cloud",
        "Reconstruction Error over Time",
        "UV Parameterization (colored by Z)",
    ),
    specs=[
        [{"type": "xy"}, {"type": "heatmap"}, {"type": "heatmap"}],
        [{"type": "scatter3d"}, {"type": "xy"}, {"type": "xy"}],
    ],
    horizontal_spacing=0.08,
    vertical_spacing=0.12,
)

# 1. UV space trajectory with contour
fig.add_trace(
    go.Contour(
        z=G,
        x=X[0, :],
        y=Y[:, 0],
        colorscale="Viridis",
        showscale=False,
        contours=dict(coloring="heatmap"),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=r_x_uv[0, :],
        y=r_x_uv[1, :],
        mode="lines",
        line=dict(color="red", width=2),
        name="Trajectory",
        showlegend=False,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=[r_x_uv[0, 0]],
        y=[r_x_uv[1, 0]],
        mode="markers",
        marker=dict(size=10, color="white", line=dict(color="black", width=2)),
        name="Start",
        showlegend=False,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=[r_x_uv[0, -1]],
        y=[r_x_uv[1, -1]],
        mode="markers",
        marker=dict(
            size=10, color="red", symbol="square", line=dict(color="black", width=2)
        ),
        name="End",
        showlegend=False,
    ),
    row=1,
    col=1,
)

# 2. Desired Fourier coefficients
fig.add_trace(
    go.Heatmap(
        z=np.reshape(w_hat, [nbFct, nbFct]).T,
        colorscale="Gray_r",
        showscale=False,
    ),
    row=1,
    col=2,
)

# 3. Reproduced Fourier coefficients
fig.add_trace(
    go.Heatmap(
        z=np.reshape(r_w[:, -1], [nbFct, nbFct]).T,
        colorscale="Gray_r",
        showscale=False,
    ),
    row=1,
    col=3,
)

# 4. 3D point cloud with trajectory
fig.add_trace(
    go.Scatter3d(
        x=points_3d[:, 0],
        y=points_3d[:, 1],
        z=points_3d[:, 2],
        mode="markers",
        marker=dict(size=2, color=color_values, colorscale="Viridis", opacity=0.3),
        name="Point Cloud",
        showlegend=False,
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter3d(
        x=r_x_3d[0, :],
        y=r_x_3d[1, :],
        z=r_x_3d[2, :],
        mode="lines",
        line=dict(color="red", width=5),
        name="Trajectory",
        showlegend=False,
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter3d(
        x=[r_x_3d[0, 0]],
        y=[r_x_3d[1, 0]],
        z=[r_x_3d[2, 0]],
        mode="markers",
        marker=dict(size=5, color="white", line=dict(color="black", width=2)),
        name="Start",
        showlegend=False,
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter3d(
        x=[r_x_3d[0, -1]],
        y=[r_x_3d[1, -1]],
        z=[r_x_3d[2, -1]],
        mode="markers",
        marker=dict(
            size=5, color="red", symbol="square", line=dict(color="black", width=2)
        ),
        name="End",
        showlegend=False,
    ),
    row=2,
    col=1,
)

# 5. Reconstruction error over time
fig.add_trace(
    go.Scatter(
        x=np.arange(len(r_e)),
        y=r_e,
        mode="lines",
        line=dict(color="blue", width=2),
        name="Error",
        showlegend=False,
    ),
    row=2,
    col=2,
)

# 6. UV parameterization visualization
fig.add_trace(
    go.Scatter(
        x=uv_coords[:, 0],
        y=uv_coords[:, 1],
        mode="markers",
        marker=dict(size=3, color=points_3d[:, 2], colorscale="RdBu", opacity=0.5),
        name="UV coords",
        showlegend=False,
    ),
    row=2,
    col=3,
)
fig.add_trace(
    go.Scatter(
        x=r_x_uv[0, :],
        y=r_x_uv[1, :],
        mode="lines",
        line=dict(color="red", width=2),
        name="Trajectory",
        showlegend=False,
    ),
    row=2,
    col=3,
)

# Update axes labels
fig.update_xaxes(title_text="U", row=1, col=1)
fig.update_yaxes(title_text="V", row=1, col=1)
fig.update_xaxes(showticklabels=False, row=1, col=2)
fig.update_yaxes(showticklabels=False, row=1, col=2)
fig.update_xaxes(showticklabels=False, row=1, col=3)
fig.update_yaxes(showticklabels=False, row=1, col=3)
fig.update_xaxes(title_text="Iteration", row=2, col=2)
fig.update_yaxes(title_text="Error", row=2, col=2)
fig.update_xaxes(title_text="U", row=2, col=3)
fig.update_yaxes(title_text="V", row=2, col=3)

# Update 3D scene
fig.update_scenes(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    zaxis=dict(visible=False),
    aspectmode="data",
    row=2,
    col=1,
)

# Update layout
fig.update_layout(
    height=1000,
    width=1800,
    title_text="Point Cloud Ergodic Control Results",
    showlegend=False,
)

# Save and show
output_html = str(config.get_animation_path("pointcloud_ergodic_control_results_plotly.html"))
fig.write_html(output_html)
print(f"Results saved to '{output_html}'")

# Save individual plots as PDF (excluding 3D point cloud)
print("Saving individual plots as PDF...")

# 1. UV space trajectory
fig_uv = go.Figure()
fig_uv.add_trace(
    go.Contour(
        z=G,
        x=X[0, :],
        y=Y[:, 0],
        colorscale="Viridis",
        showscale=False,
        contours=dict(coloring="heatmap"),
    )
)
fig_uv.add_trace(
    go.Scatter(
        x=r_x_uv[0, :],
        y=r_x_uv[1, :],
        mode="lines",
        line=dict(color="red", width=2),
        showlegend=False,
    )
)
fig_uv.add_trace(
    go.Scatter(
        x=[r_x_uv[0, 0]],
        y=[r_x_uv[1, 0]],
        mode="markers",
        marker=dict(size=10, color="white", line=dict(color="black", width=2)),
        showlegend=False,
    )
)
fig_uv.add_trace(
    go.Scatter(
        x=[r_x_uv[0, -1]],
        y=[r_x_uv[1, -1]],
        mode="markers",
        marker=dict(
            size=10, color="red", symbol="square", line=dict(color="black", width=2)
        ),
        showlegend=False,
    )
)
fig_uv.update_xaxes(title_text="U")
fig_uv.update_yaxes(title_text="V", scaleanchor="x", scaleratio=1)
fig_uv.update_layout(title_text="Trajectory in UV Space", width=600, height=600)
fig_uv.write_image(str(config.get_plot_path("trajectory_uv.pdf")))

# 2. Desired Fourier coefficients
fig_w_hat = go.Figure()
fig_w_hat.add_trace(
    go.Heatmap(
        z=np.reshape(w_hat, [nbFct, nbFct]).T,
        colorscale="Gray_r",
        showscale=False,
    )
)
fig_w_hat.update_xaxes(showticklabels=False)
fig_w_hat.update_yaxes(showticklabels=False)
fig_w_hat.update_layout(
    title_text="Desired Fourier coefficients w_hat", width=500, height=500
)
fig_w_hat.write_image(str(config.get_plot_path("desired_coefficients.pdf")))

# 3. Reproduced Fourier coefficients
fig_w = go.Figure()
fig_w.add_trace(
    go.Heatmap(
        z=np.reshape(r_w[:, -1], [nbFct, nbFct]).T,
        colorscale="Gray_r",
        showscale=False,
    )
)
fig_w.update_xaxes(showticklabels=False)
fig_w.update_yaxes(showticklabels=False)
fig_w.update_layout(
    title_text="Reproduced Fourier coefficients w", width=500, height=500
)
fig_w.write_image(str(config.get_plot_path("reproduced_coefficients.pdf")))

# 4. Reconstruction error over time
fig_error = go.Figure()
fig_error.add_trace(
    go.Scatter(
        x=np.arange(len(r_e)),
        y=r_e,
        mode="lines",
        line=dict(color="blue", width=2),
        showlegend=False,
    )
)
fig_error.update_xaxes(title_text="Iteration")
fig_error.update_yaxes(title_text="Error")
fig_error.update_layout(
    title_text="Reconstruction Error over Time", width=600, height=500
)
fig_error.write_image(str(config.get_plot_path("reconstruction_error.pdf")))

# 5. UV parameterization
fig_uv_param = go.Figure()
fig_uv_param.add_trace(
    go.Scatter(
        x=uv_coords[:, 0],
        y=uv_coords[:, 1],
        mode="markers",
        marker=dict(size=3, color=points_3d[:, 2], colorscale="RdBu", opacity=0.5),
        showlegend=False,
    )
)
fig_uv_param.add_trace(
    go.Scatter(
        x=r_x_uv[0, :],
        y=r_x_uv[1, :],
        mode="lines",
        line=dict(color="red", width=2),
        showlegend=False,
    )
)
fig_uv_param.update_xaxes(title_text="U")
fig_uv_param.update_yaxes(title_text="V", scaleanchor="x", scaleratio=1)
fig_uv_param.update_layout(
    title_text="UV Parameterization (colored by Z)", width=600, height=600
)
fig_uv_param.write_image(str(config.get_plot_path("uv_parameterization.pdf")))

print("Individual plots saved as PDF files")

fig.show("browser")

# Save trajectory data
# ===============================
npz_path = str(config.get_data_path("pointcloud_trajectory.npz"))
np.savez(
    npz_path,
    trajectory_uv=r_x_uv,
    trajectory_3d=r_x_3d,
    reconstruction_error=r_e,
    uv_coords=uv_coords,
    points_3d=points_3d,
    distribution=g,
)
print(f"Trajectory data saved to '{npz_path}'")
