"""
Laplacian SMC Ergodic Exploration (without GP estimation)

This script runs pure ergodic control using Laplacian eigenfunctions
without online GP density estimation. The target distribution comes
directly from the point cloud's color/texture data.

Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>
"""

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)
torch.set_default_device(device)

from tactile_ergodic.utils import config
from tactile_ergodic.utils.pointcloud_utils import process_point_cloud
from tactile_ergodic.exploration.laplacian_smc import (
    LaplacianSMCController,
    compute_laplacian_eigenpairs,
    normalize_density_from_coeffs,
)

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def run_surface_exploration(
    ply_path: str = "bun270_X.ply",
    num_eigen: int = 200,
    n_steps: int = 1200,
    seed: int = 0,
    dt: float = 0.05,
    u_max: float = 0.1,
    rho_u: float = 1e-3,
    knn: int = 30,
    voxel_size: float = 0.002,
):
    """
    Loads the point cloud, computes Laplacian eigenpairs, and runs
    the Laplacian SMC ergodic controller.
    """
    rng = np.random.default_rng(seed)

    # Create parameter object for process_point_cloud
    class Param:
        def __init__(self):
            self.voxel_size = voxel_size
            self.alpha = 1  # scaling factor for dt calculation

    param = Param()
    filename = config.get_point_cloud_path(ply_path)
    pcloud = process_point_cloud(filename, param)
    points = pcloud.vertices

    # Target distribution: red channel (already normalized in process_point_cloud)
    phi = pcloud.u0.astype(float)
    max_phi = float(np.max(phi))
    if max_phi > 0:
        phi /= max_phi
    else:
        raise ValueError("Red channel is all zeros; cannot build a target density.")

    lambdas, eigenvecs, M = compute_laplacian_eigenpairs(points, num_eigen)

    controller = LaplacianSMCController(
        points_N3=points,
        M_NN=M,
        F_NK=eigenvecs,
        lambdas_K=lambdas,
        phi_N=phi,
        dt=dt,
        u_max=u_max,
        rho_u=rho_u,
        knn=knn,
    )

    # Run control
    traj = np.zeros((n_steps + 1, 3), dtype=float)
    traj[0] = points[rng.integers(len(points))]

    est_density = np.zeros((len(points), n_steps + 1), dtype=float)
    est_density[:, 0] = normalize_density_from_coeffs(controller.F, controller.mu_k, M)

    # Track coefficient evolution
    mu_k_history = np.zeros((controller.K, n_steps + 1), dtype=float)
    mu_k_history[:, 0] = controller.mu_k.copy()

    debug = []

    for t in range(1, n_steps + 1):
        x_next, u, dbg = controller.step(traj[t - 1])
        traj[t] = x_next
        est_density[:, t] = normalize_density_from_coeffs(
            controller.F, controller.mu_k, M
        )
        mu_k_history[:, t] = controller.mu_k.copy()
        debug.append(dbg)

    # Color frames use the fixed target distribution (red channel)
    color_frames = np.repeat(phi[:, None], n_steps + 1, axis=1)

    return {
        "points": points,
        "colors": phi,  # Use red channel (u0) as colors
        "phi": phi,
        "phi_k": controller.phi_k,
        "mu_k_history": mu_k_history,
        "trajectory": traj,
        "estimated_density": est_density,
        "color_frames": color_frames,
        "debug": debug,
        "num_eigen": num_eigen,
    }


def animate_with_coefficients(
    results, save_path=None, is_show=True
):
    """
    Creates an animation with 3 panels:
    - 3D trajectory on point cloud
    - Desired Laplacian eigenfunction coefficients (phi_k)
    - Reproduced coefficients evolution (mu_k over time)
    """
    traj = results["trajectory"]
    points = results["points"]
    color_frames = results["color_frames"]
    phi_k = results["phi_k"]
    mu_k_history = results["mu_k_history"]
    num_eigen = results["num_eigen"]

    # Animation parameters
    point_size = 2
    timestep_multiplier = 10
    n_steps = traj.shape[0]
    n_frames = (n_steps + timestep_multiplier - 1) // timestep_multiplier

    # Reshape coefficients for visualization (use square root for approximate square)
    nrows = int(np.sqrt(num_eigen))
    ncols = (num_eigen + nrows - 1) // nrows
    phi_k_img = np.zeros((nrows, ncols))
    phi_k_img.flat[:num_eigen] = phi_k

    # Camera settings
    camera_params = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0.0, y=0.0, z=1.2),
    )

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=3,
        column_widths=[0.5, 0.25, 0.25],
        subplot_titles=("3D Trajectory", "Desired φ_k", "Reproduced μ_k(t)"),
        specs=[[{"type": "scatter3d"}, {"type": "heatmap"}, {"type": "heatmap"}]],
    )

    # Initial point cloud
    point_cloud = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(
            size=point_size,
            opacity=0.5,
            color=color_frames[:, 0],
            colorscale="viridis",
        ),
        showlegend=False,
    )

    # Initial trajectory
    trajectory_trace = go.Scatter3d(
        x=[traj[0, 0]],
        y=[traj[0, 1]],
        z=[traj[0, 2]],
        mode="lines",
        line=dict(width=5, color="red"),
        opacity=0.4,
        showlegend=False,
    )

    # Desired coefficients (static)
    desired_coeffs = go.Heatmap(
        z=phi_k_img,
        colorscale="RdBu",
        showscale=False,
        zmid=0,
    )

    # Initial reproduced coefficients
    mu_k_img_0 = np.zeros((nrows, ncols))
    mu_k_img_0.flat[:num_eigen] = mu_k_history[:, 0]
    reproduced_coeffs = go.Heatmap(
        z=mu_k_img_0,
        colorscale="RdBu",
        showscale=False,
        zmid=0,
    )

    # Add traces
    fig.add_trace(point_cloud, row=1, col=1)
    fig.add_trace(trajectory_trace, row=1, col=1)
    fig.add_trace(desired_coeffs, row=1, col=2)
    fig.add_trace(reproduced_coeffs, row=1, col=3)

    # Update 3D scene
    fig.update_scenes(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="data",
        camera=camera_params,
    )

    # Remove tick labels from heatmaps
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_xaxes(showticklabels=False, row=1, col=3)
    fig.update_yaxes(showticklabels=False, row=1, col=3)

    # Create animation frames
    frames = []
    for k in range(n_frames):
        frame_idx = min(k * timestep_multiplier, n_steps - 1)

        # Point cloud
        pc_frame = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(
                size=point_size,
                opacity=0.5,
                color=color_frames[:, frame_idx],
                colorscale="viridis",
            ),
            showlegend=False,
        )

        # Trajectory so far
        traj_frame = go.Scatter3d(
            x=traj[:frame_idx, 0],
            y=traj[:frame_idx, 1],
            z=traj[:frame_idx, 2],
            mode="lines",
            line=dict(width=5, color="red"),
            opacity=1.0,
            showlegend=False,
        )

        # Desired coefficients (unchanged)
        desired_frame = go.Heatmap(
            z=phi_k_img,
            colorscale="RdBu",
            showscale=False,
            zmid=0,
        )

        # Reproduced coefficients at current timestep
        mu_k_img_t = np.zeros((nrows, ncols))
        mu_k_img_t.flat[:num_eigen] = mu_k_history[:, frame_idx]
        reproduced_frame = go.Heatmap(
            z=mu_k_img_t,
            colorscale="RdBu",
            showscale=False,
            zmid=0,
        )

        frames.append(
            go.Frame(
                data=[pc_frame, traj_frame, desired_frame, reproduced_frame],
                name=f"frame{k}",
                traces=[0, 1, 2, 3],
            )
        )

    fig.update(frames=frames)

    # Add slider
    sliders = [
        dict(
            steps=[
                dict(
                    method="animate",
                    args=[
                        [f"frame{k}"],
                        dict(
                            mode="immediate",
                            frame=dict(duration=400, redraw=True),
                            transition=dict(duration=0),
                        ),
                    ],
                    label=f"{k+1}",
                )
                for k in range(n_frames)
            ],
            active=0,
            transition=dict(duration=0),
            x=0,
            y=0,
            currentvalue=dict(
                font=dict(size=12), prefix="frame: ", visible=True, xanchor="center"
            ),
            len=1.0,
        )
    ]

    fig.update_layout(
        width=1800,
        height=600,
        sliders=sliders,
        showlegend=False,
    )

    # Save and/or show
    if save_path:
        fig.write_html(save_path)
        print(f"Animation saved to '{save_path}'")

    if is_show:
        fig.show("browser")

    return fig


if __name__ == "__main__":
    results = run_surface_exploration()
    save_path = str(config.get_animation_path("laplacian_smc_exploration_bun270_X.html"))
    animate_with_coefficients(
        results,
        save_path=save_path,
        is_show=True,
    )
    print(
        f"Finished {results['trajectory'].shape[0]-1} steps. "
        f"Final ergodic metric: {results['debug'][-1]['E']:.4f}"
    )
