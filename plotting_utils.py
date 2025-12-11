"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of diffusionVirtualFixtures.

diffusionVirtualFixtures is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 3 as
published by the Free Software Foundation.

diffusionVirtualFixtures is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with diffusionVirtualFixtures. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import open3d as o3d
import plotly.graph_objects as go


def show_plot(plots, camera_params=None, showlegend=True):
    layout = go.Layout(scene=dict(aspectmode="data"))
    fig = go.Figure(data=plots, layout=layout)
    if camera_params is None:
        camera_params = dict(
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0.0, y=0.0, z=2.0),  # plane
            # eye=dict(x=0.0, y=0.0, z=0.5),  # others
        )
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        showlegend=showlegend,
        scene_camera=camera_params,
    )

    # fig.update(layout_coloraxis_showscale=False)
    return fig


def visualize_trajectory(
    x_arr,
    plots=None,
    color="black",
    legendgroup=None,
    showlegend=True,
    experiment_index=0,
    is_show_plot=True,
):
    """
    Visualizes a point cloud in a 3D scatter plot.

    Args:
        vertices (numpy.ndarray): The vertices of the point cloud.
        colors (numpy.ndarray, optional): The colors of the points. Defaults to None.
        plots (list, optional): The existing plots to be updated. Defaults to None.
        point_size (int, optional): The size of the points in the scatter plot.
        Defaults to 2.
    """
    if plots is None:
        plots = []
    initial_position_plot = go.Scatter3d(
        x=[x_arr[0, 0]],
        y=[x_arr[0, 1]],
        z=[x_arr[0, 2]],
        legendgroup=legendgroup,
        showlegend=False,
        name=f"{experiment_index}",
        # legendgroup=legendgroup,
        marker=dict(
            size=8,
            color="green",
        ),
    )
    plots.append(initial_position_plot)
    trajectory_plot = go.Scatter3d(
        x=x_arr[:, 0],
        y=x_arr[:, 1],
        z=x_arr[:, 2],
        mode="lines",  # Change mode to "lines"
        name=f"Agent Trajectory {legendgroup}",
        showlegend=showlegend,
        legendgroup=legendgroup,
        line=dict(width=10, color=color),
        # line=dict(width=8),
    )
    plots.append(trajectory_plot)
    final_position_plot = go.Scatter3d(
        x=[x_arr[-1, 0]],
        y=[x_arr[-1, 1]],
        z=[x_arr[-1, 2]],
        legendgroup=legendgroup,
        showlegend=False,
        marker=dict(
            size=4,
            color="magenta",
        ),
    )
    plots.append(final_position_plot)

    if is_show_plot:
        return show_plot(plots)
    else:
        return plots


def visualize_point_cloud(
    vertices, colors=None, plots=None, is_show_plot=True, point_size=3, voxel_size=None
):
    """
    Visualizes a point cloud in a 3D scatter plot.

    Args:
        vertices (numpy.ndarray): The vertices of the point cloud.
        colors (numpy.ndarray, optional): The colors of the points. Defaults to None.
        plots (list, optional): The existing plots to be updated. Defaults to None.
        point_size (int, optional): The size of the points in the scatter plot.
          Defaults to 2.
    """
    if plots is None:
        plots = []
    if colors is None:
        color_arr = np.zeros((len(vertices), 3))
        colors = color_arr
    elif colors.ndim == 1:
        # print("1D colors")
        color_arr = np.zeros((len(vertices), 3))
        color_arr[:, 0] = colors
    else:
        color_arr = colors

    # first construct the open3d point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.colors = o3d.utility.Vector3dVector(color_arr)
    if voxel_size is not None:
        # here we downsample the point cloud to have a smaller plot to upload to plotly
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)  # downsample
    vertices = np.asarray(pcd.points)
    color_arr = np.asarray(pcd.colors)

    marker = dict(
        size=point_size,    
        showscale=True,
        opacity=0.8,
    )
    if colors.ndim == 2:
        # print("3D colors")
        colors = colors.astype(int).astype(str)
        # Create a scatter3d trace with colors
        color_string = [f'rgb({",".join(c)})' for c in colors]
        marker["color"] = color_string

    else:

        marker["color"] = color_arr[:, 0]
        colorscale = "bluered"
        # colorscale="jet"
        # colorscale="rainbow"
        # colorscale = "turbo"
        marker["colorscale"] = colorscale

    scatter = go.Scatter3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        mode="markers",
        name="Point Cloud",
        marker=marker,
    )

    plots.append(scatter)
    if is_show_plot:
        return show_plot(plots)

    else:
        return plots


def visualize_gradient_field(
    vertices,
    gradient_arr,
    plots=None,
    legendgroup=None,
    showlegend=True,
    is_show_plot=True,
    sizeref=1,
):

    # Visualize the gradient field
    # ==============================================================================
    if plots is None:
        plots = []  # we will append the plots to this list]

    # max_val = np.max(ut)
    # scaled_gradient_arr = gradient_arr * (max_val - ut)[:, None]
    # scaled_gradient_arr = gradient_arr

    gradient_field_plot = go.Cone(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        u=gradient_arr[:, 0],
        v=gradient_arr[:, 1],
        w=gradient_arr[:, 2],
        # sizemode="absolute",
        sizeref=sizeref,
    )
    plots.append(gradient_field_plot)
    if is_show_plot:
        return show_plot(plots)

    else:
        return plots


def streamline_plot(
    vertices,
    gradient_arr,
    plots=None,
    legendgroup=None,
    showlegend=True,
    is_show_plot=True,
):

    # Visualize the gradient field
    # ==============================================================================
    if plots is None:
        plots = []  # we will append the plots to this list]

    streamline_plot = go.Streamtube(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        u=gradient_arr[:, 0],
        v=gradient_arr[:, 1],
        w=gradient_arr[:, 2],
    )
    plots.append(streamline_plot)
    if is_show_plot:
        return show_plot(plots)

    else:
        return plots


def animate_trajectory_pcloud(
    x_arr,
    vertices,
    color_frames,
    timesteps,
    circle_radius=0.1,
    look_step=50,
    save_path=None,
):
    point_size = 10
    # Set the camera
    camera_params = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0.0, y=0.0, z=1.2),  # Change the z value to -2 to view from the back
    )

    # Initial Point Cloud (static positions, dynamic color)
    point_cloud = go.Scatter3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        mode="markers",
        marker=dict(
            size=point_size,
            opacity=0.8,
            color=color_frames[..., 0],  # Use the first frame's colors
            colorscale="bluered",
        ),
        name="Reconstructed Target",
    )

    # Initial Trajectory Marker
    trajectory = go.Scatter3d(
        x=[x_arr[0, 0]],
        y=[x_arr[0, 1]],
        z=[x_arr[0, 2]],
        mode="markers",
        marker=dict(size=10, color="green"),
        name="Trajectory",
        opacity=0.3,  # Set the opacity of the line
    )
    # Add a dummy circle to start, even if invisible
    empty_circle = go.Scatter3d(
        x=[],
        y=[],
        z=[],
        mode="lines",
        line=dict(dash="dash", color="red", width=4),
        name="Circle",
    )

    # Create Figure
    fig = go.Figure(
        data=[point_cloud, trajectory, empty_circle],
        layout=go.Layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data",
            ),
            showlegend=False,
            scene_camera=camera_params,
        ),
    )

    # Animation Frames (Update Trajectory + Colors)
    timestep_multiplier = 100
    n_frames = timesteps // timestep_multiplier

    # Precompute unit circle in x-y plane
    theta = np.linspace(0, 2 * np.pi, 40)
    unit_circle = (
        np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)
        * circle_radius
    )  # radius 0.1
    all_circle_points = []

    # Build frames with optional circle overlay
    frames = []
    for k in range(n_frames):
        frame_data = []

        # Point cloud with updated color
        frame_data.append(
            go.Scatter3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                mode="markers",
                marker=dict(
                    size=point_size,
                    opacity=0.9,
                    color=color_frames[..., k * timestep_multiplier - 1],
                    colorscale="bluered",
                ),
                name="Reconstructed Target",
            )
        )

        # Trajectory so far
        frame_data.append(
            go.Scatter3d(
                x=x_arr[: k * timestep_multiplier, 0],
                y=x_arr[: k * timestep_multiplier, 1],
                z=x_arr[: k * timestep_multiplier, 2],
                mode="lines",
                line=dict(width=5, color="black"),
                name="Trajectory",
                opacity=0.4,
            )
        )

        # # Optional: Add dashed circle every 30 steps (but not at frame 0)
        # if (k * timestep_multiplier) % look_step == 0 and k > 0:
        #     center = x_arr[k * timestep_multiplier]
        #     circle_points = unit_circle + center  # shape (100, 3)
        #     all_circle_points.append(circle_points)
        #     circle_points = np.array(all_circle_points).reshape(-1, 3)

        #     frame_data.append(
        #         go.Scatter3d(
        #             x=circle_points[:, 0],
        #             y=circle_points[:, 1],
        #             z=circle_points[:, 2],
        #             mode="markers",
        #             marker=dict(size=2, color="yellow", opacity=0.5),
        #             name="Circle",
        #         )
        #     )
        #     trace_ids = [0, 1, 2]
        # else:
        #     trace_ids = [0, 1, 2]

        # Add frame
        frames.append(go.Frame(data=frame_data, name=f"frame{k}", traces=[0,1,2]))
    fig.update(frames=frames)

    # Sliders
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

    fig.update_layout(width=1200, height=800, sliders=sliders)

    if save_path:
        fig.write_html(save_path)

    fig.show()


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_distribution_evolution_column_auto(vertices,
                                            original_density,
                                            estimated_density_arr,
                                            pdf_name="distribution_evolution_column.pdf",
                                            agent_trajectory=None):
    """
    Creates a 1-column × 6-row figure.
    Row 1 = original distribution
    Rows 2–6 = estimated distributions automatically chosen at 5 evenly-spaced steps.
    No labels, no titles, no ticks, no grid.
    """

    num_steps = estimated_density_arr.shape[1]

    # Automatically pick 5 evenly spaced steps (excluding the last one for safety)
    steps = np.linspace(0, num_steps - 1, 6, dtype=int)[1:]  # skip the first (original)

    # Height = 4 cm
    height_in = 4 / 2.54

    fig, axes = plt.subplots(
        nrows=1, ncols=6,
        # figsize=(3, height_in),  # 3 inch width can be adjusted
        constrained_layout=True
    )

    # Remove grid, ticks, labels, titles
    for ax in axes:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        # Remove axes frame/box and make axes backgrounds transparent
        fig.patch.set_alpha(0)
        for ax in axes:
            ax.set_frame_on(False)
            ax.patch.set_alpha(0)
            for spine in ax.spines.values():
                spine.set_visible(False)
    # Column 1: original distribution
    axes[0].scatter(vertices[:, 0], vertices[:, 1],
                    c=original_density, s=1.5, cmap="viridis")

    # Columns 2–6: estimated distributions
    for i, step in enumerate(steps):
        axes[i + 1].scatter(
            vertices[:, 0],
            vertices[:, 1],
            c=estimated_density_arr[:, step],
            s=1.5,  # decreased point size
            cmap="viridis",
            # linewidths=0,
            # alpha=1.5,
        )

        # Overlay the agent trajectory in black (if provided)
        if agent_trajectory is not None:
            traj = np.asarray(agent_trajectory)
            if traj.ndim == 2 and traj.shape[1] >= 2:
                axes[i + 1].plot(traj[:step, 0], traj[:step, 1], color="r", linewidth=0.8, zorder=10,alpha=0.6)
            elif traj.ndim == 1 and traj.size >= 2:
                axes[i + 1].plot(traj[0], traj[1], marker="o", color="r", zorder=10)

    # Save to PDF
    with PdfPages(pdf_name) as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0, dpi=600, transparent=True)

    plt.close(fig)
    print(f"Saved '{pdf_name}' with steps: {steps.tolist()}")


