"""
Flow Matching Ergodic Coverage Tutorial (Optimized)
3D coverage using Sinkhorn divergence flow with point mass dynamics

This tutorial uses lqrax (https://github.com/MaxMSun/lqrax/tree/main) to solve
the continuous time Riccati equation for the LQ flow matching problem and ott
(https://github.com/ott-jax/ott) to compute the Sinkhorn divergence.
"""

import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad, jacfwd, jit, vmap
from jax.scipy.stats import gaussian_kde as kde
from jax.scipy.stats import multivariate_normal as mvn

# Configure JAX
jax.config.update("jax_enable_x64", False)

# Device setup - use GPU if available, otherwise CPU
cpu = jax.devices("cpu")[0]
try:
    gpu = jax.devices("gpu")[0]
    print(f"Using GPU: {gpu}")
    compute_device = gpu
except:
    print(f"GPU not available, using CPU: {cpu}")
    compute_device = cpu

jnp.set_printoptions(precision=4)
from lqrax import LQR
# try:
#     from lqrax import LQR
# except:
#     import subprocess

#     subprocess.run(["pip", "install", "lqrax"])
#     from lqrax import LQR

# try:
#     import ott
# except:
#     import subprocess

#     subprocess.run(["pip", "install", "ott-jax"])
#     import ott

from ott.geometry import pointcloud
from ott.geometry.costs import PNormP
from ott.tools.sinkhorn_divergence import sinkhorn_divergence


def visualize_3d_animation(x0, x_traj_list, tgt_samples):
    import plotly.graph_objects as go

    traj_color = "#ff7f0e"
    sample_color = "#9467bd"

    fig = go.Figure()

    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=60),
        plot_bgcolor="white",
        width=800,
        height=600,
    )

    # Target samples
    fig.add_trace(
        go.Scatter3d(
            x=tgt_samples[:, 0],
            y=tgt_samples[:, 1],
            z=tgt_samples[:, 2],
            mode="markers",
            marker=dict(size=3, color=sample_color, opacity=0.1),
            showlegend=False,
        )
    )

    # Final trajectory
    fig.add_trace(
        go.Scatter3d(
            x=x_traj_list[-1][:, 0],
            y=x_traj_list[-1][:, 1],
            z=x_traj_list[-1][:, 2],
            mode="lines+markers",
            line=dict(color=traj_color),
            marker=dict(size=3, color=traj_color),
            showlegend=False,
        )
    )

    # Starting point
    fig.add_trace(
        go.Scatter3d(
            x=[x0[0]],
            y=[x0[1]],
            z=[x0[2]],
            mode="markers",
            marker=dict(color="black", size=10),
            showlegend=False,
        )
    )

    # Build animation frames
    frames = []
    skip = 1
    for i, traj in enumerate(x_traj_list[::skip]):
        frames.append(
            go.Frame(
                name=str(i * skip),
                data=[
                    go.Scatter3d(
                        x=traj[:, 0],
                        y=traj[:, 1],
                        z=traj[:, 2],
                        mode="lines+markers",
                        line=dict(color=traj_color),
                        marker=dict(size=3, color=traj_color),
                        showlegend=False,
                    )
                ],
                traces=[1],
            )
        )
    fig.frames = frames

    # Slider
    steps = [
        dict(
            method="animate",
            args=[
                [str(i)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0),
                ),
            ],
            label=str(i),
        )
        for i in range(len(x_traj_list))
    ]

    fig.update_layout(
        sliders=[
            dict(
                active=len(x_traj_list) - 1,
                y=-0.05,
                x=0.5,
                xanchor="center",
                pad=dict(t=10),
                len=0.5,
                steps=steps,
                currentvalue=dict(prefix="Iteration: ", font=dict(size=12)),
            )
        ]
    )

    # Start / Pause / Reset buttons
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.5,
                y=-0.25,
                xanchor="center",
                yanchor="top",
                pad=dict(r=10, t=10),
                buttons=[
                    dict(
                        label="Start",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=10, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                transition=dict(duration=0),
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="Reset",
                        method="animate",
                        args=[
                            [str(0)],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                transition=dict(duration=0),
                                mode="immediate",
                            ),
                        ],
                    ),
                ],
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            camera=dict(eye=dict(x=0.7, y=0.7, z=0.7)),
            xaxis=dict(range=[-0.6, 0.6], showgrid=False, visible=False),
            yaxis=dict(range=[-0.6, 0.6], showgrid=False, visible=False),
            zaxis=dict(range=[-0.6, 0.6], showgrid=False, visible=False),
            aspectmode="cube",
        ),
        paper_bgcolor="white",
        scene_bgcolor="white",
        showlegend=False,
    )

    fig.show(renderer="browser")


class PointMassLQR(LQR):
    def __init__(self, dt, x_dim, u_dim, Q, R):
        super().__init__(dt, x_dim, u_dim, Q, R)

    def dyn(self, xt, ut):
        return jnp.array(
            [
                xt[3],  # dx = vx
                xt[4],  # dy = vy
                xt[5],  # dz = vz
                ut[0],  # ax
                ut[1],  # ay
                ut[2],  # az
            ]
        )


def main():
    # Load target samples from PLY file
    try:
        from plyfile import PlyData
    except:
        import subprocess

        subprocess.run(["pip", "install", "plyfile"])
        from plyfile import PlyData

    object_name = "plate_shapes"
    # object_name = "bun270_X"
    print(f"object: {object_name}")

    # Load PLY file from local filesystem
    # Assuming script is in tutorials/ directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ply_path = os.path.join(script_dir, f"{object_name}.ply")

    print(f"Loading from: {ply_path}")

    # Read PLY file
    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"].data

    # Extract 3D coordinates
    tgt_samples_dense = np.vstack((vertex["x"], vertex["y"], vertex["z"])).T

    # Extract red channel as target distribution
    # Check if color attributes exist
    if "red" in vertex.dtype.names:
        target_density = np.array(vertex["red"], dtype=float)
    elif "r" in vertex.dtype.names:
        target_density = np.array(vertex["r"], dtype=float)
    else:
        # Fallback to uniform distribution
        target_density = np.ones(len(tgt_samples_dense))
        print("Warning: No red channel found, using uniform distribution")

    # Normalize to create a probability distribution
    target_density = target_density / np.sum(target_density)
    print(
        f"Target density stats - min: {target_density.min():.6f}, max: {target_density.max():.6f}, mean: {target_density.mean():.6f}"
    )

    # Resample points according to red channel intensity
    num_resampled = 2000  # Number of points to resample (increased for better coverage)
    resampled_indices = np.random.choice(
        len(tgt_samples_dense), size=num_resampled, replace=True, p=target_density
    )
    tgt_samples_resampled = tgt_samples_dense[resampled_indices]

    print(f"Original point cloud size: {len(tgt_samples_dense)}")
    print(f"Resampled point cloud size: {len(tgt_samples_resampled)}")

    # Use more resampled points for better coverage
    tgt_samples = tgt_samples_resampled[
        :2000
    ]  # Increased from 1000 for better accuracy
    num_samples = tgt_samples.shape[0]
    print(f"samples.shape (after subsampling): {tgt_samples.shape}")

    # Keep data on CPU, only compute on GPU
    with jax.default_device(cpu):
        tgt_samples = jnp.array(tgt_samples)

    # Setup LQR problem
    Q = jnp.diag(
        jnp.array(
            [
                1.0,
                1.0,
                1.0,
                1e-03,
                1e-03,
                1e-03,
            ]
        )
    )
    R = jnp.diag(jnp.array([0.01, 0.01, 0.1]))

    pointmass_lqr = PointMassLQR(dt=0.05, x_dim=6, u_dim=3, Q=Q, R=R)

    # LQR solving on CPU is typically faster for small problems
    linearize_dyn = jit(pointmass_lqr.linearize_dyn, device=cpu)
    solve_lqr = jit(pointmass_lqr.solve, device=cpu)

    # Initialize trajectory with shorter horizon
    tsteps = 150  # Reduced from 200

    # Set initial position to center of the point cloud
    center = np.mean(tgt_samples_dense, axis=0)
    print(f"Point cloud center: {center}")
    x0 = jnp.array([center[0], center[1], center[2], 0.0, 0.0, 0.01])
    print(f"Initial state x0: {x0}")

    u_traj = jnp.zeros((tsteps, 3))
    x_traj, A_traj, B_traj = linearize_dyn(x0, u_traj)

    # Setup Sinkhorn divergence (default values from notebook)
    def sinkhorn_div(x_samples):
        """Return the OT cost and OT output given a geometry"""
        eps = 0.005
        cost_fn = PNormP(p=1)
        sinkhorn_cost = sinkhorn_divergence(
            pointcloud.PointCloud,
            x_samples[:, :3],
            tgt_samples[:, :3],
            cost_fn=cost_fn,
            epsilon=eps,
        )[0]
        return sinkhorn_cost * 1e3

    # JIT compile the Sinkhorn gradient on compute device (GPU if available)
    sinkhorn_grad = jit(
        lambda _xs: -1.0 * grad(sinkhorn_div)(_xs), device=compute_device
    )

    # Warmup JIT compilation
    print("Warming up JIT compilation...")
    _ = sinkhorn_grad(x_traj)
    print("Warmup complete!")

    # Compute gradient
    sinkhorn_dx_traj = sinkhorn_grad(x_traj)
    sinkhorn_dx_traj = np.array(sinkhorn_dx_traj)
    print(
        f"sinkhorn_dx_traj.shape: {sinkhorn_dx_traj.shape} == x_traj.shape: {x_traj.shape}"
    )

    # Solve the flow matching ergodic coverage problem
    z0 = jnp.zeros(6)
    step_size = 0.001  # Larger step size for faster convergence
    num_iters = 100  # Reduced iterations
    x_traj_list = []

    print("Optimizing trajectory...")
    start_time = time.time()

    for i in range(num_iters):
        if i % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Iteration {i}/{num_iters} - Elapsed: {elapsed:.2f}s")

        x_traj, A_traj, B_traj = linearize_dyn(x0, u_traj)
        sinkhorn_dx_traj = sinkhorn_grad(x_traj)
        v_traj, z_traj = solve_lqr(z0, A_traj, B_traj, sinkhorn_dx_traj)
        u_traj += step_size * v_traj
        x_traj_list.append(np.array(x_traj))

    final_x_traj = pointmass_lqr.traj_sim(x0, u_traj)
    x_traj_list.append(final_x_traj)
    x_traj_list = np.array(x_traj_list)

    total_time = time.time() - start_time
    print(f"Total optimization time: {total_time:.2f}s")
    print(f"Average time per iteration: {total_time/num_iters:.2f}s")

    # Visualize animation with resampled point cloud
    print("Generating visualization...")
    visualize_3d_animation(x0, x_traj_list[::2], tgt_samples_resampled)


if __name__ == "__main__":
    main()
