"""
HEDAC (Heat Equation Driven Autonomous Coverage) Exploration - Minimal Version

This is a minimal version that uses a fixed target distribution (red channel from point cloud)
without online GP estimation or target updates.

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

from tactile_ergodic.utils import (
    FirstOrderAgent,
    Pointcloud,
    PointcloudScalarDiffusion,
    config,
)
from tactile_ergodic.utils.plotting_utils import *
from tactile_ergodic.utils.pointcloud_utils import *


def run_hedac_exploration(
    obj_name="bun270_X",
    timesteps=1500,
    alpha=100,
    voxel_size=0.002,
    source_strength=1,
    nb_max_neighbors=500,
    nb_minimum_neighbors=20,
    nb_boundary_neighbors=40,
):
    """
    Run HEDAC exploration with fixed target distribution (red channel).

    Args:
        obj_name: Name of the point cloud file (without .ply extension)
        timesteps: Number of exploration timesteps
        alpha: Heat equation parameter
        voxel_size: Voxel size for point cloud downsampling
        source_strength: Strength of heat equation source term
        nb_max_neighbors: Maximum number of neighbors to consider
        nb_minimum_neighbors: Minimum number of neighbors required
        nb_boundary_neighbors: Number of neighbors for boundary detection

    Returns:
        dict: Exploration results containing trajectory, coverage, heat, etc.
    """
    # Setup point cloud
    print("=" * 50)
    print("Loading and Processing Point Cloud")
    print("=" * 50)

    class Param:
        pass

    param = Param()
    param.voxel_size = voxel_size
    param.alpha = alpha

    ply_path = str(config.get_point_cloud_path(f"{obj_name}.ply"))
    pcloud = process_point_cloud(ply_path, param)
    print(f"Point cloud processed: {len(pcloud.vertices)} points")

    # Calculate dt and h for heat equation
    pcloud.dt, pcloud.h = calculate_dt(pcloud.vertices, alpha)
    print(f"dt: {pcloud.dt:.3e}, h: {pcloud.h:.3e}, s: {voxel_size:.3e}")

    # Setup point cloud helper
    pcd_helper = Pointcloud(pcloud.vertices)
    boundary_normals = pcd_helper.get_boundary_normals()

    # Setup heat equation solver
    scalar_diffusion_solver = PointcloudScalarDiffusion(pcloud=pcd_helper)

    # Setup Laplacian matrices
    pcloud.C, pcloud.M = robust_laplacian.point_cloud_laplacian(
        pcloud.vertices, n_neighbors=nb_boundary_neighbors
    )
    A = csc_matrix(pcloud.M + pcloud.dt * pcloud.C)
    pcloud.A_factorized = splu(A)

    # Use red channel as fixed goal density
    goal_density = normalize_mat(pcloud.u0.copy())
    print(
        f"Goal density from red channel: min={goal_density.min():.3f}, max={goal_density.max():.3f}"
    )

    # Initialize agent
    print("\n" + "=" * 50)
    print("Initializing Agent")
    print("=" * 50)

    agent_radius = 2.5 * voxel_size
    max_velocity = 0.1 * voxel_size * 2
    max_acceleration = 1.0 * max_velocity * 2

    # Initialize agent
    agent = FirstOrderAgent(
        x=pcloud.vertices[1500].copy(),  # Start at 1500th vertex
        max_velocity=max_velocity,
        dim_t=timesteps,
    )
    agent.radius = agent_radius

    print(f"Agent initialized at vertex 1500: {agent.x}")
    print(f"Agent radius: {agent_radius:.3e}")

    # Initialize arrays
    print("\n" + "=" * 50)
    print("Running HEDAC Exploration")
    print("=" * 50)

    coverage_arr = np.zeros((len(pcloud.vertices), timesteps))
    heat_arr = np.zeros_like(coverage_arr)
    speed_arr = np.zeros(timesteps)

    # Initialize coverage
    coverage = np.zeros_like(goal_density)
    ut = np.array(goal_density)
    time_arr = np.zeros(timesteps)

    agent.t = 0

    # Main exploration loop
    for t in range(timesteps):
        # Get neighbors
        dists, neighbor_ids, neighbor_coords = get_pcloud_neighbors(
            pcloud.pcd_tree,
            pcloud.vertices,
            np.copy(agent.x),
            agent.radius,
            nb_max_neighbors,
            nb_minimum_neighbors,
        )

        # Update coverage
        kernel_vals = np.exp(-(1 / agent.radius) * dists**2)
        coverage[neighbor_ids] += kernel_vals

        # Compute gradient direction
        neighbor_ids = neighbor_ids[:nb_minimum_neighbors]
        dists = dists[:nb_minimum_neighbors]
        neighbor_coords = pcloud.vertices[neighbor_ids, :]

        coverage_density = normalize_mat(coverage)
        source = np.maximum(goal_density - coverage_density, 0) ** 2
        source = normalize_mat(source)

        start_time = time.time()
        ut = pcloud.A_factorized.solve(pcloud.M @ ut)
        time_arr[t] = time.time() - start_time

        ut += source_strength * source
        ut[pcd_helper.is_boundary_arr] = 0
        scalar_diffusion_solver.get_gradient(ut)

        # Get gradient at agent location and project agent back to surface
        (agent.x,) = get_gradient(
            np.copy(agent.x),
            neighbor_coords,
            neighbor_ids,
            ut,
        )

        gradient = np.mean(
            scalar_diffusion_solver.gradient_ut_3d[neighbor_ids[:10]], axis=0
        )

        # Track speed
        prev_x = np.copy(agent.x)
        agent.update(gradient)
        displacement = agent.x - prev_x
        speed_arr[t] = np.linalg.norm(displacement)

        # Project agent back to surface after update (critical for staying on manifold!)
        dists_proj, neighbor_ids_proj, neighbor_coords_proj = get_pcloud_neighbors(
            pcloud.pcd_tree,
            pcloud.vertices,
            np.copy(agent.x),
            agent.radius,
            nb_max_neighbors,
            nb_minimum_neighbors,
        )
        neighbor_ids_proj = neighbor_ids_proj[:nb_minimum_neighbors]
        neighbor_coords_proj = pcloud.vertices[neighbor_ids_proj, :]

        (agent.x,) = get_gradient(
            np.copy(agent.x),
            neighbor_coords_proj,
            neighbor_ids_proj,
            ut,
        )

        # Store results
        coverage_arr[:, t] = coverage
        heat_arr[:, t] = ut

        # Progress update
        if (t + 1) % 100 == 0:
            print(
                f"Step {t+1}/{timesteps}, speed: {speed_arr[t]:.3e}, "
                f"solve time: {time_arr[t]:.3e}s"
            )

    print("\n" + "=" * 50)
    print("Exploration Complete!")
    print("=" * 50)
    print(f"Average speed: {speed_arr.mean():.3e}")
    print(f"Average solve time: {time_arr.mean():.3e}s")

    return {
        "trajectory": agent.x_arr[: agent.t, :],  # Shape: (timesteps, 3)
        "heat": heat_arr,
        "coverage": coverage_arr,
        "time": time_arr,
        "goal_density": goal_density,
        "speed": speed_arr,
        "pcloud": pcloud,
    }


if __name__ == "__main__":
    # Run exploration
    results = run_hedac_exploration(
        obj_name="bun270_X",
        timesteps=1500,
    )

    # Save and visualize results
    obj_name = "bun270_X"
    pcloud = results["pcloud"]

    print("\n" + "=" * 50)
    print("Creating Visualizations")
    print("=" * 50)

    # Create goal distribution animation (static, single frame)
    goal_html_path = config.get_animation_path(f"hedac_goal_{obj_name}.html")
    # Replicate goal density to create animation frames
    goal_density_frames = np.tile(
        results["goal_density"][:, None], (1, results["trajectory"].shape[0])
    )
    animate_trajectory_pcloud(
        x_arr=results["trajectory"],
        vertices=pcloud.vertices,
        color_frames=goal_density_frames,
        timesteps=results["trajectory"].shape[0],
        save_path=str(goal_html_path),
    )
    print(f"Goal distribution animation saved to: {goal_html_path}")

    # Create coverage evolution animation
    coverage_html_path = config.get_animation_path(f"hedac_coverage_{obj_name}.html")
    animate_trajectory_pcloud(
        x_arr=results["trajectory"],
        vertices=pcloud.vertices,
        color_frames=results["coverage"],
        timesteps=results["trajectory"].shape[0],
        save_path=str(coverage_html_path),
        is_show=True,
    )
    print(f"Coverage animation saved to: {coverage_html_path}")

    print("\nDone!")
