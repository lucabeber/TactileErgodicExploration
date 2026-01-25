"""
HEDAC (Heat Equation Driven Autonomous Coverage) with GP Density Estimation

This script demonstrates HEDAC exploration with online Gaussian Process-based
density estimation. The agent learns the target distribution online and balances
exploration (high uncertainty areas) with exploitation (known high-density areas).

Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>
"""

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)
torch.set_default_device(device)

from tactile_ergodic.exploration import HEDACEstimation
from tactile_ergodic.utils import config
from tactile_ergodic.utils.plotting_utils import (
    animate_trajectory_pcloud,
    visualize_point_cloud,
    visualize_trajectory,
    plot_distribution_evolution_column_auto,
)


if __name__ == "__main__":
    # Select the object to explore
    obj_name = "bun270_X"  # Stanford bunny with X projected as the target
    # obj_name = "plate_shapes"  # random IKEA plate with hand-drawn shapes
    # obj_name = "cup_X"

    # Create HEDAC explorer with GP estimation
    hedac = HEDACEstimation(
        obj_name=obj_name,
        timesteps=1500,
        alpha=100,
        voxel_size=0.002,
        source_strength=1,
        nb_max_neighbors=500,
        nb_minimum_neighbors=20,
        nb_boundary_neighbors=40,
        agent_start_index=1500,
        exploit_alpha=0.6,  # Balance exploration vs exploitation
        gp_update_interval=50,  # Update GP every 50 steps
        verbose=True,
    )

    # Run exploration
    print("Starting HEDAC exploration with GP estimation...")
    results = hedac.explore()

    # Extract results
    x_arr = results["trajectory"]
    goal_arr = results["goal_density"]
    estimated_density_arr = results["estimated_density"]
    pcloud = results["pcloud"]

    # Visualize final estimated density
    print("\nGenerating visualizations...")
    plots = visualize_point_cloud(
        pcloud.vertices,
        colors=estimated_density_arr[..., -1],
        is_show_plot=False,
        point_size=5,
    )
    fig = visualize_trajectory(x_arr, plots, color="black")
    fig.show("browser")

    # Generate animations
    print("\nGenerating animated visualizations...")

    # Goal density animation (exploration target evolving over time)
    goal_html_path = config.get_animation_path(f"hedac_estimation_goal_{obj_name}.html")
    animate_trajectory_pcloud(
        x_arr=x_arr,
        vertices=pcloud.vertices,
        color_frames=goal_arr,
        timesteps=x_arr.shape[0],
        save_path=str(goal_html_path),
        is_show=False,
    )
    print(f"Goal density animation saved to: {goal_html_path}")

    # Estimated density animation (GP's learned distribution)
    est_html_path = config.get_animation_path(
        f"hedac_estimation_estimated_{obj_name}.html"
    )
    animate_trajectory_pcloud(
        x_arr=x_arr,
        vertices=pcloud.vertices,
        color_frames=estimated_density_arr,
        timesteps=x_arr.shape[0],
        save_path=str(est_html_path),
        is_show=False,
    )
    print(f"Estimated density animation saved to: {est_html_path}")

    # Coverage evolution animation
    if "coverage" in results:
        coverage_html_path = config.get_animation_path(
            f"hedac_estimation_coverage_{obj_name}.html"
        )
        animate_trajectory_pcloud(
            x_arr=x_arr,
            vertices=pcloud.vertices,
            color_frames=results["coverage"],
            timesteps=x_arr.shape[0],
            save_path=str(coverage_html_path),
            is_show=True,
        )
        print(f"Coverage animation saved to: {coverage_html_path}")

    # Plot distribution evolution
    plot_distribution_evolution_column_auto(
        vertices=pcloud.vertices,
        original_density=hedac.density_estimator.get_initial_mean(),
        estimated_density_arr=estimated_density_arr,
        pdf_name=str(config.get_plot_path(f"hedac_estimation_{obj_name}.pdf")),
        agent_trajectory=x_arr,
    )

    print("\nExploration complete!")
