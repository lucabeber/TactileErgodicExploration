"""
HEDAC (Heat Equation Driven Autonomous Coverage) Exploration - Minimal Version

This script demonstrates HEDAC exploration using a fixed target distribution
(red channel from point cloud) without online GP estimation or target updates.

Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>
"""

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device)
torch.set_default_device(device)

from tactile_ergodic.exploration import PointcloudHEDAC
from tactile_ergodic.utils import config
from tactile_ergodic.utils.plotting_utils import animate_trajectory_pcloud


if __name__ == "__main__":
    # Create HEDAC explorer
    hedac = PointcloudHEDAC(
        obj_name="bun270_X",
        timesteps=1500,
        alpha=100,
        voxel_size=0.002,
        source_strength=1,
        agent_start_index=1500,
        verbose=True,
    )

    # Run exploration
    results = hedac.explore()

    # Save and visualize results
    print("\n" + "=" * 50)
    print("Creating Visualizations")
    print("=" * 50)

    obj_name = "bun270_X"
    pcloud = results["pcloud"]

    # Create goal distribution animation
    goal_html_path = config.get_animation_path(f"hedac_goal_{obj_name}.html")
    animate_trajectory_pcloud(
        x_arr=results["trajectory"],
        vertices=pcloud.vertices,
        color_frames=results["goal_density"][:, None].repeat(
            results["trajectory"].shape[0], axis=1
        ),
        timesteps=results["trajectory"].shape[0],
        save_path=str(goal_html_path),
        is_show=False,
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
