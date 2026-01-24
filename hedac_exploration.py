"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of tactileErgodicExploration.

tactileErgodicExploration is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 3 as
published by the Free Software Foundation.

tactileErgodicExploration is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with tactileErgodicExploration. If not, see <http://www.gnu.org/licenses/>.
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

import config
from gpr_on_point_cloud import *
from plotting_utils import *
from pointcloud import Pointcloud
from pointcloud_scalar_diffusion import PointcloudScalarDiffusion
from pointcloud_utils import *
from virtual_agents import FirstOrderAgent, SecondOrderAgent


def hedac(agent, param, pcloud):
    """
    Perform HEDAC exploration using the agent and the point cloud.

    Original implementation on a 2-D rectangular grid by Ivic et al.
    Ivić, S., Crnković, B., & Mezić, I. (2017). Ergodicity-Based Cooperative
    Multiagent Area Coverage via a Potential Field. IEEE Transactions on
    Cybernetics, 47(8), 1983–1993. https://doi.org/10.1109/TCYB.2016.2634400

    Args:
        agent (Agent): The agent object representing the virtual exploration agent.
        param (Parameters): The parameters for the HEDAC algorithm.
        pcloud (PointCloud): The point cloud object representing the exploration
             domain and target

    Returns:
        tuple: A tuple containing the agent's trajectory, heat array,
            coverage array, and time array.
    """
    coverage_arr = np.zeros((len(pcloud.vertices), param.timesteps))
    heat_arr = np.zeros_like(coverage_arr)
    goal_density_arr = np.zeros_like(coverage_arr)
    estimated_density_arr = np.zeros_like(coverage_arr)

    # Array to store 3D speeds at each timestep
    speed_arr = np.zeros(param.timesteps)

    # ===================================================================
    # INITIALIZATION: Create online GP model for density estimation
    # ===================================================================
    # 1. Sample ground truth density at agent's initial position
    sample_points = torch.tensor(agent.x, dtype=torch.float32).reshape(1, -1)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        gpr_original_density = likelihood_real(model_real(sample_points))
    density_sample = gpr_original_density.mean

    # 2. Initialize online GP with this single sample
    train_x = sample_points.clone()
    train_y = density_sample.clone()

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPROnPointCloud(train_x, train_y, likelihood, km, pcloud.vertices)
    model.train()
    likelihood.train()
    model.eval()
    likelihood.eval()

    # 3. Predict on full point cloud to get initial goal density
    test_x = torch.tensor(pcloud.vertices, dtype=torch.float32, device=device)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))

    goal_density = observed_pred.mean.cpu().numpy()
    goal_density = normalize_mat(goal_density)  # Normalize to probability distribution
    mean_tmp = normalize_mat(observed_pred.mean.cpu().numpy())
    ut = np.array(goal_density)  # Heat equation state

    # we keep this and add coverage at each timestep on top of it
    coverage = np.zeros_like(goal_density)

    # for keeping the runtime of each timestep
    time_arr = np.zeros(param.timesteps)

    agent.t = 0

    # ===================================================================
    # MAIN EXPLORATION LOOP
    # ===================================================================
    for t in range(param.timesteps):
        # 1. Get neighbors around agent's current position
        dists, neighbor_ids, neighbor_coords = get_pcloud_neighbors(
            pcloud.pcd_tree,
            pcloud.vertices,
            np.copy(agent.x),
            agent.radius,
            param.nb_max_neighbors,
            param.nb_minimum_neighbors,
        )

        # 2. Update coverage map (RBF kernel around agent)
        kernel_vals = np.exp(-(1 / agent.radius) * dists**2)
        coverage[neighbor_ids] += kernel_vals

        # 3. Compute heat equation source term
        neighbor_ids = neighbor_ids[: param.nb_minimum_neighbors]
        dists = dists[: param.nb_minimum_neighbors]
        neighbor_coords = pcloud.vertices[neighbor_ids, :]
        coverage_density = normalize_mat(coverage)
        source = np.maximum(goal_density - coverage_density, 0) ** 2
        source = normalize_mat(source)

        # 4. Evolve heat equation (one timestep)
        start_time = time.time()
        ut = pcloud.A_factorized.solve(pcloud.M @ ut)
        time_arr[t] = time.time() - start_time
        ut += param.source_strength * source
        ut[pcd_helper.is_boundary_arr] = 0

        # 5. Compute gradient of heat field
        scalar_diffusion_solver.get_gradient(ut)

        # 6. Project agent back to surface (keeps agent on manifold)
        (agent.x,) = get_gradient(
            np.copy(agent.x),
            neighbor_coords,
            neighbor_ids,
            ut,
        )

        # 7. Get gradient direction from heat field
        gradient = np.mean(
            scalar_diffusion_solver.gradient_ut_3d[neighbor_ids[:10]], axis=0
        )

        # 8. Move agent along gradient
        prev_x = np.copy(agent.x)
        agent.update(gradient)
        displacement = agent.x - prev_x
        speed_arr[t] = np.linalg.norm(displacement)

        coverage_arr[..., t] = coverage
        heat_arr[..., t] = np.copy(ut)
        goal_density_arr[..., t] = goal_density
        estimated_density_arr[..., t] = mean_tmp

        # ===================================================================
        # PERIODIC GP UPDATE: Every 50 timesteps, update density estimate
        # ===================================================================
        if t % 50 == 0 and t > 0:
            print(f"Time step: {t}/{param.timesteps}")

            # 1. Sample trajectory (every 5th point from 0 to t)
            sample_points = torch.tensor(
                agent.x_arr[:t:5, :], dtype=torch.float32, device=device
            )

            # 2. Get ground truth density at these points
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                gpr_original_density = likelihood_real(model_real(sample_points))
            density_sample = gpr_original_density.mean

            # 3. REPLACE online GP training data with full trajectory
            train_x = sample_points.clone()
            train_y = density_sample.clone()
            model.train()
            likelihood.train()
            model.set_train_data(
                train_x, train_y, strict=False
            )  # Replace, don't append!
            model.eval()
            likelihood.eval()

            # 4. Predict on full point cloud
            test_x = torch.tensor(pcloud.vertices, dtype=torch.float32)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = likelihood(model(test_x))

            var_tmp = normalize_mat(observed_pred.variance.cpu().numpy())
            mean_tmp = normalize_mat(observed_pred.mean.cpu().numpy())

            # 5. Compute new goal density (exploration + exploitation)
            goal_density = param.exploit_alpha * normalize_mat(
                np.maximum(mean_tmp - np.mean(mean_tmp), 0)
            ) + (1 - param.exploit_alpha) * normalize_mat(var_tmp)
            goal_density = normalize_mat(goal_density)

    return (
        agent.x_arr,
        heat_arr,
        coverage_arr,
        time_arr,
        goal_density_arr,
        estimated_density_arr,
    )


# ===================================================================
# MAIN SCRIPT: Setup and run exploration
# ===================================================================

# Select the object to explore
obj_name = "bun270_X"  # Stanford bunny with X projected as the target


# Parameters
class param:
    pass


param.exploit_alpha = 0.6  # Balance: 0=pure exploration, 1=pure exploitation
param.timesteps = 1500
param.alpha = 100  # Heat equation parameter
param.voxel_size = 0.002
param.agent_radius = 2.5 * param.voxel_size
param.max_velocity = 0.1 * param.voxel_size * 2
param.max_acceleration = 1.0 * param.max_velocity * 2
param.source_strength = 1
param.nb_max_neighbors = 500
param.nb_minimum_neighbors = 20
param.nb_boundary_neighbors = 40

# Load and process point cloud
filename = config.get_point_cloud_path(f"{obj_name}.ply")
pcloud = process_point_cloud(filename, param)
pcd_helper = Pointcloud(pcloud.vertices)
boundary_normals = pcd_helper.get_boundary_normals()


# Setup heat equation solver
scalar_diffusion_solver = PointcloudScalarDiffusion(pcloud=pcd_helper)
pcloud.C, pcloud.M = robust_laplacian.point_cloud_laplacian(
    pcloud.vertices, n_neighbors=param.nb_boundary_neighbors
)
A = csc_matrix(pcloud.M + pcloud.dt * pcloud.C)
pcloud.A_factorized = splu(A)

# ===================================================================
# GROUND TRUTH GP MODEL: Represents the "unknown" density to explore
# ===================================================================
# This GP model simulates the unknown target distribution that the agent
# is trying to learn through exploration. In a real tactile scenario, this
# would be replaced by actual sensor readings.

# GP hyperparameters
l = 0.010  # Length scale
sigma = 1.0  # Signal variance
n_eig = 500  # Number of eigenfunctions for manifold kernel

# Create RBF kernel adapted to the point cloud manifold
km = rbf_manifold_kernel(pcloud.vertices, l, sigma, n_eig)

# Train on full point cloud with target density (boundary = 1, interior = 0)
train_x = torch.tensor(pcloud.vertices, dtype=torch.float32)
train_y = torch.tensor(pcloud.u0, dtype=torch.float32)

likelihood_real = gpytorch.likelihoods.GaussianLikelihood()
model_real = GPROnPointCloud(train_x, train_y, likelihood_real, km, pcloud.vertices)

# Set to evaluation mode (no training needed for ground truth)
model_real.eval()
likelihood_real.eval()

# Get the smoothed ground truth prediction (used for visualization)
with torch.no_grad():
    observed_pred = likelihood_real(model_real(train_x))
mean = observed_pred.mean.cpu().numpy()

# ===================================================================
# AGENT INITIALIZATION
# ===================================================================
agent = FirstOrderAgent(
    x=np.zeros(3),
    dim_t=param.timesteps,
    max_velocity=param.max_velocity,
)

# Set agent starting position (vertex 1500)
agent.x = pcloud.vertices[1500]
agent.radius = param.agent_radius

# ===================================================================
# RUN HEDAC EXPLORATION
# ===================================================================
x_arr, heat_arr, coverage_arr, time_arr, goal_arr, estimated_density_arr = hedac(
    agent, param, pcloud
)


# ===================================================================
# VISUALIZATION AND OUTPUT
# ===================================================================

# Static plot: Final estimated density with trajectory
print("\nGenerating visualizations...")
plots = visualize_point_cloud(
    pcloud.vertices,
    colors=estimated_density_arr[..., -1],  # Final estimated density
    is_show_plot=False,
    point_size=5,
)
fig = visualize_trajectory(x_arr[:, :], plots, color="black")
fig.show("browser")

# Animation 1: Goal density evolution (what the agent is trying to explore)
print("Creating goal density animation...")
goal_html_path = config.get_animation_path(f"ergodic_goal_density_{obj_name}.html")
animate_trajectory_pcloud(
    x_arr=x_arr,
    vertices=pcloud.vertices,
    color_frames=goal_arr,
    timesteps=param.timesteps,
    save_path=str(goal_html_path),
)
print(f"Goal density animation saved to {goal_html_path}")

# Animation 2: Estimated density evolution (what the agent learned)
print("Creating estimated density animation...")
est_html_path = config.get_animation_path(f"ergodic_estimated_density_{obj_name}.html")
animate_trajectory_pcloud(
    x_arr=x_arr,
    vertices=pcloud.vertices,
    color_frames=estimated_density_arr,
    timesteps=param.timesteps,
    save_path=str(est_html_path),
)
print(f"Estimated density animation saved to {est_html_path}")

# Static plot: Distribution evolution over time
plot_distribution_evolution_column_auto(
    vertices=pcloud.vertices,
    original_density=mean,
    estimated_density_arr=estimated_density_arr,
    pdf_name=str(config.get_plot_path(f"distribution_evolution_{obj_name}.pdf")),
    agent_trajectory=x_arr[:, :],
)

print("\nExploration complete!")
