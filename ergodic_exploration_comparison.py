

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

from gpr_on_point_cloud import *
from plotting_utils import *
from pointcloud import Pointcloud
from pointcloud_scalar_diffusion import PointcloudScalarDiffusion
from pointcloud_utils import *
from virtual_agents import FirstOrderAgent, SecondOrderAgent


def hedac(agent, param, pcloud, update_interval=50):
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

    # Initialize goal density
    sample_points = torch.tensor(agent.x, dtype=torch.float32).reshape(1, -1)
    # Make prediction
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        gpr_original_density = likelihood_real(model_real(sample_points))

    density_sample = gpr_original_density.mean
    # print(stiffness_sample)
    # Construct training data
    train_x = sample_points.clone()
    train_y = density_sample.clone()
    print(train_x)
    print(train_y)

    # Initialize the likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPROnPointCloud(train_x, train_y, likelihood, km, pcloud.vertices)

    # set to training mode and train
    model.train()
    likelihood.train()

    # Get into evaluation (predictive posterior) mode and predict
    model.eval()
    likelihood.eval()

    test_x = torch.tensor(pcloud.vertices, dtype=torch.float32, device=device)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))

    goal_density = observed_pred.mean.cpu().numpy()

    # we normalize the goal because it should be a probability distribution
    goal_density = normalize_mat(goal_density)
    mean_tmp = normalize_mat(observed_pred.mean.cpu().numpy())
    mean_not_normalized = observed_pred.mean.cpu().numpy()
    ut = np.array(goal_density)

    # we keep this and add coverage at each timestep on top of it
    coverage = np.zeros_like(goal_density)

    # fig.show("browser")
    # for keeping the runtime of each timestep
    time_arr = np.zeros(param.timesteps)

    agent.t = 0  # reset the agent's time
    # do absolute minimum inside the main loop
    for t in range(param.timesteps):
        dists, neighbor_ids, neighbor_coords = get_pcloud_neighbors(
            pcloud.pcd_tree,
            pcloud.vertices,
            np.copy(agent.x),
            agent.radius,
            param.nb_max_neighbors,
            param.nb_minimum_neighbors,
        )

        # Compute the coverage using RBF kernel
        kernel_vals = np.exp(-(1 / agent.radius) * dists**2)
        coverage[neighbor_ids] += kernel_vals

        neighbor_ids = neighbor_ids[: param.nb_minimum_neighbors]
        dists = dists[: param.nb_minimum_neighbors]
        neighbor_coords = pcloud.vertices[neighbor_ids, :]
        coverage_density = normalize_mat(coverage)
        source = np.maximum(goal_density - coverage_density, 0) ** 2
        source = normalize_mat(source)

        start_time = time.time()

        ut = pcloud.A_factorized.solve(pcloud.M @ ut)
        time_arr[t] = time.time() - start_time

        ut += param.source_strength * source
        ut[pcd_helper.is_boundary_arr] = 0
        scalar_diffusion_solver.get_gradient(ut)
        (agent.x,) = get_gradient(
            np.copy(agent.x),
            neighbor_coords,
            neighbor_ids,
            ut,
        )
        # Interpolate the gradient at the agent location using its closest 5 neighbors' values
        gradient = np.mean(
            scalar_diffusion_solver.gradient_ut_3d[neighbor_ids[:10]], axis=0
        )

        agent.update(gradient)

        coverage_arr[..., t] = coverage
        heat_arr[..., t] = np.copy(ut)
        goal_density_arr[..., t] = goal_density
        estimated_density_arr[..., t] = mean_not_normalized

        if t % update_interval == 0 and t > 0:
            # print(f"Time step: {t}/{param.timesteps}")
            # Update the goal density
            # Extract the trajectory
            sample_points = torch.tensor(
                agent.x_arr[:t:5, :], dtype=torch.float32, device=device
            )
            # print(sample_points.shape)
            # Make prediction
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                gpr_original_density = likelihood_real(model_real(sample_points))

            density_sample = gpr_original_density.mean

            # Construct training data
            train_x = sample_points.clone()
            train_y = density_sample.clone()

            # Training the model
            model.train()
            likelihood.train()

            model.set_train_data(train_x, train_y, strict=False)

            # Switch to evaluation mode
            model.eval()
            likelihood.eval()

            test_x = torch.tensor(pcloud.vertices, dtype=torch.float32)

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = likelihood(model(test_x))

            # # Map stiffness to RGB colors using a colormap
            # colormap = cm.get_cmap('jet')  # Change to 'jet' or other colormaps if needed
            # colors = colormap(observed_pred.mean.cpu().numpy())[:, :3]  # Convert to RGB

            # # Create Open3D point cloud object
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(pcloud.vertices)
            # pcd.colors = o3d.utility.Vector3dVector(colors)

            # # Visualise with Open3D
            # o3d.visualization.draw_geometries([pcd], window_name="Target density")

            var_tmp = normalize_mat(observed_pred.variance.cpu().numpy())
            mean_tmp = normalize_mat(observed_pred.mean.cpu().numpy())
            mean_not_normalized = observed_pred.mean.cpu().numpy()
            # Set variance to zero along the borders of the point cloud


            goal_density = (
                    param.exploit_alpha * normalize_mat(np.maximum(mean_tmp - np.mean(mean_tmp), 0))
                    + (1 - param.exploit_alpha) * normalize_mat(var_tmp)
            )
            goal_density = normalize_mat(goal_density)
            # goal_density[border_indices] = 0
            # plots = visualize_point_cloud(
            #     pcloud.vertices,
            #     colors=goal_density,
            #     # colors=heat_arr[...,-1],
            #     is_show_plot=False, point_size=5
            # )
            # fig = visualize_trajectory(agent.x_arr[:t,:], plots, color="black")
            # fig.show()

            # plots = visualize_point_cloud(
            #     pcloud.vertices,
            #     colors=heat_arr[...,t],
            #     # colors=heat_arr[...,-1],
            #     is_show_plot=False, point_size=5
            # )
            # fig = visualize_trajectory(agent.x_arr[:t,:], plots, color="black")

            # fig.show()
    return agent.x_arr, heat_arr, coverage_arr, time_arr, goal_density_arr, estimated_density_arr


point_cloud_dir = "point_clouds/"

# Select the object to explore

# obj_name = "bun270_X" # Stanford bunny with X projected as the target
obj_name = (
    "plate_shapes"  # random IKEA plate with hand-drawn shapes
)
# obj_name = "cup_X" # random cup that we scanned with X projected as the target

experiment_index = 2  # choose which initial position to use from x0_arr_10.npz


class param:
    pass  # c-style struct


param.exploit_alpha = 0.6  # total simulation timesteps

param.timesteps = 6000  # total simulation timesteps

# tuning: [1,100] increasing alpha result in global exploration closer to SS
# decreasing alpha result in local exploration lower limited
param.alpha = 100

param.method = "exact"

# voxel filter size for downsampling the point cloud
param.voxel_size = 0.002
# radius for the agent footprint that'd be used in coverage
param.agent_radius = 2.5 * param.voxel_size # for the cup and the bunny
# param.agent_radius = 5 * param.voxel_size  # for the plate
# define speed and acceleration in terms of voxel size
param.max_velocity = 0.1 * param.voxel_size * 2
param.max_acceleration = 1.0 * param.max_velocity * 2

# tuning: doesn't have much effect on exploration so we keep it at 1
param.source_strength = 1

# max. num. of neighbors to consider for computing the neighbors in agent radius
param.nb_max_neighbors = 500
# num. of neighbors to consider for tangent space and gradient computation
param.nb_minimum_neighbors = 20
# num. of neighbors to consider for implicitly determining the boundary
# setting this lower in bunny resutls in right ear considered as a seperate body
# setting this higher in bunny results in the right ear being considered as part
# of the main body
param.nb_boundary_neighbors = 40


# Select the object and load the point cloud
# ==========================================
filename = f"{point_cloud_dir}{obj_name}.ply"
pcloud = process_point_cloud(filename, param)
pcd_helper = Pointcloud(pcloud.vertices)
boundary_normals = pcd_helper.get_boundary_normals()

u0 = np.zeros(len(pcloud.vertices))
u0[pcd_helper.is_boundary_arr] = 1

scalar_diffusion_solver = PointcloudScalarDiffusion(pcloud=pcd_helper)

pcloud.C, pcloud.M = robust_laplacian.point_cloud_laplacian(
    pcloud.vertices, n_neighbors=param.nb_boundary_neighbors
)

A = csc_matrix(pcloud.M + pcloud.dt * pcloud.C)  # Ensure sparse format
pcloud.A_factorized = splu(A)  # LU factorization


import os

import gpytorch
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import open3d as o3d

# Define the goal density
# ========================
import torch


# Load gp on pc class
# ====================
l = 0.010
sigma = 1.0
n_eig = 500
km = rbf_manifold_kernel(pcloud.vertices, l, sigma, n_eig)

# Construct training data
train_x = torch.tensor(pcloud.vertices, dtype=torch.float32)
train_y = torch.tensor(pcloud.u0, dtype=torch.float32)

# Initialize the likelihood and model
likelihood_real = gpytorch.likelihoods.GaussianLikelihood()
model_real = GPROnPointCloud(train_x, train_y, likelihood_real, km, pcloud.vertices)

# set to training mode and train
model_real.train()
likelihood_real.train()


model_real.eval()
likelihood_real.eval()

import time

start = time.time()
with torch.no_grad():
    observed_pred = likelihood_real(model_real(train_x))
end = time.time()
# print(f"Time to predict: {end - start}")
global mean_original
mean_original = observed_pred.mean.cpu().numpy()
# var = observed_pred.variance.cpu().numpy()


camera = dict(
    up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=0, y=0.7, z=1.25)
)

plot = plot_point_cloud(train_x.cpu().numpy(), point_colors=mean_original)
fig = go.Figure(plot)
update_figure(fig)
fig.update_layout(scene_camera=camera)

fig.show("browser")

agent = SecondOrderAgent(
    x=np.zeros(3),
    max_velocity=param.max_velocity,
    max_acceleration=param.max_acceleration * 2,
    dim_t=param.timesteps,
)

# agent = FirstOrderAgent(
#     x=np.zeros(3),
#     dim_t=param.timesteps,
#     max_velocity=param.max_velocity,
# )

random_vertex = np.random.randint(0, len(pcloud.vertices))
# agent.x = pcloud.vertices[810]
agent.x = pcloud.vertices[1500]
agent.radius = param.agent_radius

# plots = visualize_gradient_field(
#     pcloud.vertices[pcd_helper.is_boundary_arr],
#     boundary_normals,
#     sizeref=10,
# )
# fig = go.Figure(plots)
# fig.show("browser")

# Run HEDAC with different update intervals
update_intervals = [5, 10, 20, 50, 100, 500, 1000, 2000]
results = {}

results = {ui: {'ergodic_metrics': [], 'rmses': []} for ui in update_intervals}

for update_interval in update_intervals:
    print(f"\n{'='*50}")
    print(f"Running HEDAC with update_interval={update_interval}")
    print(f"{'='*50}")

    for sim in range(50):  # Run 50 simulations for each update interval
        # Reset agent for each run
        agent = SecondOrderAgent(
            x=np.zeros(3),
            max_velocity=param.max_velocity,
            max_acceleration=param.max_acceleration * 2,
            dim_t=param.timesteps,
        )
        random_vertex = np.random.randint(0, len(pcloud.vertices))
        agent.x = pcloud.vertices[random_vertex]
        agent.radius = param.agent_radius

        x_arr, heat_arr, coverage_arr, time_arr, goal_arr, estimated_density_arr = hedac(
            agent, param, pcloud, update_interval=update_interval
        )

        # Compute ergodic metric (coverage-goal density difference)
        final_coverage = coverage_arr[..., -1]
        final_coverage_normalized = normalize_mat(final_coverage)
        ergodic_metric = np.linalg.norm(final_coverage_normalized - normalize_mat(goal_arr[..., -1]))

        # Compute RMSE between estimated and goal distribution
        rmse = np.sqrt(np.mean((estimated_density_arr[..., -1] - mean_original)**2))

        results[update_interval]['ergodic_metrics'].append(ergodic_metric)
        results[update_interval]['rmses'].append(rmse)


# Save the results in a numpy file
np.savez_compressed("hedac_comparison_results.npz", results=results)

# Create box plots comparing results across different update intervals
fig, axes = plt.subplots(1, 2, figsize=(14, 10))
fig.suptitle('HEDAC Performance Comparison Across Update Intervals', fontsize=16)

# Plot 1: Ergodic Metric Comparison
ax = axes[0]
ax.boxplot([results[ui]['ergodic_metrics'] for ui in update_intervals], tick_labels=update_intervals)
ax.set_xlabel('Update Interval')
ax.set_ylabel('Ergodic Metric (L2 norm)')
ax.set_title('Ergodic Metric vs Update Interval')
ax.grid(axis='y', alpha=0.3)

# Plot 2: RMSE Comparison
ax = axes[1]
ax.boxplot([results[ui]['rmses'] for ui in update_intervals], tick_labels=update_intervals)
ax.set_xlabel('Update Interval')
ax.set_ylabel('RMSE')
ax.set_title('RMSE vs Update Interval')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_results_boxplots.png', dpi=150, bbox_inches='tight')
plt.show()

# Print summary statistics
print(f"\n{'='*50}")
print("SUMMARY OF RESULTS")
print(f"{'='*50}")
for update_interval in update_intervals:
    ergodic_metrics = results[update_interval]['ergodic_metrics']
    rmses = results[update_interval]['rmses']
    print(f"Update Interval: {update_interval:4d}")
    print(f"  Ergodic Metric - Mean: {np.mean(ergodic_metrics):.6f}, Std: {np.std(ergodic_metrics):.6f}")
    print(f"  RMSE           - Mean: {np.mean(rmses):.6f}, Std: {np.std(rmses):.6f}")

plots = visualize_point_cloud(
    pcloud.vertices,
    colors=estimated_density_arr[..., -1],
    # colors=heat_arr[...,-1],
    is_show_plot=False,
    point_size=5,
)
fig = visualize_trajectory(x_arr[:, :], plots, color="black")

fig.show("browser")

# import plotly.io as pio

# animate_trajectory_pcloud(
#     x_arr,
#     vertices=pcloud.vertices,
#     color_frames=goal_arr,
#     timesteps=param.timesteps,
#     save_path="pl_3dk_target_distribution.html",
# )

# animate_trajectory_pcloud(
#     x_arr,
#     vertices=pcloud.vertices,
#     color_frames=heat_arr,
#     timesteps=param.timesteps,
#     save_path="pl_3dk_goal_density.html",
# )

# --- Example usage ---
# Choose 5 steps to visualise
steps_to_plot = [0, 100, 500, 1000, 2000]

plot_distribution_evolution_column_auto(
    vertices=pcloud.vertices,
    original_density=mean_original,
    estimated_density_arr=estimated_density_arr,
    pdf_name="distribution_evolution6_" + obj_name + ".pdf",
    agent_trajectory=x_arr[:, :],
)