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

device = torch.device("cpu")
# print("Using device: ", device)
torch.set_default_device(device)
import robust_laplacian
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from scipy.stats import multivariate_normal

from gpr_on_point_cloud import *
from plotting_utils import *
from pointcloud import Pointcloud
from pointcloud_scalar_diffusion import PointcloudScalarDiffusion
from pointcloud_utils import *
from virtual_agents import FirstOrderAgent, SecondOrderAgent

print(o3d.__version__)


class param:
    pass  # c-style struct


param.timesteps = 1500  # total simulation timesteps

# tuning: [1,100] increasing alpha result in global exploration closer to SS
# decreasing alpha result in local exploration lower limited
param.alpha = 1000

param.method = "exact"

# voxel filter size for downsampling the point cloud
param.voxel_size = 0.02
# param.voxel_size = 0.005
# radius for the agent footprint that'd be used in coverage
param.agent_radius = 2.5 * param.voxel_size  # for the cup and the bunny
# param.agent_radius = 5 * param.voxel_size  # for the plate
# define speed and acceleration in terms of voxel size
param.max_velocity = 0.5 * param.voxel_size
param.max_acceleration = 2 * param.max_velocity

# tuning: doesn't have much effect on exploration so we keep it at 1
param.source_strength = 1

# max. num. of neighbors to consider for computing the neighbors in agent radius
param.nb_max_neighbors = 500
# num. of neighbors to consider for tangent space and gradient computation
param.nb_minimum_neighbors = 10
# num. of neighbors to consider for implicitly determining the boundary
# setting this lower in bunny resutls in right ear considered as a seperate body
# setting this higher in bunny results in the right ear being considered as part
# of the main body
param.nb_boundary_neighbors = 40

param.alpha_exploit = 0.2
param.fov_radius = 0.075
param.beta = 1.0
param.look_step = 50


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
    gp_arr = np.zeros_like(coverage_arr)

    # Initialize goal density
    sample_points = torch.tensor(agent.x, dtype=torch.float32).reshape(1, -1)
    # Make prediction
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        gpr_original_density = likelihood_real(model_real(sample_points))

    density_sample = gpr_original_density.mean.cpu()
    # Construct training data
    train_x = sample_points.clone()
    train_y = density_sample.clone()

    # Initialize the likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPROnPointCloud(train_x, train_y, likelihood, km, pcloud.vertices)

    # set to training mode and train
    model.train()
    likelihood.train()

    # Get into evaluation (predictive posterior) mode and predict
    model.eval()
    likelihood.eval()

    test_x = torch.tensor(pcloud.vertices, dtype=torch.float32)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))

    # goal_density = observed_pred.mean.cpu().numpy()

    # we normalize the goal because it should be a probability distribution
    goal_density = normalize_mat(pcloud.u_ht)
    ut = np.array(goal_density)

    # we keep this and add coverage at each timestep on top of it
    coverage = np.zeros_like(goal_density)

    # plot the initial goal density
    # plot = plot_point_cloud(test_x, point_colors=coverage)
    # fig = go.Figure(plot)
    # update_figure(fig)
    # fig.update_layout(
    #     scene_camera=camera
    # )

    # fig.show('browser')

    # for keeping the runtime of each timestep
    time_arr = np.zeros(param.timesteps)

    agent.t = 0  # reset the agent's time
    gp_val = goal_density
    indices = []
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
        scalar_diffusion_solver.get_gradient(ut)
        (
            agent.x,
            gradient,
            projected_neighbor_coords,
        ) = get_gradient(
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

        gp_arr[..., t] = gp_val

        if t % param.look_step == 0 and t > 0:
            print(f"Time step: {t}/{param.timesteps}")
            # Update the goal density
            # Extract the trajectory

            [k, idx, dists] = pcloud.pcd_tree.search_radius_vector_3d(
                agent.x_arr[t, :], param.fov_radius
            )
            idx = np.asarray(idx)
            indices.append(idx)
            # Step 1: Concatenate all arrays
            all_indices = np.concatenate(indices)

            # Step 2: Get unique values
            unique_indices = np.unique(all_indices)

            sample_points = torch.tensor(
                pcloud.vertices[unique_indices], dtype=torch.float32
            )

            # Make prediction
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                gpr_original_density = likelihood_real(model_real(sample_points))

            density_sample = gpr_original_density.mean.cpu()

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

            var_tmp = observed_pred.variance.cpu().numpy()
            mean_tmp = observed_pred.mean.cpu().numpy()

            # Set variance to zero along the borders of the point cloud
            # border_indices = get_border_indices(
            #     pcloud.vertices, param.nb_boundary_neighbors
            # )
            border_indices = pcd_helper.is_boundary_arr

            weighted_gp = normalize_mat(
                param.alpha_exploit * normalize_mat(mean_tmp)
                + (1 - param.alpha_exploit) * normalize_mat(var_tmp)
            )
            goal_density = normalize_mat(
                param.beta * normalize_mat(pcloud.u_ht) + (1 - param.beta) * weighted_gp
            )
            goal_density[border_indices] = 0
            gp_val = (normalize_mat(mean_tmp) + (normalize_mat(var_tmp))) / 2

    return agent.x_arr, heat_arr, coverage_arr, time_arr, goal_density_arr, gp_arr


point_cloud_dir = "point_clouds/"

# Select the object to explore

# obj_name = "bun270_X"  # Stanford bunny with X projected as the target
obj_name = "rectangular_grid_10k_RLI"  # random IKEA plate with hand-drawn shapes
# obj_name = "pointcloud_0"  # random IKEA plate with hand-drawn shapes
# obj_name = "cup_X"  # random cup that we scanned with X projected as the target
# obj_name = "plate_shapes"  # random cup that we scanned with X projected as the target

# Select the object and load the point cloud
# ==========================================
filename = f"{point_cloud_dir}{obj_name}.ply"
pcloud = process_point_cloud(filename, param)
pcd_helper = Pointcloud(pcloud.vertices)
boundary_normals = pcd_helper.get_boundary_normals()

u0 = np.zeros(len(pcloud.vertices))
u0[pcd_helper.is_boundary_arr] = 1

scalar_diffusion_solver = PointcloudScalarDiffusion(pcloud=pcd_helper)

# fig = visualize_point_cloud(
#     pcloud.vertices,
#     colors=u0,
#     is_show_plot=True,
# )
# fig.show()

# fig = visualize_gradient_field(
#     pcloud.vertices[pcd_helper.is_boundary_arr],
#     gradient_arr=boundary_normals,
#     sizeref=10,
#     is_show_plot=True,
# )
# fig.show()


pcloud.C, pcloud.M = robust_laplacian.point_cloud_laplacian(
    pcloud.vertices, n_neighbors=param.nb_boundary_neighbors
)

A = csc_matrix(pcloud.M + pcloud.dt * pcloud.C)  # Ensure sparse format
pcloud.A_factorized = splu(A)  # LU factorization

import gpytorch

# Define the goal density
# ========================
import torch


def get_border_indices(vertices, nb_boundary_neighbors):
    """
    Identify the border indices of the point cloud.

    Args:
        vertices (np.ndarray): The vertices of the point cloud.
        nb_boundary_neighbors (int): The number of neighbors to consider for boundary detection.

    Returns:
        np.ndarray: The indices of the border vertices.
    """
    from sklearn.neighbors import NearestNeighbors

    # Find the nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=nb_boundary_neighbors).fit(vertices)
    distances, indices = nbrs.kneighbors(vertices)

    # Calculate the mean distance to the neighbors
    mean_distances = distances.mean(axis=1)

    # Identify the border vertices as those with the highest mean distance to neighbors
    threshold = np.percentile(mean_distances, 85)  # Adjust this threshold as needed
    border_indices = np.where(mean_distances > threshold)[0]

    return border_indices


# Load gp on pc class
# ====================
l = 0.002
sigma = 1.0
n_eig = 200
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

# Get into evaluation (predictive posterior) mode and predict
model_real.eval()
likelihood_real.eval()
import time

start = time.time()
with torch.no_grad():
    observed_pred = likelihood_real(model_real(train_x))
end = time.time()
print(f"Time to predict: {end - start}")
mean = observed_pred.mean.cpu().numpy()
var = observed_pred.variance.cpu().numpy()


# agent = SecondOrderAgent(
#     x=np.zeros(3), dim_t=param.timesteps, max_velocity=param.max_velocity,max_acceleration=param.max_acceleration
# )

agent = FirstOrderAgent(
    x=np.zeros(3), dim_t=param.timesteps, max_velocity=param.max_velocity
)

random_vertex = np.random.randint(0, len(pcloud.vertices))
agent.x = pcloud.vertices[100]
agent.radius = param.agent_radius

# Gaussian centers
centers = np.array([[0.17, 0.78], [0.62, 0.62], [0.439, 0.237]])

# Example: Use the same covariance matrix for all
cov = np.array([[0.01, 0], [0, 0.01]])  # isotropic, adjust for spread

from scipy.stats import multivariate_normal

points = pcloud.vertices[:, :2]
# Evaluate Gaussians
values = np.zeros(len(points))
for mu in centers:
    rv = multivariate_normal(mean=mu, cov=cov)
    values += rv.pdf(points)  # sum the densities

pcloud.u_ht = values  # Gaussian target
pcloud.u_ht = np.ones(len(pcloud.vertices))

# camera = dict(
#     up=dict(x=0, y=1, z=0),
#     center=dict(x=0, y=0, z=0),
#     eye=dict(x=0, y=0, z=1.2)
# )
# plot = plot_point_cloud(train_x.cpu().numpy(), point_colors=mean)
# # plot = plot_point_cloud(train_x.cpu().numpy(), point_colors=pcloud.u_ht)
# fig = go.Figure(plot)
# update_figure(fig)
# fig.update_layout(
#     scene_camera=camera
# )

# fig.show('browser')

x_arr, heat_arr, coverage_arr, time_arr, goal_arr, gp_arr = hedac(agent, param, pcloud)

u_ht_arr = np.tile(pcloud.u_ht[:, np.newaxis], (1, heat_arr.shape[-1]))

animate_trajectory_pcloud(
    x_arr,
    vertices=pcloud.vertices,
    color_frames=gp_arr,
    timesteps=param.timesteps,
    circle_radius=param.fov_radius,
    look_step=param.look_step,
    save_path="pl_3dk_gp.html",
)
animate_trajectory_pcloud(
    x_arr,
    vertices=pcloud.vertices,
    color_frames=u_ht_arr,
    timesteps=param.timesteps,
    circle_radius=param.fov_radius,
    look_step=param.look_step,
    save_path="pl_3dk_target_distribution.html",
)
