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



import robust_laplacian
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from plotting_utils import *
from pointcloud_utils import *
from virtual_agents import SecondOrderAgent


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

    # we normalize the goal because it should be a probability distribution
    goal_density = normalize_mat(pcloud.u0) ## change here

    # we keep this and add coverage at each timestep on top of it
    coverage = np.zeros_like(goal_density)
    ut = np.array(goal_density)

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
        (
            agent.x,
            gradient,
            _,
        ) = get_gradient(
            np.copy(agent.x),
            neighbor_coords,
            neighbor_ids,
            ut,
        )
        agent.update(gradient)

        coverage_arr[..., t] = coverage
        heat_arr[..., t] = np.copy(ut)
    return agent.x_arr, heat_arr, coverage_arr, time_arr

point_cloud_dir = "point_clouds/"

# Select the object to explore

# obj_name = "bun270_X" # Stanford bunny with X projected as the target
obj_name = "plate_shapes"  # random IKEA plate with hand-drawn shapes
# obj_name = "cup_X" # random cup that we scanned with X projected as the target

experiment_index = 2  # choose which initial position to use from x0_arr_10.npz

class param:
    pass  # c-style struct


param.timesteps = 500  # total simulation timesteps

# tuning: [1,100] increasing alpha result in global exploration closer to SS
# decreasing alpha result in local exploration lower limited
param.alpha = 100

param.method = "exact"

# voxel filter size for downsampling the point cloud
param.voxel_size = 0.003
# radius for the agent footprint that'd be used in coverage
param.agent_radius = 2.5 * param.voxel_size # for the cup and the bunny
# param.agent_radius = 5 * param.voxel_size  # for the plate
# define speed and acceleration in terms of voxel size
param.max_velocity = 1 * param.voxel_size
param.max_acceleration = 1 * param.max_velocity

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


pcloud.C, pcloud.M = robust_laplacian.point_cloud_laplacian(
    pcloud.vertices, n_neighbors=param.nb_boundary_neighbors
)

A = csc_matrix(pcloud.M + pcloud.dt * pcloud.C)  # Ensure sparse format
pcloud.A_factorized = splu(A)  # LU factorization




agent = SecondOrderAgent(
    x=np.zeros(3), dim_t=param.timesteps, max_velocity=param.max_velocity,max_acceleration=param.max_acceleration*2
)

random_vertex = np.random.randint(0,len(pcloud.vertices))
agent.x = pcloud.vertices[random_vertex]
agent.radius = param.agent_radius


x_arr, heat_arr, coverage_arr, time_arr = hedac(agent, param, pcloud)


plots = visualize_point_cloud(
    pcloud.vertices, 
    colors=heat_arr[...,0], 
    # colors=heat_arr[...,-1], 
    is_show_plot=False, point_size=5
)
fig = visualize_trajectory(x_arr[:,:], plots, color="black")

fig.show()