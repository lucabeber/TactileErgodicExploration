import numpy as np

np.set_printoptions(formatter={"float": lambda x: "{0:0.3e}".format(x)})

import time



import robust_laplacian
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from plotting_utils import *
from pointcloud_utils import *
from virtual_agents import SecondOrderAgent

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


# Define the goal density
# ========================
import torch
import gpytorch
import matplotlib.pyplot as plt
import open3d as o3d
import os
import matplotlib.cm as cm

from virtual_agents import SecondOrderAgent

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
    threshold = np.percentile(mean_distances, 95)
    border_indices = np.where(mean_distances > threshold)[0]

    # # Show the border vertices in the point cloud
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(vertices)
    # pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(vertices))
    # colors = np.zeros_like(vertices)
    # colors[border_indices] = [1, 0, 0]  # Red color for the border vertices
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd], window_name="Border vertices")


    return border_indices

# Construct training data
train_x = torch.tensor(pcloud.vertices, dtype=torch.float64)
train_y = torch.tensor(pcloud.u0, dtype=torch.float64)



# Define the GP model without derivatives
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood_real):
        super(GPModel, self).__init__(train_x, train_y, likelihood_real)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=3))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood_real = gpytorch.likelihoods.GaussianLikelihood()
model_real = GPModel(train_x, train_y, likelihood_real)
model_real = model_real.double()
likelihood_real.double()

# Training the model
model_real.train()
likelihood_real.train()

model_state_path = obj_name + "_model_state.pth"
likelihood_state_path = obj_name + "_likelihood_state.pth"

if os.path.exists(model_state_path) and os.path.exists(likelihood_state_path):
    # Load parameters from the saved model
    model_real.load_state_dict(torch.load(model_state_path))
    likelihood_real.load_state_dict(torch.load(likelihood_state_path))
    # Convert to double precision after loading
    model_real = model_real.double()
    likelihood_real.double()
else:
    optimizer = torch.optim.Adam(model_real.parameters(), lr=0.2)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_real, model_real)

    training_iterations = 20
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model_real(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        print(f"Iter {i+1}/{training_iterations} - Loss: {loss.item()}")

    # Save parameters
    torch.save(model_real.state_dict(), model_state_path)
    torch.save(likelihood_real.state_dict(), likelihood_state_path)

# Switch to evaluation mode
model_real.eval()
likelihood_real.eval()

# Plot the predicted density
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.tensor(pcloud.vertices, dtype=torch.float64)
    observed_pred = likelihood_real(model_real(test_x))

# # Map stiffness to RGB colors using a colormap
# colormap = cm.get_cmap('jet')  # Change to 'jet' or other colormaps if needed
# tmp = observed_pred.mean.cpu().numpy()
# colors = colormap(tmp)[:, :3]  # Convert to RGB
# # 
# # Create Open3D point cloud object
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(pcloud.vertices)
# pcd.colors = o3d.utility.Vector3dVector(colors)

# # Visualise with Open3D
# o3d.visualization.draw_geometries([pcd], window_name="Target density")

agent = SecondOrderAgent(
    x=np.zeros(3), dim_t=param.timesteps, max_velocity=param.max_velocity,max_acceleration=param.max_acceleration*2
)

random_vertex = np.random.randint(0,len(pcloud.vertices))
agent.x = pcloud.vertices[random_vertex]
agent.radius = param.agent_radius

# Initialize BO optimization loop
# =================================
import botorch 
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

import time

# Constants
MM_TO_UNIT = 1.0  # Adjust based on your point cloud scale
dtype = torch.float64
device = torch.device("cpu")
torch.set_default_dtype(dtype)
torch.set_default_device(device)

# Precompute geodesic distances on point cloud using graph-based approach
def precompute_geodesic_distances(vertices):
    """Precompute geodesic distances between all vertices using shortest path on k-NN graph."""
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    
    n_vertices = len(vertices)
    # Build k-NN graph
    nbrs = NearestNeighbors(n_neighbors=10).fit(vertices)
    distances, indices = nbrs.kneighbors(vertices)
    
    # Create sparse distance matrix
    row, col, data = [], [], []
    for i in range(n_vertices):
        for j, dist in zip(indices[i], distances[i]):
            row.append(i)
            col.append(j)
            data.append(dist)
    
    graph = csr_matrix((data, (row, col)), shape=(n_vertices, n_vertices))
    # Compute shortest paths
    geodesic_distances = shortest_path(graph, directed=False)
    return torch.tensor(geodesic_distances, dtype=dtype, device=device)

geodesic_distances = precompute_geodesic_distances(pcloud.vertices)

def get_geodesic_distance(idx_a, idx_b):
    """Get precomputed geodesic distance between two vertices."""
    return geodesic_distances[idx_a, idx_b].item()

# --- GP Model ---
def get_fitted_model(X, Y):
    # Ensure inputs are float64
    X = X.to(dtype=torch.float64)
    Y = Y.to(dtype=torch.float64)
    # Ensure Y is 2D: (n_samples, 1)
    if Y.dim() == 1:
        Y = Y.unsqueeze(-1)
    
    # Standardize Y to zero mean and unit variance to avoid warnings and improve training
    Y_mean = Y.mean()
    Y_std = Y.std()
    
    # Handle case where Y has only 1 sample or very small variance
    if Y_std < 1e-8 or torch.isnan(Y_std):
        Y_std = torch.tensor(1.0, dtype=torch.float64)
    
    Y_normalized = (Y - Y_mean) / Y_std
    
    model_gp = SingleTaskGP(X, Y_normalized, outcome_transform=None)
    model_gp = model_gp.double()
    mll = ExactMarginalLogLikelihood(model_gp.likelihood, model_gp)
    fit_gpytorch_mll(mll)
    return model_gp, Y_mean, Y_std

# --- BO Settings ---
bounds = torch.tensor(
    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=dtype, device=device
)

def f_torch(X):
    """Evaluate predicted density at 3D points using the trained GP model."""
    # Ensure X is float64 to match model dtype
    X = X.to(dtype=torch.float64)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood_real(model_real(X))
    return preds.mean.unsqueeze(-1).to(dtype=dtype)

def generate_initial_data(n=1):
    # Random points from the point cloud
    indices = np.random.choice(len(pcloud.vertices), n, replace=False)
    X_init = torch.tensor(pcloud.vertices[indices], dtype=torch.float64, device=device)
    Y_init = f_torch(X_init)
    return X_init, Y_init, indices

def bayesian_optimisation(n_iter=100):
    X_train, Y_train, train_indices = generate_initial_data()
    X_positions = X_train.clone()
    visited_indices = set(train_indices)
    time_simulation = 0.0
    i = 0
    mean_iteration_time = 0.0
    
    while time_simulation < 200:
        start_time = time.time()
        model, y_mean, y_std = get_fitted_model(X_train, Y_train)
        acq_func = LogExpectedImprovement(
            model=model,
            best_f=y_mean + 0.0 * y_std,  # since we normalized Y
        )

        last_idx = train_indices[-1]
        
        # Find nearby unvisited vertices on point cloud
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=100).fit(pcloud.vertices)
        _, local_indices = nbrs.kneighbors([pcloud.vertices[last_idx]])
        local_indices = local_indices.flatten()
        
        # Filter out already visited points
        unvisited_local = [idx for idx in local_indices if idx not in visited_indices]
        if not unvisited_local:
            print("All nearby points visited, exploring all unvisited points")
            unvisited_local = [idx for idx in range(len(pcloud.vertices)) if idx not in visited_indices]
        
        local_vertices = torch.tensor(
            pcloud.vertices[unvisited_local], dtype=torch.float64, device=device
        )

        # Evaluate acquisition function on local vertices
        with torch.no_grad():
            # Reshape for acquisition function evaluation: (n_points, 1, 3)
            acq_values = acq_func(local_vertices.unsqueeze(1).double())
        
        best_local_idx = torch.argmax(acq_values).item()
        new_idx = unvisited_local[best_local_idx]
        new_x = torch.tensor(
            pcloud.vertices[new_idx], dtype=torch.float64, device=device
        ).unsqueeze(0)

        # Compute geodesic distance
        geodesic_dist = get_geodesic_distance(last_idx, new_idx)
        
        # Sample points along geodesic path (on point cloud)
        step_size = 2.5 * MM_TO_UNIT
        num_steps = max(1, int(np.ceil(geodesic_dist / step_size)))
        
        # Linear interpolation in 3D (approximation of geodesic)
        last_x = torch.tensor(
            pcloud.vertices[last_idx], dtype=torch.float64, device=device
        ).unsqueeze(0)
        alpha = torch.linspace(0, 1, num_steps + 1, dtype=torch.float64, device=device)[1:]
        path_x = last_x + alpha.unsqueeze(1) * (new_x - last_x)
        4
        new_y = f_torch(path_x)
        
        print(f"Previous point index: {last_idx}, coords: {last_x.cpu().numpy().squeeze()}")
        print(f"New point index: {new_idx}, coords: {new_x.cpu().numpy().squeeze()}")
        print(f"Geodesic distance: {geodesic_dist:.4f}")
        
        X_train = torch.cat([X_train, path_x], dim=0)
        Y_train = torch.cat([Y_train, new_y], dim=0)
        train_indices = np.append(train_indices, new_idx)
        visited_indices.add(new_idx)
        X_positions = torch.cat([X_positions, new_x], dim=0)

        end_time = time.time()
        print(f"Iteration {i+1} took {end_time - start_time:.2f} seconds")
        time_simulation += end_time - start_time + geodesic_dist / 10 * 0.9
        mean_iteration_time += end_time - start_time
        
        i += 1
    
    print(f"Mean iteration time: {mean_iteration_time / i:.3f} seconds")
    return X_train, Y_train, X_positions, visited_indices

X_train, Y_train, X_positions, visited_indices = bayesian_optimisation()
# Train a new GP model on the collected data
model_gpr, y_mean_gpr, y_std_gpr = get_fitted_model(X_train, Y_train)

# Evaluate the trained GPR model on all vertices for visualization
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    all_pred_gpr = model_gpr(torch.tensor(pcloud.vertices, dtype=torch.float64))
    pred_mean_gpr = all_pred_gpr.mean.cpu().numpy()
    pred_normalized = (pred_mean_gpr - pred_mean_gpr.min()) / (pred_mean_gpr.max() - pred_mean_gpr.min() + 1e-6)
# Create visualization with explored points and paths
colormap = cm.get_cmap('RdYlGn')  # Red-Yellow-Green colormap
colors = np.zeros((len(pcloud.vertices), 3))

# Color all points by estimated density
for i, density in enumerate(pred_normalized):
    colors[i] = colormap(density)[:3]

# Create Open3D point cloud with estimated distribution
pcd_all = o3d.geometry.PointCloud()
pcd_all.points = o3d.utility.Vector3dVector(pcloud.vertices)
pcd_all.colors = o3d.utility.Vector3dVector(colors)

# Create small spheres for visited points
visited_spheres = []
for idx in visited_indices:
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=param.agent_radius * 0.1)
    mesh_sphere.translate(pcloud.vertices[idx])
    mesh_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red for visited points
    visited_spheres.append(mesh_sphere)

# Create line segments for the exploration path
line_set = o3d.geometry.LineSet()
path_points = X_positions.cpu().numpy()
line_set.points = o3d.utility.Vector3dVector(path_points)

# Create line segments between consecutive exploration points
lines = []
for i in range(len(path_points) - 1):
    lines.append([i, i + 1])
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.paint_uniform_color([0.0, 0.0, 1.0])  # Blue for path

print(f"Total points in point cloud: {len(pcloud.vertices)}")
print(f"Total visited points: {len(visited_indices)}")
print(f"Coverage: {100 * len(visited_indices) / len(pcloud.vertices):.2f}%")
print(f"Total exploration steps: {len(X_positions)}")

# Visualize
print("\nVisualization legend:")
print("- Point cloud: colored by ESTIMATED density (red=low, green=high)")
print("- Red spheres: visited exploration points")
print("- Blue lines: exploration path connecting consecutive points")
print("\nClose the visualization window to continue.")

o3d.visualization.draw_geometries(
    [pcd_all, line_set] + visited_spheres,
    window_name="BO Exploration Results",
    width=1200,
    height=800
)
