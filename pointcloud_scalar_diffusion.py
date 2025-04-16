"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of manifold_diffusion.

manifold_diffusion is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 3 as
published by the Free Software Foundation.

manifold_diffusion is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with manifold_diffusion. If not, see <http://www.gnu.org/licenses/>.
"""

import random
import time

import numpy as np
import potpourri3d as pp3d
import robust_laplacian
import scipy.sparse.linalg as sla

# from manifold_diffusion.quaternion_utils import *
from pcdiff import build_grad_div, estimate_basis, knn_graph
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from scipy.spatial.transform import Rotation as R

# from manifold_diffusion.pointcloud import Pointcloud


class DiffusionSolver:
    def __init__(
        self,
        diffusion_scalar=1,
        method="heat",
        num_eigen=None,
        num_integration_steps=1,
    ):
        self.diffusion_scalar = diffusion_scalar
        valid_methods = {"invert", "LU", "eigen", "laplace", "LU_laplace"}

        if method not in valid_methods:
            raise ValueError(
                f"Unknown method '{method}'. Valid methods are: {valid_methods}"
            )
        self.method = method
        if method == "eigen":
            assert num_eigen is not None, "Please provide the number of eigenvalues"
            self.num_eigen = num_eigen
        self.num_integration_steps = num_integration_steps


"""
Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

This file is part of manifold_diffusion.

manifold_diffusion is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 3 as
published by the Free Software Foundation.

manifold_diffusion is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with manifold_diffusion. If not, see <http://www.gnu.org/licenses/>.
"""

import random
import time

import numpy as np
import polyscope as ps
import potpourri3d as pp3d
import robust_laplacian
import scipy.sparse.linalg as sla
import yaml

# from manifold_diffusion.plotting_ps import *
# from manifold_diffusion.pointcloud_diffusion import DiffusionSolver
from pcdiff import build_grad_div, estimate_basis, knn_graph
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


def apply_A_inv(A_factorized, b):
    return A_factorized.solve(b)


class PointcloudScalarDiffusion(DiffusionSolver):
    def __init__(
        self,
        pcloud,
        diffusion_scalar=1,
        # method="laplace",
        method="LU",
        num_eigen=None,
        num_integration_steps=1,
    ):

        super().__init__(
            diffusion_scalar=diffusion_scalar,
            method=method,
            num_eigen=num_eigen,
            num_integration_steps=num_integration_steps,
        )

        self.pcloud = pcloud
        if not hasattr(self.pcloud, "mean_edge_length"):
            self.pcloud.get_mean_edge_length()

        self.dt = self.diffusion_scalar * (self.pcloud.mean_edge_length**2)
        print(f"dt: {self.dt:.3e}")

        self.prefactored = False

    def load_parameters(self):
        config_filepath = self.pcloud.file_directory + "config.yaml"
        # Load the YAML file
        with open(config_filepath, "r") as file:
            config = yaml.safe_load(file)

        # Retrieve the parameters for the object
        params = config.get(self.pcloud.object_name, {})

        scalar_diffusion_dict = params.get("scalar_diffusion", {})
        if "vertices" in scalar_diffusion_dict:
            self.source_vertices = scalar_diffusion_dict["vertices"]
        else:
            self.source_vertices = None
        print(f"Source Vertices: {self.source_vertices}")

    def get_heat_method_heat_method_solver(self):
        self.heat_method_solver = pp3d.PointCloudHeatheat_method_solver(self.vertices)

    def solve_heat_method(self, sources):
        if not hasattr(self, "heat_method_solver"):
            self.get_heat_method_heat_method_solver()
        geodesic_arr = np.zeros((len(sources), len(self.vertices)))
        for i in range(len(sources)):
            if type(sources[i]) == list:
                geodesic_arr[i, :] = (
                    self.heat_method_solver.compute_distance_multisource(sources[i])
                )
            else:
                geodesic_arr[i, :] = self.heat_method_solver.compute_distance(
                    sources[i]
                )
        return geodesic_arr

    def precompute_geodesics_and_gradients(self, points):
        if not hasattr(self, "heat_method_solver"):
            self.get_heat_method_heat_method_solver()

        if not hasattr(self, "scalar_diffusion"):
            self.scalar_diffusion = PointcloudScalarDiffusion(self)
        geodesic_arr = np.zeros((len(points), len(self.vertices)))
        geodesic_gradient_arr = np.zeros((len(points), len(self.vertices), 3))
        for i in range(len(points)):
            if type(points[i]) == list:
                geodesic_arr[i, :] = (
                    self.heat_method_solver.compute_distance_multisource(points[i])
                )
            else:
                geodesic_arr[i, :] = self.heat_method_solver.compute_distance(points[i])
            self.scalar_diffusion.ut = geodesic_arr[i, :]
            self.scalar_diffusion.get_gradient()
            geodesic_gradient_arr[i, :, :] = self.scalar_diffusion.gradient_ut_3d

        return (
            geodesic_arr,
            geodesic_gradient_arr,
        )

    def set_sources(self):
        if not hasattr(self, "source_vertices"):
            self.load_parameters()
        u0 = np.zeros(len(self.pcloud.vertices))
        if len(self.source_vertices) == 2:
            u0[self.source_vertices[0]] = -1  # same behavior as the heat method
            u0[self.source_vertices[1]] = 1
        elif len(self.source_vertices) == 1:
            u0[self.source_vertices[0]] = -1  # for same behavior as the heat method
        else:
            for i in range(len(self.source_vertices)):
                u0[self.source_vertices[i]] = -1  # for same behavior as the heat method
        self.u0 = u0

    def get_local_bases(
        self,
    ):
        if not hasattr(self, "u0"):
            self.set_sources()

        # u0 = self.solve_heat_method(self.source_vertices)
        # self.scalar_diffusion.get_gradient(u0[0,:])
        # diffused_vectors = -self.scalar_diffusion.gradient_ut_3d

        self.prefactor_matrices()
        self.integrate_diffusion(self.u0)
        self.get_gradient()
        self.diffused_vectors = self.gradient_ut_3d  #

        self.pcloud.get_bases_from_tangent_vector_and_normal(self.diffused_vectors)

    def get_label(self):

        # TODO : Below would only work if the pcloud is already segmented to
        # target and obstacle and to boundaries in RGB channels
        # Red channel is for obstacle, green channel is for target,
        # blue channel is for boundary, target boundary, obstacle boundary combines
        # two channels accordingly
        # get the bool arrays
        is_obstacle = self.pcloud.colors[:, 0] > 0
        is_target = self.pcloud.colors[:, 1] > 0
        is_boundary = self.pcloud.colors[:, 2] > 0
        not_target = np.invert(is_target)
        not_obstacle = np.invert(is_obstacle)

        # combined bool arrays
        is_obstacle_boundary = np.logical_and(is_obstacle, is_boundary)
        is_target_boundary = np.logical_and(is_target, is_boundary)

        # added for interpolation experiments
        # is_neutral = np.logical_or(is_neutral, is_obstacle_boundary)

        # not target and not obstacle + target boundary
        # we need this because we will set the boundary condition
        # at the target boundary for attraction behavior
        is_neutral = np.logical_and(not_target, not_obstacle)
        self.is_neutral = np.logical_or(is_neutral, is_target_boundary)
        # we need this nested indexing because is_target_boundary
        # is the original point cloud size
        self.is_neutral_target_boundary = is_target_boundary[self.is_neutral]
        self.is_target_target_boundary = is_target_boundary[is_target]
        self.is_obstacle_obstacle_boundary = is_obstacle_boundary[is_obstacle]

        self.is_boundary = is_boundary
        self.is_free_boundary = np.logical_and(is_boundary, not_target)
        self.is_free_boundary = np.logical_and(self.is_free_boundary, not_obstacle)
        self.is_obstacle = is_obstacle
        self.is_target = is_target
        self.is_target_boundary = is_target_boundary
        self.is_obstacle_boundary = is_obstacle_boundary
        print(
            f"Out of {len(self.pcloud.vertices)} vertices "
            f"{np.sum(is_obstacle)} are obstacle, "
            f"{np.sum(is_target)} are target, "
            f"{np.sum(is_boundary)} are boundary"
        )

    def sample_points(self, num_samples, u0=None):
        sampled_points = []
        if u0 is not None:
            self.u0 = u0
        else:
            self.u0 = np.zeros(len(self.pcloud.vertices))
            random_vertex = random.randint(0, len(self.pcloud.vertices))
            self.u0[random_vertex] = 1
        for i in range(num_samples):
            ut = self.integrate_diffusion()
            vertex = np.argmin(ut)
            sampled_points.append(vertex)
            self.u0[vertex] = 1
        return sampled_points

    # Compute the heat diffusion for later computing the gradient
    # of the temperature field
    # =======================================================
    def integrate_diffusion(self, u0=None):
        if not self.prefactored:
            self.prefactor_matrices()
        # start_time = time.time()
        if u0 is None:
            ut = np.copy(self.u0)
        else:
            ut = np.copy(u0)
        for _ in range(self.num_integration_steps):
            if self.method == "invert":
                ut = self.A_invM @ ut
            elif self.method == "LU":
                ut = self.A_factorized.solve(self.M @ ut)
            elif self.method == "eigen":
                second_term = self.PhiT_M @ ut
                third_term = self.exp_vector * second_term
                ut = self.Phi @ third_term
            elif self.method == "laplace":
                ut = self.A_inv @ ut
            elif self.method == "LU_laplace":
                ut = self.A_factorized.solve(u0)

        self.ut = ut
        # self.max_ut = np.max(self.ut)
        # self.min_ut = np.min(self.ut)
        # self.ut_normalized = (self.ut - self.min_ut) / (self.max_ut - self.min_ut)
        # print(f"Integrated diffusion in {(time.time() - start_time)*1e3:.2f} ms")
        return np.copy(self.ut)

    def get_laplacian(self):
        start_time = time.time()
        C, M = robust_laplacian.point_cloud_laplacian(self.pcloud.vertices)
        print(f"Computed the Laplacian in {(time.time() - start_time)*1e3:.2f} ms")
        if hasattr(self.pcloud, "boundary_points"):
            for i in range(len(self.pcloud.boundary_points)):
                point = self.pcloud.boundary_points[i]
                C[point, :] = np.zeros(len(self.pcloud.vertices))
                C[point, point] = 1
            print("Dirichlet boundary conditions are set")
        else:
            print("Zero Neumann boundary conditions are used")

        self.C, self.M = C, M

    def prefactor_matrices(self):
        if not hasattr(self, "C"):
            self.get_laplacian()
        start_time = time.time()
        if self.method == "invert":
            self.A_inv = sla.inv(self.M + self.dt * self.C)
            self.A_invM = self.A_inv @ self.M
        elif self.method == "LU":
            A = csc_matrix(self.M + self.dt * self.C)  # Ensure sparse format
            self.A_factorized = splu(A)  # LU factorization
        elif self.method == "laplace":
            self.A_inv = sla.inv(self.C)
        elif self.method == "LU_laplace":
            A = csc_matrix(self.M + self.dt * self.C)  # Ensure sparse format
            self.A_factorized = splu(A)  # LU factorization
            # self.A_inv = sla.inv(self.C)
        elif self.method == "eigen":
            # compute the eigenvalue decomposition of Laplace-Beltrami
            print(f"Computing the first {self.num_eigen} eigenvalues")
            evals, evecs = sla.eigsh(self.C, self.num_eigen, self.M, sigma=1e-12)
            self.Phi = evecs
            self.exp_vector = np.zeros(self.num_eigen)
            for i in range(self.num_eigen):
                self.exp_vector[i] = np.exp(-evals[i] * self.dt)
            self.PhiT_M = self.Phi.T @ self.M
        print(f"Prefactored the Laplacian in {(time.time() - start_time)*1e3:.2f} ms")
        self.prefactored = True

    def get_gradient_operator(self):
        # Generate kNN graph
        edge_index = knn_graph(self.pcloud.vertices, 20)
        # Estimate normals and local frames
        self.normal, self.x_basis, self.y_basis = estimate_basis(
            self.pcloud.vertices, edge_index
        )
        self.local_bases = np.stack([self.x_basis, self.y_basis, self.normal], axis=2)

        # Build gradient and divergence operators (Scipy sparse matrices)
        self.grad, div = build_grad_div(
            self.pcloud.vertices, self.normal, self.x_basis, self.y_basis, edge_index
        )

    def get_gradient(self, ut=None):
        if not hasattr(self, "grad"):
            self.get_gradient_operator()
        # if not hasattr(self.pcloud, "boundary_normals"):
        #     self.pcloud.get_boundary_normals()
        if ut is None:
            ut = self.ut
        self.gradient_ut = self.grad @ ut

        self.gradient_ut_3d = self.project_from_pcloud_to_3d(self.gradient_ut)
        if hasattr(self.pcloud, "boundary_normals"):
            self.gradient_ut_3d[self.pcloud.is_boundary_arr, :] = (
                self.pcloud.boundary_normals
            )

    def project_from_pcloud_to_3d(self, ut):
        ut = ut.reshape(-1, 2)
        ut = ut[:, 0:1] * self.x_basis + ut[:, 1:] * self.y_basis
        ut = ut / np.linalg.norm(ut, axis=1).reshape(-1, 1)
        return ut

    def visualize_diffused_field(self):
        ps.init()
        ps_field = plot_point_cloud(vertices=self.pcloud.vertices, name="pcloud")
        ps_field.add_scalar_quantity(
            f"u0",
            self.u0,
        )

        ps_field.add_scalar_quantity(
            f"ut",
            self.ut,
        )

        ps.show()
