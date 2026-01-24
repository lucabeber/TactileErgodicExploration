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

from pathlib import Path

import numpy as np
import open3d as o3d
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R


class Manifold:

    def __init__(
        self,
        type=None,
        scale=None,
        translation=None,
        rotation=None,
    ):
        self.type = type
        self.scale = scale
        self.translation = translation
        self.rotation = rotation


class Sphere(Manifold):
    def __init__(
        self, radius=1.0, center=(0, 0, 0), scale=None, translation=None, rotation=None
    ):
        # Initialize the base Manifold class
        super().__init__(
            type="Sphere", scale=scale, translation=translation, rotation=rotation
        )

        # Sphere-specific attributes
        self.radius = radius
        self.center = np.array(center)

    def apply_scale(self):
        """Apply scaling to the radius."""
        if self.scale is not None:
            self.radius *= self.scale

    def get_south_north_poles(self):
        # Define the south pole and north pole in world coordinates
        self.south_pole = self.center + np.array([0.0, 0.0, -self.radius])
        self.north_pole = self.center + np.array([0.0, 0.0, +self.radius])

    def get_left_right_poles(self):

        self.left_pole = self.center + np.array([0.0, -self.radius, 0.0])
        self.right_pole = self.center + np.array([0.0, +self.radius, 0.0])

    def apply_translation(self):
        """Translate the sphere's center."""
        if self.translation is not None:
            self.center += np.array(self.translation)

    def __repr__(self):
        return (
            f"Sphere(radius={self.radius}, center={self.center.tolist()}, "
            f"scale={self.scale}, translation={self.translation}, rotation={self.rotation})"
        )

    def get_local_bases_south_to_north(self, points_on_sphere):
        """
        Computes the vector field for a batch of points on the surface of the sphere.
        The field emanates from the south pole (0, 0, -radius) and sinks into the north
        pole (0, 0, +radius).

        :self.sphere_center: The center of the sphere (3D vector)
        :self.sphere_radius: The radius of the sphere (scalar)
        :self.batch_points_on_sphere: A batch of points on the surface of the sphere
        (Nx3 matrix)
        :return: An Nx3 matrix representing the vector field for each point in world
        coordinates.
        """
        self.get_south_north_poles()
        # Step 1: Compute the radial vectors from the south pole to each point on the sphere
        radial_vectors = points_on_sphere - self.south_pole  # Nx3 matrix

        # Step 2: Compute the normal vectors (center to each point on the sphere)
        normal_vectors = points_on_sphere - self.center  # Nx3 matrix

        # Normalize the normal vectors (get the direction for each point)
        normal_unit_vectors = normal_vectors / np.linalg.norm(
            normal_vectors, axis=1, keepdims=True
        )  # Nx3 matrix

        # Step 3: Project the radial vectors onto the tangent plane at each point (i.e., remove the normal component)
        dot_products = np.sum(
            radial_vectors * normal_unit_vectors, axis=1, keepdims=True
        )  # Nx1 vector of dot products
        tangent_vectors = (
            radial_vectors - dot_products * normal_unit_vectors
        )  # Nx3 matrix (tangent vectors)

        # Step 4: Normalize the tangent vectors to get unit vectors
        tangent_unit_vectors = tangent_vectors / np.linalg.norm(
            tangent_vectors, axis=1, keepdims=True
        )  # Nx3 matrix
        tangent_unit_vectors_2 = np.cross(normal_unit_vectors, tangent_unit_vectors)

        sphere_basis = np.stack(
            (tangent_unit_vectors, tangent_unit_vectors_2, normal_unit_vectors), axis=2
        )

        return sphere_basis

    def get_local_bases_east_west(self, points_on_sphere):
        """
        Computes the vector field for a batch of points on the surface of the sphere.
        The field emanates from the south pole (0, 0, -radius) and sinks into the north
        pole (0, 0, +radius).

        :self.sphere_center: The center of the sphere (3D vector)
        :self.sphere_radius: The radius of the sphere (scalar)
        :self.batch_points_on_sphere: A batch of points on the surface of the sphere
        (Nx3 matrix)
        :return: An Nx3 matrix representing the vector field for each point in world
        coordinates.
        """
        self.get_left_right_poles()

        # self.get_south_north_poles()
        # Step 1: Compute the radial vectors from the south pole to each point on the sphere
        radial_vectors = points_on_sphere - self.left_pole  # Nx3 matrix
        # radial_vectors = points_on_sphere - self.south_pole  # Nx3 matrix

        # Step 2: Compute the normal vectors (center to each point on the sphere)
        normal_vectors = points_on_sphere - self.center  # Nx3 matrix

        # Normalize the normal vectors (get the direction for each point)
        normal_unit_vectors = -normal_vectors / np.linalg.norm(
            normal_vectors, axis=1, keepdims=True
        )  # Nx3 matrix

        # Step 3: Project the radial vectors onto the tangent plane at each point (i.e., remove the normal component)
        dot_products = np.sum(
            radial_vectors * normal_unit_vectors, axis=1, keepdims=True
        )  # Nx1 vector of dot products
        tangent_vectors = (
            radial_vectors - dot_products * normal_unit_vectors
        )  # Nx3 matrix (tangent vectors)

        # Step 4: Normalize the tangent vectors to get unit vectors
        tangent_unit_vectors = tangent_vectors / np.linalg.norm(
            tangent_vectors, axis=1, keepdims=True
        )  # Nx3 matrix
        tangent_unit_vectors = np.cross(normal_unit_vectors, tangent_unit_vectors)
        tangent_unit_vectors_2 = np.cross(normal_unit_vectors, tangent_unit_vectors)

        sphere_basis = np.stack(
            (tangent_unit_vectors, tangent_unit_vectors_2, normal_unit_vectors),
            axis=2,
        )

        return sphere_basis


def bilinear_interpolation_2(grid, pos):
    """
    Perform bilinear interpolation on a 2-D grid.

    Parameters:
    -----------
    grid: numpy.ndarray
        A 2D numpy array representing the grid to interpolate on.
    pos: Tuple[float, float]
        A tuple containing the x and y coordinates of the point to
        interpolate at.

    Returns:
    --------
    int
        The interpolated value at the given point.
    """
    x_arr = np.linspace(0, 1, grid.shape[0])
    y_arr = np.linspace(0, 1, grid.shape[1])
    # find the index of the closest smaller pixel after resampling between
    # [0,1]
    for i in range(grid.shape[0]):
        if pos[0] <= x_arr[i]:
            x = i - 1
            break

    # find the index of the closest smaller pixel after resampling between
    # [0,1]
    for j in range(grid.shape[1]):
        if pos[1] <= y_arr[j]:
            y = j - 1
            break

    def border_interpolate(value, size):
        """
        Interpolate the value for the border handling.

        Parameters:
        -----------
        value: int
            The original value.
        size: int
            The size of the dimension.

        Returns:
        --------
        int
            The interpolated value for the border handling.
        """
        return max(0, min(size - 1, value))

    # find the nearest integers by minding the borders
    x0 = border_interpolate(x, grid.shape[1])
    x1 = border_interpolate(x + 1, grid.shape[1])
    y0 = border_interpolate(y, grid.shape[0])
    y1 = border_interpolate(y + 1, grid.shape[0])

    # Distance from lower integers
    xd = pos[0] - x_arr[x0]
    yd = pos[1] - y_arr[y0]

    # Interpolate on x-axis
    c01 = grid[y0, x0] * (1 - xd) + grid[y0, x1] * xd
    c11 = grid[y1, x0] * (1 - xd) + grid[y1, x1] * xd
    # Interpolate on y-axis
    c = c01 * (1 - yd) + c11 * yd
    return int(c)


def import_system_matrix(
    grid,
    obj_name,
    method,
    dt,
    directory="/Users/cembilaloglu/repos/grid_diffusion/data",
):
    """
    For large grids, it is more efficient to save the system matrix and load it
    """
    filename = f"{directory}/grid_{grid.Nx}_{obj_name}_{method}.npy"
    try:

        # Try to load the laplacian from file

        # Object name matters because the scale is different?
        invA = np.load(filename, allow_pickle=True)
        print("invA loaded from file.")

    except FileNotFoundError:
        print("File not found. Creating the laplacian matrix...")
        # If the file doesn't exist, preprocess the laplacian and save it
        L = laplacian_3d_matrix(grid)
        if method == "laplace":
            invA = sla.inv(-L)

        elif method == "diffusion":
            A = get_system_matrix_implicit_heat(L, dt=dt)
            invA = sla.inv(A)

        import pickle

        with open(filename, "wb") as f:
            pickle.dump(invA, f, protocol=4)
    return invA


def find_closest_point_on_grid(grid_vertices, point):
    """
    Finds the index of the closest point in the grid_vertices array to the given point.

    Parameters:
    grid_vertices (numpy.ndarray): Array of grid vertices.
    point (numpy.ndarray): The point to find the closest point to.

    Returns:
    int: The index of the closest point in the grid_vertices array.
    """
    dists = np.linalg.norm(grid_vertices - point, axis=1)
    min_idx = np.argmin(dists)
    return min_idx


def laplacian_3d_matrix(grid):
    """
    TODO: I am not sure whether we use the sparsity structure
    to the best extent that we can

    Generate a sparse Laplacian matrix for a 3D grid.

    Parameters:
    - Nx (int): Number of grid points along the x-axis.
    - Ny (int): Number of grid points along the y-axis.
    - Nz (int): Number of grid points along the z-axis.
    - h (float): Uniform grid spacing.

    Returns:
    - laplacian (scipy.sparse.csr_matrix): Sparse Laplacian matrix.

    The Laplacian matrix is generated using the finite difference method.
    It represents the discretized Laplace operator for a 3D grid.
    The matrix is returned in Compressed Sparse Row (CSR) format for efficient storage and computation.
    """

    N = grid.Nx * grid.Ny * grid.Nz
    data = []
    rows = []
    cols = []

    def index(x, y, z):
        """
        Calculates the index of a point in a 3D grid.

        Parameters:
        x (int): The x-coordinate of the point.
        y (int): The y-coordinate of the point.
        z (int): The z-coordinate of the point.

        Returns:
        int: The calculated index of the point in the grid.
        """
        return x * (grid.Ny * grid.Nz) + y * grid.Nz + z

    for x in range(grid.Nx):
        for y in range(grid.Ny):
            for z in range(grid.Nz):
                i = index(x, y, z)
                rows.append(i)
                cols.append(i)
                data.append(-6 / grid.h**2)  # Center point

                if x > 0:
                    rows.append(i)
                    cols.append(index(x - 1, y, z))
                    data.append(1 / grid.h**2)
                if x < grid.Nx - 1:
                    rows.append(i)
                    cols.append(index(x + 1, y, z))
                    data.append(1 / grid.h**2)
                if y > 0:
                    rows.append(i)
                    cols.append(index(x, y - 1, z))
                    data.append(1 / grid.h**2)
                if y < grid.Ny - 1:
                    rows.append(i)
                    cols.append(index(x, y + 1, z))
                    data.append(1 / grid.h**2)
                if z > 0:
                    rows.append(i)
                    cols.append(index(x, y, z - 1))
                    data.append(1 / grid.h**2)
                if z < grid.Nz - 1:
                    rows.append(i)
                    cols.append(index(x, y, z + 1))
                    data.append(1 / grid.h**2)

    laplacian = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    return laplacian


class Pointcloud(Manifold):
    def __init__(
        self,
        vertices=None,
        colors=None,
        filename=None,
        voxel_size=None,
        scale=None,
        translation=None,
        rotation=None,
        normal_orientation=None,
        file_directory=None,
        cluster_points=True,
        *args,
        **kwargs,
    ):
        super().__init__(
            type=type(self), scale=scale, translation=translation, rotation=rotation
        )

        self.file_directory = file_directory
        self.voxel_size = voxel_size
        self.normal_orientation = normal_orientation

        if self.file_directory is None:
            script_path = Path(__file__).resolve()
            self.file_directory = (
                str(script_path.parent.parent / "data" / "pointclouds/") + "/"
            )

        # Construct the point cloud
        # ==============================================================================
        object_name = "default"
        if vertices is not None:
            pcd_vertices = o3d.utility.Vector3dVector(vertices)
            pcd_tmp = o3d.geometry.PointCloud(pcd_vertices)
            num_vertices_original = vertices.shape[0]
        elif filename is not None:  # initialize from file
            object_name = filename.split(".")[0]
            filepath = self.file_directory + filename
            print(f"Reading point cloud from {filepath}")
            pcd_tmp = o3d.io.read_point_cloud(filepath)
            vertices = np.asarray(pcd_tmp.points)
            num_vertices_original = vertices.shape[0]
        else:  # initialize with an empty point cloud
            self.point_cloud = o3d.geometry.PointCloud()
            num_vertices_original = 0
        self.object_name = object_name

        # self.load_object_parameters()

        if colors is not None:
            pcd_tmp.colors = o3d.utility.Vector3dVector(colors)

        # pcd_tmp.scale(self.scale, center=pcd_tmp.get_center())
        # pcd_tmp.rotate(self.rotation.as_matrix(), center=pcd_tmp.get_center())
        # pcd_tmp.translate(self.translation, relative=True)

        # Downsample the point cloud
        # ==============================================================================
        if self.voxel_size is None:
            pcd = pcd_tmp
        else:
            pcd = pcd_tmp.voxel_down_sample(voxel_size=self.voxel_size)  # downsample
        self.pcd = pcd
        num_vertices_downsampled = len(pcd.points)

        # Transform the point cloud
        # ==============================================================================
        self.vertices = np.asarray(self.pcd.points)
        self.colors = np.asarray(self.pcd.colors)
        if self.colors.size == 0:  # pointcloud have no color
            self.colors = np.zeros_like(self.vertices)

        # if cluster_points:
        #     self.cluster_points()
        # print(
        #     f"Original Point cloud with {num_vertices_original}"
        #     + f" points is downsampled with voxel size {self.voxel_size}"
        #     + f"\n resulted in {num_vertices_downsampled} points"
        # )

    def get_center(self):
        if not hasattr(self, "kd_tree"):
            self.get_kd_tree()
        # Find the center point of the point cloud
        self.center_point = np.mean(
            self.vertices, axis=0
        )  # this is not necessarily on the pointcloud

        _, center_vertex = self.kd_tree.query(
            np.array([self.center_point]), k=1
        )  # this is on the pointcloud
        self.center_vertex = center_vertex[0]
        return self.center_point, self.center_vertex

    def get_kd_tree(self):
        # Construct the KD-tree and adjacency graph
        # ==============================================================================
        self.kd_tree = cKDTree(self.vertices)
        return self.kd_tree

    def get_boundary_kd_tree(self):
        # Construct the KD-tree and adjacency graph
        # ==============================================================================
        self.boundary_kd_tree = cKDTree(self.vertices[self.is_boundary_arr])
        return self.boundary_kd_tree

    def load_object_parameters(self):
        config_filepath = self.file_directory + "config.yaml"
        # Load the YAML file
        with open(config_filepath, "r") as file:
            config = yaml.safe_load(file)

        # Retrieve the parameters for the object
        params = config.get(self.object_name, {})

        if self.voxel_size is None:
            if "voxel_size" in params:
                self.voxel_size = params["voxel_size"]
            else:
                self.voxel_size = None
        if self.scale is None:
            if "scale" in params:
                self.scale = params["scale"]
            else:
                self.scale = 1.0
        if self.translation is None:
            if "translation" in params:
                self.translation = params["translation"]
            else:
                self.translation = np.array([0.0, 0.0, 0.0])
        if self.rotation is None:
            if "rotation" in params:
                euler_params = params["rotation"]
                self.rotation = R.from_euler("xyz", euler_params, degrees=True)
            else:
                self.rotation = R.from_euler("xyz", [0, 0, 0], degrees=True)

        if self.normal_orientation is None:
            if "normal_orientation" in params:
                self.normal_orientation = params["normal_orientation"]
            else:
                self.normal_orientation = 1

        # Now, you can use scale_factor, rot, and voxel_size
        print(f"scale_factor: {self.scale}")
        print(f"rot: {self.rotation.as_euler('xyz', degrees=True)}")
        print(f"voxel_size: {self.voxel_size}")
        print(f"translation: {self.translation}")
        print(f"normal_orientation: {self.normal_orientation}")

    def get_rotations(self, boundary_points):
        if not hasattr(self, "diffused_rotations"):
            print("Diffused rotations are not available.")
        boundary_vertices = boundary_points[:, 0].astype(
            int
        )  # this the way that I handle in wos for efficiency
        print(boundary_vertices)
        return self.diffused_rotations[boundary_vertices]

    def get_closest_points(self, points):
        if not hasattr(self, "kd_tree"):
            self.get_kd_tree()
        distances, indices = self.kd_tree.query(points, k=1)
        return distances, indices

    def get_bounding_box(self):
        self.oriented_bounding_box = self.pcd.get_oriented_bounding_box()
        # Get the corner points of the OBB to compute the radius of the enclosing sphere
        self.oriented_bounding_box_corners = np.asarray(
            self.oriented_bounding_box.get_box_points()
        )
        return self.oriented_bounding_box

    def get_bases_from_tangent_vector_and_normal(self, tangent_vector):
        if not hasattr(self, "normals"):
            self.get_normals()

        y_vector = np.cross(self.normals, tangent_vector)

        tangent_vector = (
            tangent_vector / np.linalg.norm(tangent_vector, axis=1)[:, np.newaxis]
        )
        if np.any(np.isnan(tangent_vector)):
            print("ERROR! NaN in vector in diffuse_rotations")
        y_vector = y_vector / np.linalg.norm(y_vector, axis=1)[:, np.newaxis]
        normals = self.normals / np.linalg.norm(self.normals, axis=1)[:, np.newaxis]
        print(f"Shape of the tangent_vector is : {tangent_vector.shape}")
        print(f"Shape of the normals is : {normals.shape}")
        local_bases = np.stack([tangent_vector, y_vector, normals], axis=2)
        # Consider the coordinate frames at vertices as rotations transforming the world
        # coordinate frame to the local coordinate frame at the vertex. Represent the
        # rotations as quaternions. Map the quaternions to the Lie algebra elements
        self.local_bases = local_bases

    def get_bounding_sphere(self):
        if not hasattr(self, "center_point"):
            self.get_center()
        if not hasattr(self, "oriented_bounding_box"):
            self.get_bounding_box()
        sphere_center = self.center_point

        # Compute the distance from the center to the farthest corner (this is the radius
        #  of the enclosing sphere)
        enclosing_sphere_radius = np.max(
            np.linalg.norm(
                self.oriented_bounding_box_corners - self.oriented_bounding_box.center,
                axis=1,
            )
        )
        self.bounding_sphere = Sphere(
            radius=enclosing_sphere_radius, center=sphere_center
        )

    def surround_object_with_sphere(self, radius_scalar=2.0):
        if not hasattr(self, "bounding_sphere"):
            self.get_bounding_sphere()

        sphere_pcloud = Pointcloud(
            filename="sphere_1k_uniform.ply",
            voxel_size=None,
            scale=self.bounding_sphere.radius * radius_scalar,
            translation=self.bounding_sphere.center,
        )

        combined_vertices = np.concatenate(
            (self.vertices, sphere_pcloud.vertices), axis=0
        )
        self.combined_pcloud = Pointcloud(vertices=combined_vertices)

        self.sphere_pcloud = sphere_pcloud

    def surround_object_with_hemisphere(self, radius_scalar=2.0):
        if not hasattr(self, "bounding_sphere"):
            self.get_bounding_sphere()

        sphere_pcloud = Pointcloud(
            filename="sphere_1k_uniform_top_hemisphere.ply",
            voxel_size=None,
            scale=self.bounding_sphere.radius * radius_scalar,
            translation=self.bounding_sphere.center,
        )

        combined_vertices = np.concatenate(
            (self.vertices, sphere_pcloud.vertices), axis=0
        )
        self.combined_pcloud = Pointcloud(vertices=combined_vertices)

        self.sphere_pcloud = sphere_pcloud

    def get_bounding_box_grid(self, bounding_box_scalar=2.0, nb_points=5):
        if not hasattr(self, "oriented_bounding_box"):
            self.get_bounding_box()

        self.oriented_bounding_box.scale(
            bounding_box_scalar, center=self.oriented_bounding_box.get_center()
        )

        # Find the min and max for x, y, z
        min_vals = np.min(self.oriented_bounding_box_corners, axis=0)
        max_vals = np.max(self.oriented_bounding_box_corners, axis=0)

        x_min, y_min, z_min = min_vals
        x_max, y_max, z_max = max_vals
        grid = Grid(
            Nx=nb_points,
            Ny=nb_points,
            Nz=nb_points,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
        )
        return grid

    def get_signed_distance(self, position):
        distance, point_index = self.kd_tree.query(position, k=1)
        projected_point = self.vertices[point_index]
        projected_normal = self.normals[point_index]

        # if the direction from the point to its projection to the surface
        # is in the same direction with the normal, the distance is positive
        sign = np.sign(np.dot(position - projected_point, projected_normal))
        signed_distance = distance * sign
        return signed_distance, point_index

    def correct_distance_smooth(
        self, position, distance_target, epsilon=1e-3, max_iterations=10, max_error=1e-1
    ):
        position = np.float32(position)
        for i in range(max_iterations):
            # actual_distance, index = self.kd_tree.query(np.array([position]), k=1)
            signed_distance, point_index = self.get_signed_distance(position)
            error = distance_target - signed_distance
            # print(
            #     f"Distance error: {error*1e3:.1f}, target: {distance_target*1e3:.1f},
            # actual: {signed_distance*1e3:.1f} in mm"
            # )
            if np.abs(error) < epsilon:
                return position, signed_distance, point_index
            elif np.abs(error) > max_error:
                error_sign = np.sign(error)
                error = error_sign * max_error

            # local_basis, _d, _ = wos_local_basis(position, pcloud, wos_param)
            # local_normal = local_basis[:, 2]
            local_normal = self.normals[point_index]
            correction = error * 0.5 * local_normal
            position += correction
        print(f"Could not correct the distance after {max_iterations} iterations")
        return position, signed_distance, point_index

    def cluster_points(self, distance_threshold=1e-2, min_points=10):
        # Apply DBSCAN clustering (eps is the distance threshold, min_points is the
        # minimum number of points in a cluster)
        labels = np.array(
            self.pcd.cluster_dbscan(eps=distance_threshold, min_points=min_points)
        )

        # Number of clusters (ignoring noise points, labeled as -1)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        print(f"Clustering points...{num_clusters}")
        if num_clusters > 0:
            unique_labels, num_points = np.unique(labels, return_counts=True)
            desired_label = unique_labels[np.argmax(num_points)]
            print(f"Number of clusters: {num_clusters}")
            # Check if cluster 1 exists in the labels
            print(
                f"Cluster {desired_label} has {num_points[np.argmax(num_points)]} points"
            )
            if desired_label not in labels:
                print(f"Cluster {desired_label} not found in the point cloud.")
            else:
                # Filter points corresponding to cluster 1
                indices_cluster = np.where(labels == desired_label)[0]

                # Select points corresponding to cluster 1
                self.pcd = self.pcd.select_by_index(indices_cluster)
        self.vertices = np.asarray(self.pcd.points)
        self.colors = np.asarray(self.pcd.colors)

    def get_normals(self, num_neighbors=30):
        if not hasattr(self, "center_vertex"):
            self.get_center()

        self.pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=num_neighbors)
        )
        # force all the normals point to the same direction: outwards from the object

        self.pcd.orient_normals_consistent_tangent_plane(k=num_neighbors)

        normals = np.asarray(self.pcd.normals)
        # outward = self.center_vertex - self.center_point
        # sign = np.sign(np.dot(outward, normals[self.center_vertex]))
        # normals *= sign
        # self.normals = normals * self.normal_orientation
        self.normals = normals
        return self.normals

    def get_local_basis(self):
        if not hasattr(self, "normals"):
            self.get_normals()
        # set u vector as a vector orthogonal to the normal
        self.tangent_vectors_u = np.column_stack(
            [self.normals[:, 1], -self.normals[:, 0], np.zeros(len(self.vertices))]
        )
        # ordering matters for the right handedness
        self.tangent_vectors_v = np.cross(self.normals, self.tangent_vectors_u)

    def get_k_edges(self, num_neighbors=3):
        """
        Calculate the edges between points based on the KD-tree.

        Parameters:
        -----------
        metric: numpy.ndarray
            The metric to calculate the edges on.
        num_neighbours: int
            The number of nearest neighbors to consider.

        Returns:
        --------
        numpy.ndarray
            The edges between points.
        """
        if not hasattr(self, "kd_tree"):
            self.get_kd_tree()
        # Query the num_neighbor neighbourhoods for each point of the selected feature
        # space to each point
        self.d_kdtree, idx = self.kd_tree.query(self.vertices, k=num_neighbors)

        # Remove the first point in the neighborhood as this is just the
        # queried point itself
        idx = idx[:, 1:]

        # Create the edges array between all the points and their closest
        # neighbours
        point_numbers = np.arange(len(self.vertices))
        # Repeat each point in the point numbers array the number of closest
        # neighbours -> 1,2,3,4... becomes 1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4...
        point_numbers = np.repeat(point_numbers, num_neighbors - 1)
        # Flatten  the neighbour indices array -> from [1,3,10,14], [4,7,17,23]
        # , ... becomes [1,3,10,4,7,17,23,...]
        idx_flatten = idx.flatten()
        # Create the edges array by combining the two other ones as a vertical
        # stack and transposing them to get the input that LineSet requires
        edges = np.vstack((point_numbers, idx_flatten)).T

        return edges

    def get_mean_edge_length(self):
        """
        Calculate the mean edge length of a point cloud.

        Parameters:
        -----------
        vertices : numpy.ndarray
            The vertices of the point cloud.

        Returns:
        --------
        float
            The mean edge length of the point cloud.
        """
        # if self.voxel_size is None:
        #     edges = self.get_k_edges()
        #     edge_vectors = self.vertices[edges[:, 1], :] - self.vertices[edges[:, 0], :]
        #     edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        #     self.mean_edge_length = np.mean(edge_lengths)
        # else:
        #     self.mean_edge_length = self.voxel_size

        edges = self.get_k_edges()
        edge_vectors = self.vertices[edges[:, 1], :] - self.vertices[edges[:, 0], :]
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        self.mean_edge_length = np.mean(edge_lengths)

        print(f"Mean edge length: {self.mean_edge_length*1e3:.1f} mm")
        return self.mean_edge_length

    def get_boundary(
        self,
        max_neighbors=30,
        angle_threshold=np.pi / 3,
    ):
        def is_boundary(
            vertex,
            neighbor_vertices,
            tangent_vector_u,
            tangent_vector_v,
            angle_threshold=np.pi / 2,
        ):
            """
            Given a point and its neighbors, this function checks if the point is on
            the boundary
            Inspired by the implementation of the boundaryEstimation() function of PCL
            https://pointclouds.org/documentation/boundary_8hpp_source.html
            """
            # Compute the angles between the point and its neighbors
            # in the tangent plane
            angles = np.zeros(len(neighbor_vertices))
            for j in range(len(neighbor_vertices)):
                neighbor = neighbor_vertices[j]
                delta = neighbor - vertex
                angles[j] = np.arctan2(
                    np.dot(delta, tangent_vector_u), np.dot(delta, tangent_vector_v)
                )

            # Sort the angles and get the maximum difference
            angles = np.sort(angles)
            diff = np.zeros_like(angles)
            diff = np.diff(angles)
            # Get the angle difference between the last and the first
            diff[-1] = 2 * np.pi - angles[-1] + angles[0]
            # Check the angle condition for boundary
            if np.max(diff) > angle_threshold:
                return True
            return False

        if not hasattr(self, "kd_tree"):
            self.get_kd_tree()
        if not hasattr(self, "tangent_vectors_u"):
            self.get_local_basis()
        is_boundary_arr = np.zeros(len(self.vertices)).astype(bool)
        for i in range(len(self.vertices)):
            vertex = self.vertices[i]
            # exclude the current vertex from the search
            # [_, idx, _] = self.kd_tree.search_knn_vector_3d(vertex, max_neighbors)
            distances, indices = self.kd_tree.query(vertex, k=max_neighbors)
            neighbor_vertices = self.vertices[indices[0:]]
            is_boundary_arr[i] = is_boundary(
                vertex,
                neighbor_vertices,
                self.tangent_vectors_u[i],
                self.tangent_vectors_v[i],
                angle_threshold,
            )
        self.is_boundary_arr = is_boundary_arr

        return is_boundary_arr

    def get_boundary_normals(self):
        """
        Computes normal vectors at the boundary vertices that point inside the point
        cloud.

        Returns:
        - boundary_vertices: np.ndarray, the coordinates of the boundary vertices.
        - boundary_normal_vectors: np.ndarray, the normal vectors at boundary points
        pointing inward.
        """
        if not hasattr(self, "is_boundary_arr"):

            self.get_boundary()
        if not hasattr(self, "kd_tree"):
            self.get_kd_tree()

        # Step 1: Get boundary vertices
        boundary_vertices = self.vertices[self.is_boundary_arr]
        # self.pcd.compute_boundary_points(radius=0.01)
        # boundaries, mask = self.pcd.compute_boundary_points(0.01, 30)

        # Step 2: Compute normal vectors at the boundary points
        boundary_normal_vectors = np.zeros_like(boundary_vertices)

        for i, vertex in enumerate(boundary_vertices):
            # Query all neighbors (not just boundary vertices) for normal estimation
            distances, indices = self.kd_tree.query(vertex, k=10)
            neighbor_vertices = self.vertices[indices[1:]]  # Exclude the vertex itself

            # Step 3: Compute inward-pointing vector by averaging direction to
            # neighboring points
            inward_direction = np.mean(neighbor_vertices - vertex, axis=0)
            inward_direction /= np.linalg.norm(inward_direction)  # Normalize

            # Optionally, project onto tangent plane if you want the vector in the plane
            # of the boundary stripe
            tangent_vector_u = self.tangent_vectors_u[self.is_boundary_arr][i]
            tangent_vector_v = self.tangent_vectors_v[self.is_boundary_arr][i]

            # Project the inward direction onto the tangent vectors
            projected_u = np.dot(inward_direction, tangent_vector_u) * tangent_vector_u
            projected_v = np.dot(inward_direction, tangent_vector_v) * tangent_vector_v
            tangent_projection = projected_u + projected_v

            tangent_projection /= np.linalg.norm(
                tangent_projection
            )  # Normalize the projection

            # Store the normal vector (the inward direction)
            boundary_normal_vectors[i] = tangent_projection
        self.boundary_normals = boundary_normal_vectors
        self.boundary_tangents = np.cross(
            boundary_normal_vectors, self.normals[self.is_boundary_arr]
        )

        return boundary_normal_vectors
