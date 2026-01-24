"""
Common utilities for Point Cloud Ergodic Control with UV parameterization.

This module contains shared functions used by both pcloud_uv_smc.py and
pcloud_uv_smc_animation.py scripts.
"""

import numpy as np
from scipy.interpolate import (
    LinearNDInterpolator,
    NearestNDInterpolator,
    griddata as scipy_griddata,
)


# Helper functions
# ===============================
def hadamard_matrix(n: int) -> np.ndarray:
    """
    Constructs a Hadamard matrix of size n.

    Args:
        n (int): The size of the Hadamard matrix.

    Returns:
        np.ndarray: A Hadamard matrix of size n.
    """
    if n == 1:
        return np.array([[1]])

    half_size = n // 2
    h_half = hadamard_matrix(half_size)

    h = np.empty((n, n), dtype=int)
    h[:half_size, :half_size] = h_half
    h[half_size:, :half_size] = h_half
    h[:half_size:, half_size:] = h_half
    h[half_size:, half_size:] = -h_half

    return h


def compute_uv_parameterization_pca(points):
    """
    Compute a 2D UV parameterization using PCA projection.
    Projects points onto the first two principal components.

    Args:
        points: (N, 3) array of 3D point coordinates

    Returns:
        uv_coords: (N, 2) array of UV coordinates normalized to [0, 1]
        points: (N, 3) array of 3D point positions (unchanged)
    """
    # Center the points
    centroid = points.mean(axis=0)
    centered = points - centroid

    # Compute PCA
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (largest first)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Project onto first two principal components
    uv_coords = centered @ eigenvectors[:, :2]

    # Normalize to [0, 1] domain
    uv_min = uv_coords.min(axis=0)
    uv_max = uv_coords.max(axis=0)
    uv_coords = (uv_coords - uv_min) / (uv_max - uv_min)

    return uv_coords, points


def compute_uv_parameterization_xy(points):
    """
    Compute a 2D UV parameterization using simple XY projection.

    Args:
        points: (N, 3) array of 3D point coordinates

    Returns:
        uv_coords: (N, 2) array of UV coordinates normalized to [0, 1]
        points: (N, 3) array of 3D point positions (unchanged)
    """
    # Simple projection onto XY plane
    uv_coords = points[:, :2].copy()

    # Normalize to [0, 1] domain
    uv_min = uv_coords.min(axis=0)
    uv_max = uv_coords.max(axis=0)
    uv_coords = (uv_coords - uv_min) / (uv_max - uv_min)

    return uv_coords, points


def create_uv_interpolator(uv_coords, points):
    """
    Create pre-built interpolators for fast UV to 3D projection.

    Args:
        uv_coords: (N, 2) array - UV coordinates of points
        points: (N, 3) array - 3D positions of points

    Returns:
        tuple: (linear_interpolator, nearest_interpolator)
    """
    linear_interp = LinearNDInterpolator(uv_coords, points)
    nearest_interp = NearestNDInterpolator(uv_coords, points)
    return linear_interp, nearest_interp


def project_uv_to_surface(uv_point, linear_interp, nearest_interp):
    """
    Project a 2D UV point back to the 3D surface using pre-built interpolators.

    Args:
        uv_point: (2,) array - UV coordinates to project
        linear_interp: Pre-built LinearNDInterpolator
        nearest_interp: Pre-built NearestNDInterpolator

    Returns:
        (3,) array - 3D position on the surface
    """
    # Try linear interpolation first
    point_3d = linear_interp(uv_point)

    # If linear fails (point outside convex hull), use nearest
    if np.any(np.isnan(point_3d)):
        point_3d = nearest_interp(uv_point)

    return point_3d


def setup_fourier_basis(nbFct, nbVar, nbRes, xlim, sp):
    """
    Setup Fourier basis functions for ergodic control.

    Args:
        nbFct: Number of basis functions along x and y
        nbVar: Dimension of datapoints (2 for UV space)
        nbRes: Resolution of the grid
        xlim: Domain limits [min, max]
        sp: Sobolev norm parameter

    Returns:
        dict containing:
            - rg: Range array for basis functions
            - KX: Meshgrid of basis function indices
            - Lambda: Sobolev weights
            - xm1d: 1D grid coordinates
            - xm: 2D meshgrid
            - arg1, arg2: Arguments for cosine terms
            - phim: Fourier basis matrix
            - xx, yy: Meshgrid for coefficient indexing
            - om: Frequency parameter
            - L: Domain size
    """
    L = (xlim[1] - xlim[0]) * 2
    om = 2 * np.pi / L

    rg = np.arange(0, nbFct, dtype=float)
    KX = np.zeros((nbVar, nbFct, nbFct))
    KX[0, :, :], KX[1, :, :] = np.meshgrid(rg, rg)
    Lambda = np.array(KX[0, :].flatten() ** 2 + KX[1, :].flatten() ** 2 + 1).T ** (
        -sp
    )

    xm1d = np.linspace(xlim[0], xlim[1], nbRes)
    xm = np.zeros((2, nbRes, nbRes))  # Changed from nbGaussian to 2
    xm[0, :, :], xm[1, :, :] = np.meshgrid(xm1d, xm1d)

    arg1 = (
        KX[0, :, :].flatten().T[:, np.newaxis]
        @ xm[0, :, :].flatten()[:, np.newaxis].T
        * om
    )
    arg2 = (
        KX[1, :, :].flatten().T[:, np.newaxis]
        @ xm[1, :, :].flatten()[:, np.newaxis].T
        * om
    )
    phim = np.cos(arg1) * np.cos(arg2) * 2 ** (nbVar)

    xx, yy = np.meshgrid(np.arange(1, nbFct + 1), np.arange(1, nbFct + 1))
    hk = np.concatenate(([1], 2 * np.ones(nbFct)))
    HK = hk[xx.flatten() - 1] * hk[yy.flatten() - 1]
    phim = phim * np.tile(HK, (nbRes**nbVar, 1)).T

    return {
        "rg": rg,
        "KX": KX,
        "Lambda": Lambda,
        "xm1d": xm1d,
        "xm": xm,
        "arg1": arg1,
        "arg2": arg2,
        "phim": phim,
        "xx": xx,
        "yy": yy,
        "om": om,
        "L": L,
    }


def compute_fourier_coefficients_from_red_channel(
    uv_coords, red_channel, xm1d, arg1, arg2, nbRes, nbVar, L
):
    """
    Compute Fourier series coefficients from red channel distribution.

    Args:
        uv_coords: (N, 2) array - UV coordinates of points
        red_channel: (N,) array - Red channel values normalized to [0, 1]
        xm1d: 1D grid coordinates
        arg1, arg2: Arguments for cosine terms
        nbRes: Resolution of the grid
        nbVar: Dimension of datapoints
        L: Domain size

    Returns:
        tuple: (w_hat, g_dist) - Fourier coefficients and gridded distribution
    """
    # Create grid points in UV space
    grid_u, grid_v = np.meshgrid(xm1d, xm1d)
    grid_points = np.column_stack([grid_u.flatten(), grid_v.flatten()])

    # Interpolate red channel values onto the grid
    g_dist = scipy_griddata(
        uv_coords, red_channel, grid_points, method="linear", fill_value=0.0
    )

    # Handle NaN values (points outside convex hull)
    nan_mask = np.isnan(g_dist)
    if np.any(nan_mask):
        g_dist_nearest = scipy_griddata(
            uv_coords, red_channel, grid_points, method="nearest"
        )
        g_dist[nan_mask] = g_dist_nearest[nan_mask]

    # Ensure non-negative and normalize
    g_dist = np.maximum(g_dist, 0)
    g_dist = g_dist * nbRes**nbVar / (np.sum(g_dist) + 1e-10)

    # Compute Fourier coefficients from the gridded distribution
    phi_inv = np.cos(arg1) * np.cos(arg2) / L**nbVar / nbRes**nbVar
    w_hat = phi_inv @ g_dist

    return w_hat, g_dist


class ErgodicControlUV:
    """
    Ergodic control on UV-parameterized point cloud with 3D speed control.
    """

    def __init__(
        self,
        x0,
        nbData,
        nbFct,
        nbVar,
        nbRes,
        dt,
        u_max,
        u_max_3d,
        u_norm_reg,
        xlim,
        fourier_basis,
        w_hat,
        linear_interp,
        nearest_interp,
        phim,
        enable_3d_speed_control=True,
        debug=True,
    ):
        """
        Initialize ergodic control.

        Args:
            x0: Initial position in UV space [u, v]
            nbData: Number of timesteps
            nbFct: Number of basis functions
            nbVar: Dimension (2 for UV)
            nbRes: Grid resolution
            dt: Time step
            u_max: Maximum UV speed
            u_max_3d: Maximum 3D speed (target speed)
            u_norm_reg: Regularizer for numerical stability
            xlim: Domain limits [min, max]
            fourier_basis: Dict from setup_fourier_basis()
            w_hat: Desired Fourier coefficients
            linear_interp: Linear interpolator for UV to 3D
            nearest_interp: Nearest interpolator for UV to 3D
            phim: Fourier basis matrix
            enable_3d_speed_control: Whether to use iterative 3D speed control
            debug: Whether to print debug info for first 3 timesteps
        """
        self.x0 = np.array(x0)
        self.nbData = nbData
        self.nbFct = nbFct
        self.nbVar = nbVar
        self.nbRes = nbRes
        self.dt = dt
        self.u_max = u_max
        self.u_max_3d = u_max_3d
        self.u_norm_reg = u_norm_reg
        self.xlim = xlim
        self.w_hat = w_hat
        self.linear_interp = linear_interp
        self.nearest_interp = nearest_interp
        self.phim = phim
        self.enable_3d_speed_control = enable_3d_speed_control
        self.debug = debug

        # Extract from fourier_basis
        self.rg = fourier_basis["rg"]
        self.KX = fourier_basis["KX"]
        self.Lambda = fourier_basis["Lambda"]
        self.xx = fourier_basis["xx"]
        self.yy = fourier_basis["yy"]
        self.om = fourier_basis["om"]
        self.L = fourier_basis["L"]

        # Initialize state
        self.x = self.x0.copy()
        self.prev_3d = project_uv_to_surface(
            self.x, self.linear_interp, self.nearest_interp
        )

        # Initialize tracking arrays
        self.wt = np.zeros(nbFct**nbVar)
        self.r_x_uv = np.zeros((nbVar, nbData))
        self.r_x_3d = np.zeros((3, nbData))
        self.r_g = np.zeros((nbRes**nbVar, nbData))
        self.r_w = np.zeros((nbFct**nbVar, nbData))
        self.r_e = np.zeros((nbData))
        self.r_speed_3d = np.zeros((nbData))

    def run(self):
        """
        Run the ergodic control loop.

        Returns:
            dict containing trajectory data and statistics
        """
        print("Running ergodic control...")

        for t in range(self.nbData):
            # Fourier basis functions and derivatives
            angle = self.x[:, np.newaxis] * self.rg * self.om
            phi1 = np.cos(angle) / self.L
            dphi1 = -np.sin(angle) * np.tile(self.rg * self.om, (self.nbVar, 1)) / self.L

            # Gradient of basis functions
            phix = phi1[0, self.xx - 1].flatten()
            phiy = phi1[1, self.yy - 1].flatten()
            dphix = dphi1[0, self.xx - 1].flatten()
            dphiy = dphi1[1, self.yy - 1].flatten()

            dphi = np.vstack([[dphix * phiy], [phix * dphiy]]).T

            # Fourier series coefficients along trajectory
            self.wt = self.wt + (phix * phiy).T
            w = self.wt / (t + 1)

            # Controller with constrained velocity norm
            u = -dphi.T @ (self.Lambda * (w - self.w_hat))
            u_dir = u / (np.linalg.norm(u) + self.u_norm_reg)
            u_uv = u_dir * self.u_max

            # Apply 3D speed control if enabled
            if self.enable_3d_speed_control:
                x_trial, point_trial, speed_3d = self._apply_3d_speed_control(
                    u_uv, t
                )
            else:
                x_trial = np.clip(self.x + (u_uv * self.dt), self.xlim[0], self.xlim[1])
                point_trial = project_uv_to_surface(
                    x_trial, self.linear_interp, self.nearest_interp
                )
                speed_3d = np.linalg.norm(point_trial - self.prev_3d) / self.dt

            # Update position
            self.x = x_trial
            self.x = np.clip(self.x, self.xlim[0], self.xlim[1])

            # Log data
            self.r_x_uv[:, t] = self.x
            self.r_x_3d[:, t] = point_trial
            self.r_speed_3d[t] = np.linalg.norm(point_trial - self.prev_3d) / self.dt
            self.prev_3d = point_trial

            self.r_g[:, t] = self.phim.T @ w
            self.r_w[:, t] = w
            self.r_e[t] = np.sum((w - self.w_hat) ** 2 * self.Lambda)

        print("Ergodic control complete!")
        print(f"Final reconstruction error: {self.r_e[-1]:.6f}")

        # Print statistics
        self._print_statistics()

        return {
            "trajectory_uv": self.r_x_uv,
            "trajectory_3d": self.r_x_3d,
            "reconstruction_error": self.r_e,
            "fourier_coefficients": self.r_w,
            "distribution": self.r_g,
            "speed_3d": self.r_speed_3d,
        }

    def _apply_3d_speed_control(self, u_uv, t):
        """
        Apply iterative 3D speed control to achieve target speed.

        Args:
            u_uv: UV space velocity
            t: Current timestep

        Returns:
            tuple: (x_trial, point_trial, speed_3d)
        """
        max_iters = 10
        speed_tolerance = 0.05
        target_speed = self.u_max_3d

        # Debug tracking
        if self.debug and t < 3:
            debug_info = []

        for iter_count in range(max_iters):
            x_trial = np.clip(
                self.x + (u_uv * self.dt), self.xlim[0], self.xlim[1]
            )
            point_trial = project_uv_to_surface(
                x_trial, self.linear_interp, self.nearest_interp
            )

            # Check 3D speed
            speed_3d = np.linalg.norm(point_trial - self.prev_3d) / self.dt

            if self.debug and t < 3:
                debug_info.append(
                    {
                        "iter": iter_count,
                        "speed_3d": speed_3d,
                        "target": target_speed,
                        "u_uv_norm": np.linalg.norm(u_uv),
                    }
                )

            # Check if speed is within acceptable range
            if abs(speed_3d - target_speed) / target_speed <= speed_tolerance:
                break

            if speed_3d < 1e-6:  # Avoid division by zero
                break

            # Scale u_uv to achieve target speed
            scale = target_speed / speed_3d
            scale = np.clip(scale, 0.5, 1.5)
            u_uv = u_uv * scale

        if self.debug and t < 3:
            print(f"\n=== Timestep {t} Debug ===")
            for info in debug_info:
                print(
                    f"  Iter {info['iter']}: speed_3d={info['speed_3d']:.6f}, "
                    f"target={info['target']:.6f}, u_uv_norm={info['u_uv_norm']:.6f}"
                )
            print(
                f"  Final: converged in {iter_count+1} iterations, final speed={speed_3d:.6f}"
            )

        return x_trial, point_trial, speed_3d

    def _print_statistics(self):
        """Print trajectory statistics."""
        if self.enable_3d_speed_control:
            print(
                f"3D Speed stats: min={self.r_speed_3d.min():.4f}, "
                f"max={self.r_speed_3d.max():.4f}, "
                f"mean={self.r_speed_3d.mean():.4f}, "
                f"std={self.r_speed_3d.std():.4f}"
            )
            print(f"3D Speed limit was set to: {self.u_max_3d}")
            print(
                f"Percentage of steps at speed limit: "
                f"{100 * np.sum(self.r_speed_3d >= self.u_max_3d * 0.99) / self.nbData:.2f}%"
            )
