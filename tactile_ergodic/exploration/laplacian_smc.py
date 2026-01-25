from pathlib import Path

import numpy as np
import robust_laplacian
import scipy.sparse as sp
from scipy.sparse import linalg as sla
from scipy.spatial import cKDTree

from ..utils.pointcloud_utils import (
    compute_tangent_space,
    process_point_cloud,
    project_points2tangent_space,
)


class LaplacianSMCController:
    def __init__(
        self,
        points_N3: np.ndarray,
        M_NN: sp.spmatrix,
        F_NK: np.ndarray,
        lambdas_K: np.ndarray,
        phi_N: np.ndarray,
        dt: float = 0.05,
        u_max: float = 0.1,
        rho_u: float = 1e-3,
        knn: int = 30,
        alpha: float | None = None,
        # lambda_weight: str = "exp",
        lambda_weight: str = "mezic",
        # lambda_weight: str = "smc",
    ):
        """
        points_N3: surface samples w_i in R^3
        M_NN: mass matrix (sparse CSR/CSC)
        F_NK: eigenvectors, column k is f_k(w_i)
        lambdas_K: eigenvalues corresponding to columns of F
        phi_N: info map sampled at w_i (will be normalized w.r.t. M)
        dt: integration step for x_{t+1} = x_t + dt * u_t
        u_max: speed bound on ||u||
        rho_u: control regularization
        knn: neighbors for local PCA + local linear fit
        alpha: if None uses 1/(t+1), else exponential forgetting with fixed alpha
        lambda_weight: "exp" uses exponential weighting, "mezic" uses (1+sqrt(lam))^-2
        """
        self.P = np.asarray(points_N3, dtype=float)
        self.M = M_NN.tocsr() if sp.issparse(M_NN) else sp.csr_matrix(M_NN)
        self.F = np.asarray(F_NK, dtype=float)
        self.lam = np.asarray(lambdas_K, dtype=float)
        self.N, self.K = self.F.shape
        self.dt = float(dt)
        self.u_max = float(u_max)
        self.rho_u = float(rho_u)
        self.knn = int(knn)
        self.alpha_fixed = alpha
        self.tree = cKDTree(self.P)

        # Normalize phi to integrate to 1 under the mass matrix inner product
        phi = np.asarray(phi_N, dtype=float).reshape(-1)
        Z_phi = float(np.ones(self.N) @ (self.M @ phi))
        if Z_phi <= 0:
            raise ValueError("phi has nonpositive integral under M")
        self.phi = phi / Z_phi

        # Compute phi coefficients: phi_k = ∫ f_k(w) phi(w) dw ≈ phi^T M f_k
        self.phi_k = self.F.T @ (self.M @ self.phi)

        # Weights Lambda_k
        if lambda_weight == "exp":
            # Exponential weighting based on eigenvalues
            self.Lambda = np.exp(-0.1 * np.sqrt(np.maximum(self.lam, 0.0)))
        elif lambda_weight == "mezic":
            self.Lambda = (1.0 + np.sqrt(np.maximum(self.lam, 0.0))) ** (-2)
        elif lambda_weight == "smc":
            self.Lambda = (1.0 + (np.maximum(self.lam, 0.0))) ** (-4 / 2)
        else:
            raise ValueError("lambda_weight must be 'exp' or 'mezic'")

        # Running trajectory coefficients mu_k
        self.mu_k = np.zeros(self.K, dtype=float)
        self.t = 0  # step count

    def ergodic_metric(self) -> float:
        e = self.mu_k - self.phi_k
        return float(np.sum(self.Lambda * e * e))

    def _local_tangent_and_fit(self, x: np.ndarray):
        """
        Returns:
          f_x: (K,) eigenfunctions evaluated at x (via local linear fit)
          grad_fk: (K,3) gradients in R^3 (tangent) for each basis function
          n: (3,) estimated normal
        """
        x = np.asarray(x, dtype=float).reshape(3)

        _, idx = self.tree.query(x, k=self.knn)
        nbrs = self.P[idx]  # (knn,3)
        Y = self.F[idx, :]  # (knn,K)

        # Fit local tangent plane and project agent/neighbors using utility helpers
        coeffs, n, t1, t2 = compute_tangent_space(nbrs)
        x_proj, proj_nbrs, uv_coords = project_points2tangent_space(
            x, nbrs, coeffs, n, t1, t2
        )

        u = uv_coords[:, 0]
        v = uv_coords[:, 1]

        # Local linear model: y ≈ a + b*u + c*v
        A = np.column_stack([np.ones_like(u), u, v])  # (knn,3)
        pinvA = np.linalg.pinv(A)  # (3,knn)
        coeff = pinvA @ Y  # (3,K)

        a = coeff[0, :]  # f_k(x)
        b = coeff[1, :]  # df/du
        c = coeff[2, :]  # df/dv

        # Map gradient back to R^3
        grad = (t1[None, :] * b[:, None]) + (t2[None, :] * c[:, None])  # (K,3)

        # Ensure tangent by removing normal component numerically
        grad = grad - (grad @ n)[:, None] * n[None, :]

        return a, grad, n

    def step(self, x: np.ndarray):
        """
        One control step.
        Returns:
          x_next, u, debug_dict
        """
        x = np.asarray(x, dtype=float).reshape(3)

        # Evaluate basis and gradients at current x
        f_x, grad_fk, n = self._local_tangent_and_fit(x)  # f_x (K,), grad_fk (K,3)

        # Choose alpha
        self.t += 1
        if self.alpha_fixed is None:
            alpha = 1.0 / self.t
        else:
            alpha = float(self.alpha_fixed)

        # Linearized one step objective:
        # e'_k(u) ≈ b_k + a_k^T u
        # where a_k = alpha*dt*grad f_k(x), b_k = (1-alpha)*(mu_k - phi_k) + alpha*(f_k(x) - phi_k)
        e = self.mu_k - self.phi_k
        b_k = (1.0 - alpha) * e + alpha * (f_x - self.phi_k)  # (K,)
        A_k = (alpha * self.dt) * grad_fk  # (K,3)

        # Build 3x3 Hessian and 3x1 gradient
        # H = sum_k Lambda_k * A_k^T A_k + rho_u * I
        # g = sum_k Lambda_k * b_k * A_k
        W = self.Lambda.reshape(-1, 1)  # (K,1)
        H = (A_k.T @ (W * A_k)) + (self.rho_u * np.eye(3))  # (3,3)
        g = A_k.T @ (W[:, 0] * b_k)  # (3,)

        # Solve for unconstrained minimizer u = -H^{-1} g
        u = -np.linalg.solve(H, g)

        # Project to tangent space for safety
        u = u - n * float(np.dot(n, u))

        # Clamp speed
        nu = float(np.linalg.norm(u))
        if nu > self.u_max and nu > 1e-12:
            u = (self.u_max / nu) * u

        # Integrate and re-project to the surface samples by nearest neighbor
        x_raw = x + self.dt * u
        _, j = self.tree.query(x_raw, k=1)
        x_next = self.P[j].copy()

        # Update running coefficients using Dirac sensor: mu_k <- (1-alpha) mu_k + alpha f_k(x_next)
        f_next, _, _ = self._local_tangent_and_fit(x_next)
        self.mu_k = (1.0 - alpha) * self.mu_k + alpha * f_next

        dbg = {
            "alpha": alpha,
            "E": self.ergodic_metric(),
            "mu_k_norm": float(np.linalg.norm(self.mu_k)),
            "u_norm": float(np.linalg.norm(u)),
        }
        return x_next, u, dbg


def compute_laplacian_eigenpairs(points: np.ndarray, num_eigen: int):
    """
    Computes the point cloud Laplacian eigenpairs using robust_laplacian.
    Returns (evals, evecs, mass_matrix).
    """
    C, M = robust_laplacian.point_cloud_laplacian(points)
    evals, evecs = sla.eigsh(C, num_eigen, M, sigma=1e-12)
    order = np.argsort(evals)
    evals = np.maximum(np.real(evals[order]), 0.0)
    evecs = np.real(evecs[:, order])

    # Normalize columns so that f_k^T M f_k = 1 to avoid scaling mismatches.
    Mf = M @ evecs
    norms2 = np.sum(evecs * Mf, axis=0)
    norms = np.sqrt(np.maximum(norms2, 1e-12))
    evecs = evecs / norms[None, :]

    return evals, evecs, M


def normalize_density_from_coeffs(F: np.ndarray, coeffs: np.ndarray, M: sp.spmatrix):
    """
    Reconstructs a density from basis coefficients and normalizes it with the mass matrix.
    """
    density = F @ coeffs
    density = np.maximum(density, 0.0)
    z = float(np.ones(len(density)) @ (M @ density))
    if z > 0:
        density /= z
    return density


