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


def run_surface_exploration(
    # ply_path: str = "plate_shapes.ply",
    ply_path: str = "bun270_X.ply",
    num_eigen: int = 200,
    n_steps: int = 1200,
    seed: int = 0,
    dt: float = 0.05,
    u_max: float = 0.1,
    rho_u: float = 1e-3,
    knn: int = 30,
    voxel_size: float = 0.002,
):
    """
    Loads the plate point cloud, computes Laplacian eigenpairs, and runs
    the Laplacian SMC ergodic controller.
    """
    rng = np.random.default_rng(seed)

    # Create parameter object for process_point_cloud
    class Param:
        def __init__(self):
            self.voxel_size = voxel_size
            self.alpha = 1  # scaling factor for dt calculation

    param = Param()
    pcloud = process_point_cloud(ply_path, param)
    points = pcloud.vertices

    # Target distribution: red channel (already normalized in process_point_cloud)
    phi = pcloud.u0.astype(float)
    max_phi = float(np.max(phi))
    if max_phi > 0:
        phi /= max_phi
    else:
        raise ValueError("Red channel is all zeros; cannot build a target density.")

    lambdas, eigenvecs, M = compute_laplacian_eigenpairs(points, num_eigen)

    controller = LaplacianSMCController(
        points_N3=points,
        M_NN=M,
        F_NK=eigenvecs,
        lambdas_K=lambdas,
        phi_N=phi,
        dt=dt,
        u_max=u_max,
        rho_u=rho_u,
        knn=knn,
    )

    # Run control
    traj = np.zeros((n_steps + 1, 3), dtype=float)
    traj[0] = points[rng.integers(len(points))]

    est_density = np.zeros((len(points), n_steps + 1), dtype=float)
    est_density[:, 0] = normalize_density_from_coeffs(controller.F, controller.mu_k, M)

    # Track coefficient evolution
    mu_k_history = np.zeros((controller.K, n_steps + 1), dtype=float)
    mu_k_history[:, 0] = controller.mu_k.copy()

    debug = []

    for t in range(1, n_steps + 1):
        x_next, u, dbg = controller.step(traj[t - 1])
        traj[t] = x_next
        est_density[:, t] = normalize_density_from_coeffs(
            controller.F, controller.mu_k, M
        )
        mu_k_history[:, t] = controller.mu_k.copy()
        debug.append(dbg)

    # Color frames use the fixed target distribution (red channel)
    color_frames = np.repeat(phi[:, None], n_steps + 1, axis=1)

    return {
        "points": points,
        "colors": phi,  # Use red channel (u0) as colors
        "phi": phi,
        "phi_k": controller.phi_k,
        "mu_k_history": mu_k_history,
        "trajectory": traj,
        "estimated_density": est_density,
        "color_frames": color_frames,
        "debug": debug,
        "num_eigen": num_eigen,
    }


def animate_with_coefficients(
    results, save_path="plate_exploration.html", is_show=True
):
    """
    Creates an animation with 3 panels:
    - 3D trajectory on point cloud
    - Desired Laplacian eigenfunction coefficients (phi_k)
    - Reproduced coefficients evolution (mu_k over time)
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    traj = results["trajectory"]
    points = results["points"]
    color_frames = results["color_frames"]
    phi_k = results["phi_k"]
    mu_k_history = results["mu_k_history"]
    num_eigen = results["num_eigen"]

    # Animation parameters
    point_size = 2
    timestep_multiplier = 10
    n_steps = traj.shape[0]
    n_frames = (n_steps + timestep_multiplier - 1) // timestep_multiplier

    # Reshape coefficients for visualization (use square root for approximate square)
    nrows = int(np.sqrt(num_eigen))
    ncols = (num_eigen + nrows - 1) // nrows
    phi_k_img = np.zeros((nrows, ncols))
    phi_k_img.flat[:num_eigen] = phi_k

    # Camera settings
    camera_params = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0.0, y=0.0, z=1.2),
    )

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=3,
        column_widths=[0.5, 0.25, 0.25],
        subplot_titles=("3D Trajectory", "Desired φ_k", "Reproduced μ_k(t)"),
        specs=[[{"type": "scatter3d"}, {"type": "heatmap"}, {"type": "heatmap"}]],
    )

    # Initial point cloud
    point_cloud = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(
            size=point_size,
            opacity=0.5,
            color=color_frames[:, 0],
            # colorscale="bluered",
            colorscale="viridis",
        ),
        showlegend=False,
    )

    # Initial trajectory
    trajectory_trace = go.Scatter3d(
        x=[traj[0, 0]],
        y=[traj[0, 1]],
        z=[traj[0, 2]],
        mode="lines",
        line=dict(width=5, color="red"),
        opacity=0.4,
        showlegend=False,
    )

    # Desired coefficients (static)
    desired_coeffs = go.Heatmap(
        z=phi_k_img,
        colorscale="RdBu",
        showscale=False,
        zmid=0,
    )

    # Initial reproduced coefficients
    mu_k_img_0 = np.zeros((nrows, ncols))
    mu_k_img_0.flat[:num_eigen] = mu_k_history[:, 0]
    reproduced_coeffs = go.Heatmap(
        z=mu_k_img_0,
        colorscale="RdBu",
        showscale=False,
        zmid=0,
    )

    # Add traces
    fig.add_trace(point_cloud, row=1, col=1)
    fig.add_trace(trajectory_trace, row=1, col=1)
    fig.add_trace(desired_coeffs, row=1, col=2)
    fig.add_trace(reproduced_coeffs, row=1, col=3)

    # Update 3D scene
    fig.update_scenes(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="data",
        camera=camera_params,
    )

    # Remove tick labels from heatmaps
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_xaxes(showticklabels=False, row=1, col=3)
    fig.update_yaxes(showticklabels=False, row=1, col=3)

    # Create animation frames
    frames = []
    for k in range(n_frames):
        frame_idx = min(k * timestep_multiplier, n_steps - 1)

        # Point cloud
        pc_frame = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(
                size=point_size,
                opacity=0.5,
                color=color_frames[:, frame_idx],
                colorscale="viridis",
            ),
            showlegend=False,
        )

        # Trajectory so far
        traj_frame = go.Scatter3d(
            x=traj[:frame_idx, 0],
            y=traj[:frame_idx, 1],
            z=traj[:frame_idx, 2],
            mode="lines",
            line=dict(width=5, color="red"),
            opacity=1.0,
            showlegend=False,
        )

        # Desired coefficients (unchanged)
        desired_frame = go.Heatmap(
            z=phi_k_img,
            colorscale="RdBu",
            showscale=False,
            zmid=0,
        )

        # Reproduced coefficients at current timestep
        mu_k_img_t = np.zeros((nrows, ncols))
        mu_k_img_t.flat[:num_eigen] = mu_k_history[:, frame_idx]
        reproduced_frame = go.Heatmap(
            z=mu_k_img_t,
            colorscale="RdBu",
            showscale=False,
            zmid=0,
        )

        frames.append(
            go.Frame(
                data=[pc_frame, traj_frame, desired_frame, reproduced_frame],
                name=f"frame{k}",
                traces=[0, 1, 2, 3],
            )
        )

    fig.update(frames=frames)

    # Add slider
    sliders = [
        dict(
            steps=[
                dict(
                    method="animate",
                    args=[
                        [f"frame{k}"],
                        dict(
                            mode="immediate",
                            frame=dict(duration=400, redraw=True),
                            transition=dict(duration=0),
                        ),
                    ],
                    label=f"{k+1}",
                )
                for k in range(n_frames)
            ],
            active=0,
            transition=dict(duration=0),
            x=0,
            y=0,
            currentvalue=dict(
                font=dict(size=12), prefix="frame: ", visible=True, xanchor="center"
            ),
            len=1.0,
        )
    ]

    fig.update_layout(
        width=1800,
        height=600,
        sliders=sliders,
        showlegend=False,
    )

    # Save and/or show
    if save_path:
        fig.write_html(save_path)
        print(f"Animation saved to '{save_path}'")

    if is_show:
        fig.show("browser")

    return fig


def main():
    results = run_surface_exploration()
    animate_with_coefficients(
        results,
        save_path="plate_exploration.html",
        is_show=True,
    )
    print(
        f"Finished {results['trajectory'].shape[0]-1} steps. "
        f"Final ergodic metric: {results['debug'][-1]['E']:.4f}"
    )


if __name__ == "__main__":
    main()
