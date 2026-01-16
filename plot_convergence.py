import numpy as np
import matplotlib.pyplot as plt

def normalize_mat(a):
    a = np.array(a, dtype=float)
    # time is last axis
    if a.ndim == 1:
        s = a.sum()
        return a / (s + 1e-12)
    # flatten spatial dims, keep time as last
    spatial_shape = a.shape[:-1]
    sp = a.reshape(-1, a.shape[-1])
    sp_sum = sp.sum(axis=0, keepdims=True)  # sum over spatial for each time
    sp_norm = sp / (sp_sum + 1e-12)
    return sp_norm.reshape(*spatial_shape, a.shape[-1])

data = np.load("hedac_comparison_results.npz", allow_pickle=True)
results = data["results"].item()

sorted_keys = sorted(results.keys(), key=lambda k: float(k))
fs = 100.0  # sampling frequency in Hz (100 Hz)
dt = 1.0 / fs

fig, ax = plt.subplots(figsize=(8, 5))

for k in sorted_keys:
    entry = results[k]

    # find coverage and goal arrays by key name heuristics
    coverage = None
    goal = None
    for name, v in entry.items():
        lname = name.lower()
        if "coverage" in lname and coverage is None:
            coverage = np.array(v)
        if ("goal" in lname or "target" in lname) and goal is None:
            goal = np.array(v)

    # fallback: pick first candidate with time as last axis (dim>1)
    if coverage is None:
        for name, v in entry.items():
            try:
                a = np.array(v)
                if a.ndim >= 2 and a.shape[-1] > 1:
                    coverage = a
                    break
            except Exception:
                continue
    if goal is None:
        for name, v in entry.items():
            try:
                a = np.array(v)
                # goal can be time-varying or static; prefer same spatial size (except time)
                if a.ndim >= 1:
                    goal = a
                    break
            except Exception:
                continue

    if coverage is None or goal is None:
        print(f"Skipping update={k}: missing coverage or goal arrays")
        continue

    # Ensure coverage has runs axis: expected shapes (...spatial..., time) or (runs, ...spatial..., time)
    cov = np.array(coverage, dtype=float)
    # if cov is (n_spatial, time) -> add runs=1
    if cov.ndim == 2:
        # ambiguous: could be (runs, time) or (n_spatial, time)
        # assume spatial x time if first dim > 1 and second dim > 1 -> treat as spatial x time
        cov = cov[np.newaxis, ...]  # becomes (1, n_spatial, time)
    elif cov.ndim >= 3:
        # if shape (runs, n_spatial, time) or (runs, s1, s2, time) keep as is
        pass
    else:
        print(f"Skipping update={k}: unexpected coverage shape {cov.shape}")
        continue

    # goal final distribution: use final time slice as in original code
    g = np.array(goal, dtype=float)
    if g.ndim == 1:
        goal_final = g  # (n_spatial,)
    else:
        # take final time slice
        goal_final = g[..., -1]

    # normalize coverage and goal, flatten spatial dims
    cov_norm = normalize_mat(cov)  # shape (runs, n_spatial, time) after normalize_mat flattening spatial
    # cov_norm may have shape (runs, n_spatial, time) or (1, n_spatial, time)
    runs = cov_norm.shape[0]
    n_spatial = cov_norm.reshape(runs, -1, cov_norm.shape[-1]).shape[1]
    cov_flat = cov_norm.reshape(runs, n_spatial, cov_norm.shape[-1])

    # Normalize goal_final and shape to (runs, n_spatial, 1) for broadcasting
    g_flat = np.array(goal_final, dtype=float).reshape(-1)
    if g_flat.size != n_spatial:
        # try broadcasting spatial dims: if goal has runs axis
        if g_flat.size == runs * n_spatial:
            g_flat = g_flat.reshape(runs, n_spatial)
            g_flat = g_flat[..., np.newaxis]
        else:
            # try repeating goal across runs
            g_flat = np.resize(g_flat, n_spatial)
            g_flat = g_flat[np.newaxis, :, np.newaxis]  # (1, n_spatial, 1)
    else:
        g_flat = g_flat[np.newaxis, :, np.newaxis]  # (1, n_spatial, 1)

    # normalize goal over spatial (ensure sums to 1)
    g_norm = g_flat / (g_flat.sum(axis=1, keepdims=True) + 1e-12)

    # ensure shapes: cov_flat (runs, n_spatial, time), g_norm (runs_or_1, n_spatial, 1)
    if g_norm.shape[0] == 1 and runs > 1:
        g_norm = np.repeat(g_norm, runs, axis=0)

    # compute ergodic metric per run per time: L2 norm over spatial axis
    diff = cov_flat - g_norm  # (runs, n_spatial, time)
    erg_per_run_time = np.linalg.norm(diff, axis=1)  # (runs, time)

    mean = np.mean(erg_per_run_time, axis=0)
    std = np.std(erg_per_run_time, axis=0)
    t = np.arange(mean.size) * dt

    ax.plot(t, mean, label=f"update={k}")
    ax.fill_between(t, mean - std, mean + std, alpha=0.25)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Convergence (ergodic metric)")
ax.set_title("Mean convergence over time (±1 std) — sampling 100 Hz")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("hedac_convergence_time.png", dpi=200)
plt.show()
