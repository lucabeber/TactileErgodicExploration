# Ergodic Exploration Methods Comparison

This repository now contains three different ergodic exploration methods, all integrated with the same GPR-based active learning framework for fair comparison.

## Methods Overview

### 1. HEDAC (Heat Equation Driven Area Coverage)
**File**: `ergodic_exploration.py`

**Approach**:
- Works directly on 3D point cloud surface
- Uses heat diffusion equation to create potential field
- Local gradient descent following heat flow
- Explicit velocity/acceleration constraints

**Key Parameters**:
- Total timesteps: 1500
- GP update frequency: every 50 timesteps
- Max velocity: 0.0004 m/s (0.1 × voxel_size × 2)
- Max acceleration: 0.0008 m/s²
- Exploration-exploitation α: 0.6

**Characteristics**:
- Local surface-aware navigation
- Guaranteed surface contact at each step
- Computationally efficient per step
- Natural handling of surface topology

---

### 2. Sinkhorn Flow Matching
**File**: `sinkhorn_exploration.py`

**Approach**:
- Works in 3D Euclidean space
- Uses optimal transport (Sinkhorn divergence)
- LQR-based trajectory optimization
- MPC-style execution (plan long, execute short)
- Projects back to surface after each chunk

**Key Parameters**:
- Total timesteps: 300 (for debugging, normally 1500)
- GP update frequency: every 50 timesteps
- Planning horizon: 50 steps
- Execution chunk: 10 steps
- Optimization iterations: 50 per chunk
- Exploration-exploitation α: 0.6

**Characteristics**:
- Global trajectory optimization
- Naturally handles 3D obstacles
- Higher computational cost per iteration
- Smooth, optimal transport-based trajectories

---

### 3. SMC (Sliding Mode Control) - Fourier Based
**File**: `smc_exploration.py`

**Approach**:
- Works in 2D UV parameterized space (PCA projection)
- Fourier series representation of target distribution
- Spectral ergodic control
- Projects UV trajectory back to 3D surface

**Key Parameters**:
- Total timesteps: 1500 (matched to HEDAC)
- GP update frequency: every 50 timesteps
- Max velocity (UV space): 0.002 units/s (scaled from HEDAC)
- HEDAC equivalent velocity: 0.0004 m/s
- Fourier basis functions: 8 × 8
- Exploration-exploitation α: 0.6

**Characteristics**:
- Dimension reduction (3D → 2D)
- Spectral representation of distribution
- Computationally efficient in frequency domain
- Requires UV parameterization

---

## Common Framework

All three methods share:

1. **GPR Integration**:
   - Online Gaussian Process Regression
   - Updates target density every 50 timesteps
   - Same kernel: RBF manifold kernel (l=0.010, σ=1.0, n_eig=500)

2. **Exploration-Exploitation**:
   - α = 0.6 (60% mean, 40% variance)
   - `goal_density = α × normalize(max(mean - mean_avg, 0)) + (1-α) × normalize(variance)`

3. **Visualization**:
   - `plot_distribution_evolution_column_auto()` - PDF showing 6 snapshots
   - `animate_trajectory_pcloud()` - Interactive HTML animation
   - `visualize_trajectory()` - 3D trajectory plot

4. **Outputs**:
   - PDF: Distribution evolution over time
   - HTML: Animated trajectory with density evolution
   - NPZ: All data (trajectory, densities, vertices)

---

## Performance Comparison

| Method | Computation Time | Memory | Trajectory Smoothness | Surface Fidelity |
|--------|-----------------|---------|----------------------|------------------|
| HEDAC | Fast (~0.001s/step) | Low | Moderate | Excellent |
| Sinkhorn | Slow (~0.8s/iter) | High | Very High | Good |
| SMC | Fast (~0.01s/step) | Low | High | Good |

*Note: Timings are approximate and depend on hardware*

---

## Usage

All scripts follow the same pattern:

```bash
# Run with default parameters
python ergodic_exploration.py   # HEDAC
python sinkhorn_exploration.py  # Sinkhorn
python smc_exploration.py       # SMC
```

### Changing Object

In each script's `main()` function:
```python
# object_name = "bun270_X"      # Stanford bunny
object_name = "plate_shapes"     # IKEA plate
# object_name = "cup_X"          # Cup
```

### Adjusting Parameters

**HEDAC** (ergodic_exploration.py:263):
```python
param.timesteps = 1500
param.max_velocity = 0.1 * param.voxel_size * 2
```

**Sinkhorn** (sinkhorn_exploration.py:326):
```python
total_timesteps = 300  # Increase to 1500 for full run
param.num_opt_iters = 50
param.chunk_len = 10
```

**SMC** (smc_exploration.py:342):
```python
param.timesteps = 1500
param.u_max = hedac_max_velocity * (1.0 / typical_3d_extent)
param.nbFct = 8  # Fourier basis functions
```

---

## Evaluation Metrics

All methods can be compared using:

1. **Coverage Completeness**: How well the trajectory covers the target distribution
2. **Ergodic Metric**: L2 distance between trajectory statistics and target distribution
3. **Sample Efficiency**: How quickly the GP converges to the true distribution
4. **Computational Cost**: Total time to complete exploration
5. **Trajectory Length**: Total distance traveled

---

## Citation

If you use this code, please cite the original HEDAC paper:

```
Ivić, S., Crnković, B., & Mezić, I. (2017). Ergodic-Based Cooperative
Multiagent Area Coverage via a Potential Field. IEEE Transactions on
Cybernetics, 47(8), 1983–1993. https://doi.org/10.1109/TCYB.2016.2634400
```

For Sinkhorn flow matching, cite:
```
[Relevant optimal transport and flow matching papers]
```

For SMC ergodic control, cite:
```
[Relevant spectral ergodic control papers]
```

---

## Notes

- **Speed Matching**: SMC velocity is scaled to match HEDAC's 3D velocity when projected
- **Fair Comparison**: All methods use identical GP settings, update frequencies, and α values
- **Reproducibility**: Set random seed for consistent results
- **Point Clouds**: Compatible with .ply files containing RGB color data as target distribution
