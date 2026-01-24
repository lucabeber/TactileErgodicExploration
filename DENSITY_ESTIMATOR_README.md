# Density Estimator Module

## Overview

The `density_estimator.py` module provides a reusable framework for GP-based density estimation that can be shared across different exploration methods. This allows for consistent comparison between different exploration strategies while using the same density estimation approach.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Exploration Methods                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  HEDAC   │  │   SMC    │  │    BO    │  │  Custom  │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
└───────┼─────────────┼─────────────┼─────────────┼─────────┘
        │             │             │             │
        └─────────────┴─────────────┴─────────────┘
                          │
        ┌─────────────────▼──────────────────┐
        │    density_estimator.py            │
        │  ┌──────────────────────────────┐  │
        │  │   DensityEstimator           │  │
        │  │  - Ground truth GP model     │  │
        │  │  - Sample at exploration pts │  │
        │  └──────────────────────────────┘  │
        │  ┌──────────────────────────────┐  │
        │  │   OnlineGPEstimator          │  │
        │  │  - Updates with trajectory   │  │
        │  │  - Predicts on full cloud    │  │
        │  │  - Computes goal density     │  │
        │  └──────────────────────────────┘  │
        └─────────────────────────────────────┘
                          │
        ┌─────────────────▼──────────────────┐
        │   gpr_on_point_cloud.py            │
        │  - Manifold kernel                 │
        │  - GPROnPointCloud model           │
        └─────────────────────────────────────┘
```

## Components

### `DensityEstimator`

The main interface for density estimation. It:
- Initializes a "ground truth" GP model based on the target density
- Provides sampling of density at exploration points
- Creates `OnlineGPEstimator` instances for online learning

**Key Methods:**
- `sample_density_at_point(point)`: Get density value at a 3D point
- `create_online_estimator(trajectory, samples)`: Create online updatable estimator
- `get_initial_mean()`: Get ground truth mean prediction

### `OnlineGPEstimator`

Maintains an evolving estimate of density during exploration. It:
- Updates incrementally as new trajectory samples are collected
- Predicts density over the entire point cloud
- Computes goal densities balancing exploration vs exploitation

**Key Methods:**
- `update(new_trajectory, new_samples)`: Add new samples to the model
- `predict(test_points, return_variance)`: Predict at given points
- `get_goal_density(exploit_alpha, nb_boundary_neighbors)`: Compute exploration goal

## Usage Examples

### Basic Usage (Any Exploration Method)

```python
from density_estimator import create_density_estimator
import torch

# Create density estimator
gp_params = {'l': 0.010, 'sigma': 1.0, 'n_eig': 500}
density_estimator = create_density_estimator(
    pcloud,
    target_density=u0,
    gp_params=gp_params,
    dtype=torch.float32
)

# Sample at agent's initial position
initial_sample = density_estimator.sample_density_at_point(agent.x)

# Create online estimator
online_estimator = density_estimator.create_online_estimator(
    initial_trajectory=agent.x.reshape(1, -1),
    initial_samples=np.array([initial_sample])
)

# During exploration loop:
for t in range(timesteps):
    # ... agent moves ...

    # Periodically update the estimator
    if t % 50 == 0:
        # Sample along recent trajectory
        trajectory_samples = agent.x_arr[:t:5, :]
        density_samples = np.array([
            density_estimator.sample_density_at_point(pt)
            for pt in trajectory_samples
        ])

        # Update online estimator
        online_estimator.update(trajectory_samples, density_samples)

        # Get updated goal density
        goal_density = online_estimator.get_goal_density(
            exploit_alpha=0.6,  # Balance exploration/exploitation
            nb_boundary_neighbors=40
        )

        # Get current estimate
        mean_estimate = online_estimator.predict(return_variance=False)
```

### HEDAC with Density Estimator

See `hedac_exploration_refactored.py` for a complete example of using the density estimator with HEDAC exploration.

### Custom Exploration Method

```python
def my_exploration_method(agent, param, pcloud, density_estimator):
    # Initialize online estimator
    initial_sample = density_estimator.sample_density_at_point(agent.x)
    online_estimator = density_estimator.create_online_estimator(
        initial_trajectory=agent.x.reshape(1, -1),
        initial_samples=np.array([initial_sample])
    )

    for t in range(param.timesteps):
        # Your custom exploration logic
        # ...

        # Update density estimate periodically
        if t % 50 == 0:
            trajectory = agent.x_arr[:t:5, :]
            samples = [density_estimator.sample_density_at_point(p) for p in trajectory]
            online_estimator.update(trajectory, samples)

            # Use the estimate to guide exploration
            goal = online_estimator.get_goal_density(param.exploit_alpha)
            # ...

    return results
```

## Parameters

### GP Hyperparameters

```python
gp_params = {
    'l': 0.010,        # Length scale for RBF kernel
    'sigma': 1.0,      # Signal variance
    'n_eig': 500       # Number of eigenfunctions for manifold kernel
}
```

### Exploration/Exploitation Balance

The `exploit_alpha` parameter in `get_goal_density()` controls the trade-off:
- `exploit_alpha = 0.0`: Pure exploration (maximize variance)
- `exploit_alpha = 1.0`: Pure exploitation (maximize mean)
- `exploit_alpha = 0.6`: Balanced (recommended)

## File Organization

```
exploration_methods/
├── hedac_exploration.py              # Original HEDAC (for reference)
├── hedac_exploration_refactored.py   # HEDAC using density_estimator
├── smc_exploration.py                # SMC using density_estimator
├── bo_exploration.py                 # BO using density_estimator
└── ...

core/
├── density_estimator.py              # Reusable density estimation
├── gpr_on_point_cloud.py            # GP on manifolds
├── pointcloud_utils.py              # Utility functions
└── config.py                         # Configuration

comparison/
├── compare_methods.py                # Compare different methods
└── COMPARISON.md                     # Results and analysis
```

## Benefits of This Structure

1. **Consistency**: All methods use the same GP estimation approach
2. **Modularity**: Easy to swap out exploration strategies
3. **Comparability**: Fair comparison since density estimation is identical
4. **Reusability**: Write once, use in multiple methods
5. **Maintainability**: GP code in one place, easier to debug and improve
6. **Extensibility**: Easy to add new exploration methods

## Migrating Existing Code

To migrate an existing exploration method:

1. **Import the module:**
   ```python
   from density_estimator import create_density_estimator
   ```

2. **Create the estimator:**
   ```python
   density_estimator = create_density_estimator(pcloud, u0, gp_params)
   ```

3. **Replace direct GP code:**
   - Replace model initialization with `create_online_estimator()`
   - Replace manual sampling with `sample_density_at_point()`
   - Replace manual updates with `online_estimator.update()`
   - Replace prediction code with `online_estimator.predict()`

4. **Keep your exploration logic:**
   - All exploration-specific code (gradient computation, coverage, etc.) remains unchanged
   - Only the GP estimation parts are replaced

## Testing

Run the refactored HEDAC to verify the module works correctly:

```bash
python hedac_exploration_refactored.py
```

This should produce similar results to the original `hedac_exploration.py`.
