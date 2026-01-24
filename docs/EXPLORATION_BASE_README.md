# Exploration Base Class Architecture

## Overview

The `exploration_base.py` module provides a common base class `ExplorationWithGP` that handles shared functionality across different exploration algorithms. This eliminates code duplication and ensures consistent GP-based density estimation across methods.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              ExplorationWithGP (Base Class)             │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Common Functionality:                             │ │
│  │  - setup_point_cloud()                             │ │
│  │  - setup_density_estimator()                       │ │
│  │  - voxel_size, exploit_alpha, gp_update_interval   │ │
│  │                                                     │ │
│  │  Hook for subclasses:                              │ │
│  │  - _setup_point_cloud_hook()                       │ │
│  │                                                     │ │
│  │  Abstract method:                                  │ │
│  │  - explore()                                       │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                        ▲           ▲
                        │           │
          ┌─────────────┘           └────────────┐
          │                                      │
┌─────────┴─────────┐                  ┌─────────┴─────────┐
│       HEDAC       │                  │   SMCExplorerGP   │
│  ┌──────────────┐ │                  │  ┌──────────────┐ │
│  │ _setup_      │ │                  │  │ _setup_      │ │
│  │ point_cloud_ │ │                  │  │ point_cloud_ │ │
│  │ hook():      │ │                  │  │ hook():      │ │
│  │ - Laplacian  │ │                  │  │ - UV params  │ │
│  │ - Heat eqn   │ │                  │  │ - Fourier    │ │
│  │              │ │                  │  │   basis      │ │
│  │ explore():   │ │                  │  │              │ │
│  │ - HEDAC algo │ │                  │  │ explore():   │ │
│  └──────────────┘ │                  │  │ - SMC algo   │ │
└───────────────────┘                  │  └──────────────┘ │
                                       └───────────────────┘
```

## Base Class: `ExplorationWithGP`

### Constructor Parameters

```python
def __init__(
    self,
    voxel_size=0.002,        # Voxel size for point cloud downsampling
    exploit_alpha=0.6,        # Balance exploration vs exploitation
    gp_update_interval=50,    # GP update frequency (timesteps)
)
```

### Common Methods

#### `setup_point_cloud(filename, additional_params=None)`
Loads and processes the point cloud. Calls `_setup_point_cloud_hook()` for method-specific setup.

**Parameters:**
- `filename`: Path to point cloud file (.ply)
- `additional_params`: Dictionary with method-specific parameters (e.g., `{"alpha": 100}`)

**Example:**
```python
explorer.setup_point_cloud(
    "data/bunny.ply",
    additional_params={"alpha": 100}
)
```

#### `setup_density_estimator(gp_params=None)`
Sets up GP-based density estimation using the `density_estimator` module.

**Parameters:**
- `gp_params`: Dictionary with GP hyperparameters
  - `l`: Length scale (default: 0.010)
  - `sigma`: Signal variance (default: 1.0)
  - `n_eig`: Number of eigenfunctions (default: 500)

**Example:**
```python
explorer.setup_density_estimator({
    "l": 0.010,
    "sigma": 1.0,
    "n_eig": 500
})
```

### Hook Method (Override in Subclasses)

#### `_setup_point_cloud_hook()`
Called automatically after point cloud processing. Override this in subclasses to add method-specific setup.

**HEDAC example:**
```python
def _setup_point_cloud_hook(self):
    self.pcd_helper = Pointcloud(self.pcloud.vertices)
    self.boundary_normals = self.pcd_helper.get_boundary_normals()
    self.scalar_diffusion_solver = PointcloudScalarDiffusion(...)
    # Setup Laplacian matrices...
```

**SMC example:**
```python
def _setup_point_cloud_hook(self):
    self.uv_coords, self.points_3d = compute_uv_parameterization_pca(...)
    self.linear_interp, self.nearest_interp = create_uv_interpolator(...)
    self.fourier_basis = setup_fourier_basis(...)
```

### Abstract Method (Must Implement)

#### `explore(*args, **kwargs)`
Implements the specific exploration algorithm. Must be overridden in subclasses.

## Usage Examples

### HEDAC with Base Class

```python
from exploration_base import ExplorationWithGP
import robust_laplacian
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from pointcloud import Pointcloud
from pointcloud_scalar_diffusion import PointcloudScalarDiffusion

class HEDAC(ExplorationWithGP):
    def __init__(
        self,
        exploit_alpha=0.6,
        timesteps=1500,
        alpha=100,
        voxel_size=0.002,
        **kwargs
    ):
        # Initialize base class
        super().__init__(
            voxel_size=voxel_size,
            exploit_alpha=exploit_alpha,
            **kwargs
        )

        # HEDAC-specific parameters
        self.timesteps = timesteps
        self.alpha = alpha
        # ...

    def _setup_point_cloud_hook(self):
        """HEDAC-specific setup"""
        self.pcd_helper = Pointcloud(self.pcloud.vertices)
        self.boundary_normals = self.pcd_helper.get_boundary_normals()
        self.scalar_diffusion_solver = PointcloudScalarDiffusion(...)
        # Setup Laplacian matrices
        self.pcloud.C, self.pcloud.M = robust_laplacian.point_cloud_laplacian(...)
        A = csc_matrix(self.pcloud.M + self.pcloud.dt * self.pcloud.C)
        self.pcloud.A_factorized = splu(A)

    def explore(self, agent):
        """HEDAC exploration algorithm"""
        # ... HEDAC-specific exploration logic
        return results

# Usage
hedac = HEDAC(timesteps=1500, alpha=100)
hedac.setup_point_cloud("bunny.ply", additional_params={"alpha": hedac.alpha})
hedac.setup_density_estimator({"l": 0.010, "sigma": 1.0, "n_eig": 500})
results = hedac.explore(agent)
```

### SMC with Base Class

```python
from exploration_base import ExplorationWithGP
from pcloud_uv_smc_common import (
    compute_uv_parameterization_pca,
    create_uv_interpolator,
    setup_fourier_basis,
)

class SMCExplorerGP(ExplorationWithGP):
    def __init__(
        self,
        nbData=500,
        nbFct=50,
        voxel_size=0.002,
        exploit_alpha=0.6,
        **kwargs
    ):
        # Initialize base class
        super().__init__(
            voxel_size=voxel_size,
            exploit_alpha=exploit_alpha,
            **kwargs
        )

        # SMC-specific parameters
        self.nbData = nbData
        self.nbFct = nbFct
        # ...

    def _setup_point_cloud_hook(self):
        """SMC-specific setup"""
        self.uv_coords, self.points_3d = compute_uv_parameterization_pca(
            self.pcloud.vertices
        )
        self.linear_interp, self.nearest_interp = create_uv_interpolator(
            self.uv_coords, self.points_3d
        )
        self.fourier_basis = setup_fourier_basis(...)

    def explore(self, x0_uv=None):
        """SMC exploration algorithm"""
        # ... SMC-specific exploration logic
        return results

# Usage
smc = SMCExplorerGP(nbData=500, nbFct=50)
smc.setup_point_cloud("bunny.ply", additional_params={"alpha": 100})
smc.setup_density_estimator({"l": 0.010, "sigma": 1.0, "n_eig": 500})
results = smc.explore(x0_uv=[0.4, 0.6])
```

## Benefits

1. **Code Reuse**: Common functionality (point cloud loading, GP setup) implemented once
2. **Consistency**: All exploration methods use identical GP estimation
3. **Maintainability**: Changes to GP setup propagate to all methods
4. **Extensibility**: Easy to add new exploration methods
5. **Clean API**: Consistent interface across all exploration algorithms

## Adding a New Exploration Method

1. **Inherit from `ExplorationWithGP`**:
   ```python
   from exploration_base import ExplorationWithGP

   class MyExplorer(ExplorationWithGP):
       pass
   ```

2. **Implement `__init__`**:
   ```python
   def __init__(self, voxel_size=0.002, exploit_alpha=0.6, **kwargs):
       super().__init__(voxel_size, exploit_alpha, **kwargs)
       # Add your method-specific parameters
       self.my_param = ...
   ```

3. **Override `_setup_point_cloud_hook()` (if needed)**:
   ```python
   def _setup_point_cloud_hook(self):
       # Add method-specific point cloud setup
       self.my_data_structure = ...
   ```

4. **Implement `explore()`**:
   ```python
   def explore(self, *args, **kwargs):
       # Your exploration algorithm
       return results
   ```

5. **Use it**:
   ```python
   explorer = MyExplorer()
   explorer.setup_point_cloud("bunny.ply", additional_params={...})
   explorer.setup_density_estimator({...})
   results = explorer.explore()
   ```

## Files

- `exploration_base.py`: Base class definition
- `hedac_exploration_refactored.py`: HEDAC implementation using base class
- `smc_exploration.py`: SMC implementation using base class
- `density_estimator.py`: Shared GP density estimation module

## See Also

- [DENSITY_ESTIMATOR_README.md](DENSITY_ESTIMATOR_README.md): Details on the density estimation module
- [COMPARISON.md](COMPARISON.md): Comparison of different exploration methods
