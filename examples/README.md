# Tactile Ergodic Exploration

A library for ergodic exploration on point clouds with online Gaussian Process density estimation, designed for tactile sensing applications.

## Repository Structure

```
TactileErgodicExploration/
├── tactile_ergodic/          # Core library
│   ├── utils/                # Utility modules
│   │   ├── config.py         # Configuration and path management
│   │   ├── pointcloud.py     # Point cloud data structures
│   │   ├── pointcloud_utils.py       # Point cloud processing utilities
│   │   ├── pointcloud_scalar_diffusion.py  # Scalar diffusion on point clouds
│   │   ├── plotting_utils.py         # Visualization utilities
│   │   └── virtual_agents.py         # Agent dynamics (FirstOrder, SecondOrder)
│   │
│   ├── estimation/           # Density estimation
│   │   ├── density_estimator.py      # GP-based density estimation
│   │   └── gpr_on_point_cloud.py     # GP regression on manifolds
│   │
│   └── exploration/          # Exploration algorithms
│       ├── exploration_base.py       # Base class for all exploration methods
│       ├── ergodic_control_uv.py     # UV-parameterized ergodic control
│       └── laplacian_smc.py          # Laplacian eigenfunction-based SMC
│
├── examples/                 # Example scripts
│   ├── hedac_exploration_refactored.py   # Heat equation-driven coverage
│   ├── uv_smc_exploration.py             # UV-space SMC exploration
│   ├── laplacian_smc_exploration.py      # Laplacian SMC exploration
│   ├── BO_optimization_EI.py             # Bayesian optimization with EI
│   └── BO_optimization_UCB.py            # Bayesian optimization with UCB
│
├── docs/                     # Documentation
│   ├── EXPLORATION_BASE_README.md    # Base class architecture
│   ├── DENSITY_ESTIMATOR_README.md   # Density estimation module
│   └── COMPARISON.md                 # Comparison of methods
│
├── point_clouds/             # Point cloud data
├── data/                     # Additional data
└── results/                  # Output directory
    ├── plots/
    ├── animations/
    └── models/
```

## Installation

1. Clone the repository:
\`\`\`bash
git clone <repository-url>
cd TactileErgodicExploration
\`\`\`

2. Create a conda environment (recommended):
\`\`\`bash
conda create -n tactile_ergodic python=3.13
conda activate tactile_ergodic
\`\`\`

3. Install dependencies:
\`\`\`bash
pip install numpy scipy matplotlib plotly open3d robust-laplacian gpytorch torch scikit-learn
\`\`\`

## Running Examples

From the repository root:

\`\`\`bash
# Run HEDAC exploration
cd examples
python hedac_exploration_refactored.py

# Run UV-SMC exploration
python uv_smc_exploration.py

# Run Laplacian SMC exploration
python laplacian_smc_exploration.py
\`\`\`

## Exploration Methods

The library provides three main exploration approaches:

1. **HEDAC** - Heat Equation Driven Autonomous Coverage
   - Uses heat equation dynamics on the surface
   - Balances exploration vs exploitation with coverage metric

2. **UV-SMC** - UV-parameterized Sliding Mode Control
   - Projects surface to 2D UV space
   - Uses Fourier basis for ergodic metric

3. **Laplacian SMC** - Eigenfunction-based Sliding Mode Control
   - Operates directly on 3D surface using Laplacian eigenfunctions
   - No UV parameterization needed

All methods use online GP density estimation to adaptively learn target distributions.

## Documentation

- [Exploration Base Class](docs/EXPLORATION_BASE_README.md) - Architecture and API
- [Density Estimator](docs/DENSITY_ESTIMATOR_README.md) - GP-based estimation details
- [Method Comparison](docs/COMPARISON.md) - Performance comparison

## Citation

Copyright (c) 2024 Idiap Research Institute  
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>

## License

GNU General Public License v3.0
