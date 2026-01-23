# Configuration System

This project uses a centralized configuration file (`config.py`) to manage directory paths and ensure all scripts save outputs in consistent locations.

## Directory Structure

```
TactileErgodicExploration/
├── config.py                    # Configuration file
├── point_clouds/                # Input point cloud files (.ply)
├── data/                        # General data files
└── results/                     # All output files
    ├── plots/                   # PDF plots and figures
    ├── animations/              # HTML animations
    └── models/                  # Saved model files (.pth)
```

## Usage

### In Your Scripts

Import the config module at the top of your script:

```python
import config
```

### Getting File Paths

Use the helper functions to get proper paths:

```python
# Point cloud files
filename = config.get_point_cloud_path("bunny.ply")

# Output plots
plot_path = config.get_plot_path("my_plot.pdf")

# Animations
anim_path = config.get_animation_path("trajectory.html")

# Model files
model_path = config.get_model_path("model_state.pth")

# General data
data_path = config.get_data_path("measurements.npz")
```

### Direct Path Access

You can also access directory paths directly:

```python
config.POINT_CLOUD_DIR  # Point cloud directory
config.RESULTS_DIR      # Results directory
config.PLOTS_DIR        # Plots subdirectory
config.ANIMATIONS_DIR   # Animations subdirectory
config.MODELS_DIR       # Models subdirectory
config.DATA_DIR         # Data directory
```

## Benefits

1. **Centralized Management**: Change directory structure in one place
2. **Automatic Creation**: Directories are created automatically when config is imported
3. **Consistent Output**: All scripts save outputs to the same locations
4. **Git-Friendly**: Results are automatically gitignored while source code is tracked
5. **Path Safety**: Uses `pathlib.Path` for cross-platform compatibility

## Migrating Existing Scripts

Replace hardcoded paths:

```python
# Before
filename = "point_clouds/bunny.ply"
plot_name = "my_plot.pdf"

# After
import config
filename = config.get_point_cloud_path("bunny.ply")
plot_name = str(config.get_plot_path("my_plot.pdf"))
```

Note: Convert Path objects to strings when needed by libraries that don't accept Path objects:
```python
save_path = str(config.get_plot_path("figure.pdf"))
```
