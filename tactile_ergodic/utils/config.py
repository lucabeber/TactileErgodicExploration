"""
Configuration file for TactileErgodicExploration project.

Defines directory paths and common settings used across all scripts.
"""
import os
from pathlib import Path

# Base directory (project root)
# Go up two levels: config.py -> tactile_ergodic/utils -> tactile_ergodic -> repository root
BASE_DIR = Path(__file__).parent.parent.parent

# Data directories
POINT_CLOUD_DIR = BASE_DIR / "point_clouds"
DATA_DIR = BASE_DIR / "data"

# Output directories
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
ANIMATIONS_DIR = RESULTS_DIR / "animations"
MODELS_DIR = RESULTS_DIR / "models"

# Create directories if they don't exist
def setup_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        POINT_CLOUD_DIR,
        DATA_DIR,
        RESULTS_DIR,
        PLOTS_DIR,
        ANIMATIONS_DIR,
        MODELS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Directories initialized:")
    print(f"  Point clouds: {POINT_CLOUD_DIR}")
    print(f"  Data:         {DATA_DIR}")
    print(f"  Results:      {RESULTS_DIR}")
    print(f"  Plots:        {PLOTS_DIR}")
    print(f"  Animations:   {ANIMATIONS_DIR}")
    print(f"  Models:       {MODELS_DIR}")


# Helper functions for path construction
def get_point_cloud_path(filename):
    """Get full path for a point cloud file."""
    return POINT_CLOUD_DIR / filename


def get_plot_path(filename):
    """Get full path for a plot file."""
    return PLOTS_DIR / filename


def get_animation_path(filename):
    """Get full path for an animation file."""
    return ANIMATIONS_DIR / filename


def get_model_path(filename):
    """Get full path for a model file."""
    return MODELS_DIR / filename


def get_data_path(filename):
    """Get full path for a data file."""
    return DATA_DIR / filename


# Initialize directories on import
setup_directories()
