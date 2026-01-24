from setuptools import setup, find_packages

setup(
    name="tactile_ergodic",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "plotly",
        "open3d",
        "robust-laplacian",
        "gpytorch",
        "torch",
        "scikit-learn",
    ],
    author="Cem Bilaloglu",
    author_email="cem.bilaloglu@idiap.ch",
    description="Tactile Ergodic Exploration Library",
    url="https://sites.google.com/view/tactile-ergodic-control/",
    python_requires=">=3.8",
)
