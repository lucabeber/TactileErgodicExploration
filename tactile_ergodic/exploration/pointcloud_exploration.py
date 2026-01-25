"""
Pointcloud Exploration Base Class

This module provides the abstract base class for all point cloud-based exploration
algorithms. It handles common functionality like:
- Point cloud loading and processing
- Agent initialization
- Main exploration loop structure
- Result storage

Copyright (c) 2024 Idiap Research Institute, http://www.idiap.ch/
Written by Cem Bilaloglu <cem.bilaloglu@idiap.ch>
"""

from abc import ABC, abstractmethod

import numpy as np

from ..utils import FirstOrderAgent, SecondOrderAgent, config
from ..utils.pointcloud_utils import process_point_cloud


class PointcloudExploration(ABC):
    """
    Abstract base class for point cloud-based exploration algorithms.

    This class provides common infrastructure for exploration algorithms that
    operate on point cloud manifolds. Subclasses implement specific exploration
    strategies (HEDAC, SMC, etc.).

    Common functionality:
    - Point cloud loading and downsampling
    - Agent initialization (FirstOrder or SecondOrder)
    - Main exploration loop
    - Trajectory and metrics tracking

    Subclass responsibilities:
    - Implement setup_algorithm(): Algorithm-specific initialization
    - Implement run_exploration_step(): Single exploration step logic
    """

    def __init__(
        self,
        obj_name="bun270_X",
        voxel_size=0.002,
        timesteps=1500,
        agent_type="first_order",
        agent_start_index=None,
        verbose=True,
    ):
        """
        Initialize point cloud exploration.

        Args:
            obj_name: Name of point cloud file (without .ply extension)
            voxel_size: Voxel size for downsampling
            timesteps: Number of exploration timesteps
            agent_type: "first_order" or "second_order"
            agent_start_index: Index of vertex where agent starts (None for random)
            verbose: Whether to print progress messages
        """
        self.obj_name = obj_name
        self.voxel_size = voxel_size
        self.timesteps = timesteps
        self.agent_type = agent_type
        self.agent_start_index = agent_start_index
        self.verbose = verbose

        # Will be set during setup
        self.pcloud = None
        self.agent = None

    def setup_point_cloud(self, additional_params=None):
        """
        Load and process the point cloud.

        Args:
            additional_params: Dictionary with algorithm-specific parameters
                             (e.g., {'alpha': 100} for HEDAC)
        """
        if self.verbose:
            print("=" * 50)
            print("Loading and Processing Point Cloud")
            print("=" * 50)

        # Create parameter object for process_point_cloud
        class Param:
            pass

        param = Param()
        param.voxel_size = self.voxel_size

        # Add any algorithm-specific parameters
        if additional_params is not None:
            for key, value in additional_params.items():
                setattr(param, key, value)

        # Load and process point cloud
        ply_path = str(config.get_point_cloud_path(f"{self.obj_name}.ply"))
        self.pcloud = process_point_cloud(ply_path, param)

        if self.verbose:
            print(f"Point cloud processed: {len(self.pcloud.vertices)} points")

    def initialize_agent(
        self,
        max_velocity=None,
        max_acceleration=None,
        agent_radius=None,
    ):
        """
        Initialize the exploration agent.

        Args:
            max_velocity: Maximum velocity (default: 0.2 * voxel_size)
            max_acceleration: Maximum acceleration (default: 2.0 * max_velocity)
            agent_radius: Sensor radius (default: 2.5 * voxel_size)
        """
        if self.verbose:
            print("\n" + "=" * 50)
            print("Initializing Agent")
            print("=" * 50)

        # Set default parameters based on voxel size
        if agent_radius is None:
            agent_radius = 2.5 * self.voxel_size
        if max_velocity is None:
            max_velocity = 0.1 * self.voxel_size * 2
        if max_acceleration is None:
            max_acceleration = 1.0 * max_velocity * 2

        # Determine starting position
        if self.agent_start_index is None:
            # Random start
            start_idx = np.random.randint(len(self.pcloud.vertices))
        else:
            start_idx = self.agent_start_index

        start_position = self.pcloud.vertices[start_idx].copy()

        # Create agent based on type
        if self.agent_type == "first_order":
            self.agent = FirstOrderAgent(
                x=start_position,
                max_velocity=max_velocity,
                dim_t=self.timesteps,
            )
        elif self.agent_type == "second_order":
            self.agent = SecondOrderAgent(
                x=start_position,
                max_velocity=max_velocity,
                max_acceleration=max_acceleration,
                dim_t=self.timesteps,
            )
        else:
            raise ValueError(f"Unknown agent_type: {self.agent_type}")

        self.agent.radius = agent_radius

        if self.verbose:
            print(f"Agent type: {self.agent_type}")
            print(f"Agent initialized at vertex {start_idx}: {self.agent.x}")
            print(f"Agent radius: {agent_radius:.3e}")
            print(f"Max velocity: {max_velocity:.3e}")

    @abstractmethod
    def setup_algorithm(self):
        """
        Algorithm-specific setup.

        This method should initialize any algorithm-specific data structures
        (e.g., Laplacian matrices for HEDAC, eigenpairs for SMC, etc.).

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement setup_algorithm()")

    @abstractmethod
    def run_exploration_step(self, t, goal_density=None):
        """
        Execute a single exploration step.

        This method implements the core exploration logic for one timestep.
        It should compute the control signal and update the agent's position.

        Args:
            t: Current timestep
            goal_density: Target distribution (N,) array (optional, algorithm-specific)

        Returns:
            dict: Step results (algorithm-specific, e.g., coverage, heat, etc.)

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement run_exploration_step()")

    def explore(self, goal_density=None):
        """
        Main exploration loop.

        This method runs the complete exploration process:
        1. Setup point cloud and algorithm
        2. Initialize agent
        3. Run exploration loop
        4. Return results

        Args:
            goal_density: Target distribution (optional, can be set per-step)

        Returns:
            dict: Exploration results including trajectory and algorithm-specific data
        """
        # Setup
        self.setup_point_cloud()
        self.setup_algorithm()
        self.initialize_agent()

        if self.verbose:
            print("\n" + "=" * 50)
            print(f"Running {self.__class__.__name__} Exploration")
            print("=" * 50)

        # Initialize agent time counter
        self.agent.t = 0

        # Main exploration loop
        for t in range(self.timesteps):
            # Run algorithm-specific exploration step
            step_results = self.run_exploration_step(t, goal_density)

            # Progress update
            if self.verbose and (t + 1) % 100 == 0:
                print(f"Step {t+1}/{self.timesteps}")

        if self.verbose:
            print("\n" + "=" * 50)
            print("Exploration Complete!")
            print("=" * 50)

        # Return results (trajectory + algorithm-specific data)
        results = {
            "trajectory": self.agent.x_arr[:self.agent.t, :],  # Shape: (timesteps, 3)
            "pcloud": self.pcloud,
        }

        return results
