"""Global Optimizer module"""

import time
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Literal
from mpi4py import MPI  # pylint: disable=E0611
from ase import Atoms
from ase.io import Trajectory
import numpy as np

from src.utility import Utility


class GlobalOptimizer(ABC):
    """
    Class Interface for Global Optimizers
    """

    def __init__(
        self,
        local_optimizer: Any,
        calculator: Any,
        comm: MPI.Intracomm | None = None,
        log: bool = True
    ) -> None:
        """
        Global Optimizer Class Constructor
        :param local_optimizer: ASE Optimizer.
        :param calculator: ASE Calculator.
        :param comm: MPI global communicator object.
        """
        self.local_optimizer: Any = local_optimizer
        self.current_iteration: int = 0
        self.calculator: Any = calculator
        self.utility: Utility = Utility(self, 0, "C")
        self.execution_time: float = 0.0
        self.comm: MPI.Intracomm | None = comm
        self.current_cluster: Atoms = self.utility.generate_cluster()
        self.best_potential: float = float("inf")
        self.best_config: Atoms = self.utility.generate_cluster()
        self.potentials: List[float] = []
        self.configs: List[Atoms] = []
        self.conv_iters: int = 0
        self.num_atoms: int = 0
        self.atom_type: str = ""
        self.log = log

    @abstractmethod
    def iteration(self) -> None:
        """
        Performs single iteration of the Global Optimizer algorithm.
        :return: None
        """

    @abstractmethod
    def is_converged(self) -> bool:
        """
        Checks if convergence criteria is satisfied.
        :return: True if convergence criteria is met, otherwise False.
        """

    def setup(
        self,
        num_atoms: int,
        atom_type: str,
        initial_configuration: (
            np.ndarray[Tuple[Any, Literal[3]], np.dtype[np.float64]] | None
        ) = None,
    ) -> None:
        """
        Sets up the clusters by either initializing random clusters or using the seed provided.
        :param num_atoms: Number of atoms in cluster.
        :param atom_type: Atomic type of cluster.
        :param initial_configuration: Atomic configuration, if None or Default, randomly generated.
        :return: None.
        """
        self.current_iteration = 0
        self.num_atoms = num_atoms
        self.atom_type = atom_type
        self.utility = Utility(self, num_atoms, atom_type)
        self.current_cluster = self.utility.generate_cluster(initial_configuration)

    def run(
        self,
        num_atoms: int,
        atom_type: str,
        max_iterations: int,
        conv_iters: int = 10,
        seed: int | None = None,
        initial_configuration: (
            np.ndarray[Tuple[Any, Literal[3]], np.dtype[np.float64]] | None
        ) = None,
    ) -> None:
        """
        Executes the Global Optimizer algorithm.
        :param num_atoms: Number of atoms in cluster to optimize for.
        :param atom_type: Atomic type of cluster.
        :param max_iterations: Number of maximum iterations to perform.
        :param conv_iters: Number of iterations to be considered in the convergence criteria.
        :param seed: Seeding for reproducibility.
        :param initial_configuration: Atomic configuration, if None or Default, randomly generated.
        :param log: Are logs printed to the terminal
        :return: None.
        """
        np.random.seed(seed)
        self.conv_iters = conv_iters
        start_time = time.time()
        self.setup(num_atoms, atom_type, initial_configuration)

        while self.current_iteration < max_iterations and not self.is_converged():
            self.iteration()
            self.current_iteration += 1

        self.execution_time = time.time() - start_time

    def write_trajectory(self, filename: str) -> None:  # pragma: no cover
        """
        Writes all clusters in the history to a trajectory file
        :param filename: Path of the trajectory file, with .traj extension
        :return: None, writes to file
        """
        with Trajectory(filename, mode="w") as traj:  # type: ignore
            for cluster in self.configs:
                cluster.center()  # type: ignore
                traj.write(cluster)  # pylint: disable=E1101
