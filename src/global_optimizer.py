"""Global Optimizer module"""

import time
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Literal
from mpi4py import MPI  # pylint: disable=E0611
from ase import Atoms
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
        debug: bool = False,
    ) -> None:
        """
        Global Optimizer Class Constructor
        :param local_optimizer: ASE Optimizer.
        :param calculator: ASE Calculator.
        :param comm: MPI global communicator object.
        :param debug: Whether to print every operation for debugging purposes.
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
        self.conv_iterations: int = 0
        self.num_atoms: int = 0
        self.atom_type: str = ""
        self.debug = debug
        self.logfile = "../log.txt"

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
        seed: int | None = None,
    ) -> None:
        """
        Sets up the clusters by either initializing random clusters or using the seed provided.
        :param num_atoms: Number of atoms in cluster.
        :param atom_type: Atomic type of cluster.
        :param initial_configuration: Atomic configuration, if None or Default, randomly generated.
        :param seed: seed for the random number generator
        :return: None.
        """
        self.current_iteration = 0
        self.num_atoms = num_atoms
        self.atom_type = atom_type
        self.utility = Utility(self, num_atoms, atom_type)
        self.current_cluster = self.utility.generate_cluster(
            initial_configuration, seed
        )
        if self.comm is None:
            self.logfile = "../log.txt"
        else:
            self.logfile = "./log.txt"  # pragma: no cover

    def run(
        self,
        num_atoms: int,
        atom_type: str,
        max_iterations: int,
        conv_iterations: int = 0,
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
        :param conv_iterations: Number of iterations to be considered in the convergence criteria.
        :param seed: Seeding for reproducibility.
        :param initial_configuration: Atomic configuration, if None or Default, randomly generated.
        :param log: Are logs printed to the terminal
        :return: None.
        """
        if conv_iterations == 0:
            conv_iterations = max_iterations
        self.conv_iterations = conv_iterations
        start_time = time.time()
        self.setup(num_atoms, atom_type, initial_configuration, seed)

        while self.current_iteration < max_iterations and not self.is_converged():
            self.iteration()
            self.current_iteration += 1

        self.execution_time = time.time() - start_time

        if self.debug and self.current_iteration == max_iterations:
            print("Maximum number of iterations reached", flush=True)
