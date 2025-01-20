"""Global Optimizer module"""

import time
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Literal
from mpi4py import MPI  # pylint: disable=E0611
from ase import Atoms
import numpy as np
import threading

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
        self.utility: Utility = Utility(self, "")
        self.execution_time: float = 0.0
        self.comm: MPI.Intracomm | None = comm
        self.current_cluster: Atoms = self.utility.generate_cluster()
        self.best_potential: float = float("inf")
        self.best_config: Atoms = self.utility.generate_cluster()
        self.potentials: List[float] = []
        self.configs: List[Atoms] = []
        self.conv_iterations: int = 0
        self.debug = debug
        self.logfile = "../log.txt"
        self.atoms: str = ""
        self.finished = False
        self.stop_event = threading.Event()

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
        atoms: str,
        initial_configuration: (
            np.ndarray[Tuple[Any, Literal[3]], np.dtype[np.float64]] | None
        ) = None,
        seed: int | None = None,
    ) -> None:
        """
        Sets up the clusters by either initializing random clusters or using the seed provided.
        :param atoms: Number of atoms and atomic type in cluster.
        :param initial_configuration: Atomic configuration, if None or Default, randomly generated.
        :param seed: seed for the random number generator
        :return: None.
        """
        self.current_iteration = 0
        self.atoms = atoms
        self.utility = Utility(self, atoms)
        self.current_cluster = self.utility.generate_cluster(
            initial_configuration, seed
        )
        if self.comm is None:
            self.logfile = "../log.txt"
        else:
            self.logfile = "./log.txt"  # pragma: no cover

    def run(
        self,
        atoms: str,
        max_iterations: int,
        conv_iterations: int = 0,
        seed: int | None = None,
        initial_configuration: (
            np.ndarray[Tuple[Any, Literal[3]], np.dtype[np.float64]] | None
        ) = None,
    ) -> None:
        """
        Executes the Global Optimizer algorithm.
        :param atoms: Number of atoms and atomic type in cluster.
        :param max_iterations: Number of maximum iterations to perform.
        :param conv_iterations: Number of iterations to be considered in the convergence criteria.
        :param seed: Seeding for reproducibility.
        :param initial_configuration: Atomic configuration, if None or Default, randomly generated.
        :return: None.
        """
        self.finished = False
        if conv_iterations == 0:
            conv_iterations = max_iterations
        self.conv_iterations = conv_iterations
        start_time = time.time()
        self.setup(atoms, initial_configuration, seed)

        while self.current_iteration < max_iterations and not self.is_converged():
            if self.stop_event.is_set():
                print('stopped')
                break
            self.iteration()
            self.current_iteration += 1

        self.execution_time = time.time() - start_time
        self.finished = True
        if self.debug and self.current_iteration == max_iterations:
            print("Maximum number of iterations reached", flush=True)
