"""Global Optimizer module"""

import time
from abc import ABC, abstractmethod
from typing import Any, List, Tuple
from deprecated import deprecated
from mpi4py import MPI
from ase import Atoms
from ase.io import write, Trajectory
import numpy as np

from src.utility import Utility


class GlobalOptimizer(ABC):
    """
    Class Interface for Global Optimizers
    """

    def __init__(
        self,
        num_clusters: int,
        local_optimizer: Any,
        atoms: int,
        atom_type: str,
        calculator: Any,
        comm: MPI.Intracomm | None = None,
    ) -> None:
        self.history: List = []
        self.cluster_list: List = []
        self.optimizers: List = []
        self.local_optimizer = local_optimizer
        self.current_iteration = 0
        self.num_clusters = num_clusters
        self.atoms = atoms
        self.covalent_radius = 1.0
        self.box_length = (
            2
            * self.covalent_radius
            * (0.5 + ((3.0 * self.atoms) / (4 * np.pi * np.sqrt(2))) ** (1 / 3))
        )
        self.atom_type = atom_type
        self.calculator = calculator
        self.utility = Utility(self)
        self.execution_time: float = 0.0
        self.comm = comm

    @abstractmethod
    def iteration(self) -> None:
        """
        Performs single iteration of the Global Optimizer algorithm.
        :return: None
        """

    @abstractmethod
    def is_converged(self, conv_iters: int = 10) -> bool:
        """
        Checks if convergence criteria is satisfied.
        :param conv_iters: Number of iterations to be considered.
        :return: True if convergence criteria is met, otherwise False.
        """

    def setup(self, seed: Atoms | None = None) -> None:
        """
        Sets up the clusters by either initializing random clusters or using the seed provided
        :param seed: A cluster that is used as initial point of the optimization
        :return: None.
        """
        self.current_iteration = 0
        self.history = []
        self.cluster_list = []
        self.optimizers = []
        for _ in range(self.num_clusters):
            clus: Atoms
            if seed:
                clus = seed.copy()  # type: ignore
            else:
                while True:
                    positions = (
                        (np.random.rand(self.atoms, 3) - 0.5) * self.box_length * 1.5
                    )
                    if self.utility.configuration_validity(positions):
                        break
                clus = Atoms(self.atom_type + str(self.atoms), positions=positions)  # type: ignore
            clus.calc = self.calculator()
            self.cluster_list.append(clus)
            opt = self.local_optimizer(clus, logfile="log.txt")
            self.optimizers.append(opt)
            self.history.append([])

    def run(
        self, max_iterations: int, conv_iter: int = 10, seed: Atoms | None = None
    ) -> None:
        """
        Executes the Global Optimizer algorithm.
        :param max_iterations: Number of maximum iterations to perform.
        :param conv_iter: Number of iterations to be considered for evaluating convergence.
        :param seed: Seeding for reproducibility.
        :return: None.
        """
        start_time = time.time()
        self.setup(seed)

        while self.current_iteration < max_iterations and not self.is_converged(
            conv_iter
        ):
            self.history.append([])
            self.iteration()
            self.current_iteration += 1

        self.execution_time = time.time() - start_time

    def write_to_file(self, filename: str, cluster_index: int = 0) -> None:
        """
        Writes the cluster to a .xyz file.
        :param filename: the name of the file, does not matter if it has the .xyz extension
        :param cluster_index: which cluster will be written
        """
        filename = filename if filename[-4:] == ".xyz" else filename + ".xyz"
        write(f"clusters/{filename}", self.cluster_list[cluster_index])

    def write_trajectory(self, filename: str, cluster_index: int = 0) -> None:
        """
        Writes all clusters in the history to a trajectory file
        :param filename: File name of the trajectory file
        :param cluster_index: Which cluster history to write to the trajectory file
        :return: None, writes to file
        """
        with Trajectory(filename, mode="w") as traj:  # type: ignore
            for cluster in self.history[cluster_index]:
                traj.write(cluster)  # pylint: disable=E1101

    def append_history(self) -> None:
        """
        Appends copies of all the clusters in the clusterList to the history.
        Copies are used since clusters are passed by reference
        :return:
        """
        for i, cluster in enumerate(self.cluster_list):
            self.history[i].append(cluster.copy())

    def best_energy_cluster(self) -> Tuple[float, Atoms]:
        """
        Gets the best energy and the best cluster from the history
        """
        min_energy = float("inf")
        best_cluster: Atoms = self.cluster_list[0][0]
        for history_list in self.history:
            for cluster in history_list:
                cluster.calc = self.calculator()
                curr_energy = cluster.get_potential_energy()
                if curr_energy < min_energy:
                    min_energy = curr_energy
                    best_cluster = cluster

        return min_energy, best_cluster

    def best_energy(self) -> float:
        """
        Returns the best energy found so far
        """
        return self.best_energy_cluster()[0]

    def best_cluster(self) -> Atoms:
        """
        Returns the cluster with the best energy found so far
        """
        return self.best_energy_cluster()[1]

    def potentials_history(self, index: int = 0) -> list[float]:
        """
        Returns an array of all the energies found at a
        particular cluster index in the history
        """
        energies = []
        for cluster in self.history[index]:
            cluster.calc = self.calculator()
            curr_energy = cluster.get_potential_energy()
            energies.append(curr_energy)

        return energies

    @deprecated(reason="Get deprecated Moham, now switch to my method")
    def get_best_cluster_found(self, cluster_index: int = 0) -> Any:
        """
        Finds the cluster with the lowest energy from the cluster history
        :param cluster_index: Which cluster history to look through
        :return: Cluster with the lowest energy
        """
        # TODO Make this work for multiple clusters
        min_energy = float("inf")
        best_cluster = None
        for cluster in self.history[cluster_index]:
            cluster.calc = self.calculator()
            curr_energy = cluster.get_potential_energy()
            if curr_energy < min_energy:
                min_energy = curr_energy
                best_cluster = cluster

        return best_cluster
