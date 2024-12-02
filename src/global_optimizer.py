"""TODO: Write this."""

import time
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from mpi4py import MPI
from ase import Atoms
from ase.io import write, Trajectory
import numpy as np
from src.utility import Utility


class GlobalOptimizer(ABC):
    """
    TODO: Write this.
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
            * (1 / 2 + ((3.0 * self.atoms) / (4 * np.pi * np.sqrt(2))) ** (1 / 3))
        )
        self.atom_type = atom_type
        self.calculator = calculator
        self.utility = Utility(self)
        self.execution_time: float = 0.0
        self.comm = comm

    @abstractmethod
    def iteration(self) -> None:
        """
        TODO: Write this.
        :return:
        """

    @abstractmethod
    def is_converged(self) -> bool:
        """
        TODO: Write this.
        :return:
        """

    def setup(self, seed: Atoms | None = None) -> None:
        """
        Sets up the clusters by either initializing random clusters or using the seed provided
        :param seed: A cluster that is used as initial point of the optimization
        :return:
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
                positions = (
                    (np.random.rand(self.atoms, 3) - 0.5) * self.box_length * 1.5
                )  # 1.5 is a magic number
                # In the future, instead of number of atoms,
                # we ask the user to choose how many atoms they want for each atom type.
                clus = Atoms(self.atom_type + str(self.atoms), positions=positions)  # type: ignore
            clus.calc = self.calculator()
            self.cluster_list.append(clus)
            opt = self.local_optimizer(clus, logfile="log.txt")
            self.optimizers.append(opt)
            self.history.append([])

    def run(self, max_iterations: int, seed: Atoms | None = None) -> None:
        """
        TODO: Write this.
        :param max_iterations:
        :param seed:
        :return:
        """
        start_time = time.time()
        self.setup(seed)

        while self.current_iteration < max_iterations and not self.is_converged():
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

    def best_energy(self, index: int = 0) -> Tuple[float, Atoms]:
        """
        Gets the best energy from the history
        @param index which cluster from the history to pick from
        """
        min_energy = float("inf")
        best_cluster: Atoms = self.cluster_list[index][0]
        for cluster in self.history[index]:
            cluster.calc = self.calculator()
            curr_energy = cluster.get_potential_energy()
            if curr_energy < min_energy:
                min_energy = curr_energy
                best_cluster = cluster

        return min_energy, best_cluster

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
