"""TODO: Write this."""
import time
from abc import ABC, abstractmethod
from typing import Any, List
from ase import Atoms
from ase.io import write
import numpy as np
from src.disturber import Disturber


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
        self.disturber = Disturber(self)
        self.execution_time: float = 0.0

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

    def setup(self) -> None:
        """
        TODO: Write this.
        :return:
        """
        self.current_iteration = 0
        self.history = []
        self.cluster_list = []
        self.optimizers = []
        for _ in range(self.num_clusters):
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

    def run(self, max_iterations: int) -> None:
        """
        TODO: Write this.
        :param max_iterations:
        :return:
        """
        start_time = time.time()
        self.setup()

        while self.current_iteration < max_iterations and not self.is_converged():
            self.history.append([])
            print(self.current_iteration)
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

    def compare_clusters(self, cluster1: Atoms, cluster2: Atoms) -> np.bool:
        """
        Checks whether two clusters are equal based on their potential energy.
        This method may be changed in the future to use more sophisticated methods,
        such as overlap matrix fingerprint thresholding.
        :param cluster1: First cluster
        :param cluster2: Second cluster
        :return: boolean
        """
        return np.isclose(cluster1.get_potential_energy(), cluster2.get_potential_energy())  # type: ignore

    def append_history(self) -> None:
        """
        Appends copies of all the clusters in the clusterList to the history.
        Copies are used since clusters are passed by reference
        :return:
        """
        for i, cluster in enumerate(self.cluster_list):
            self.history[i].append(cluster.copy())
