"""Genetic Algorithms module"""

import time
from typing import List, Tuple, Literal, Any
from collections import OrderedDict
from ase import Atoms, Atom
from ase.optimize import BFGS
from ase.calculators.lj import LennardJones
import numpy as np
from mpi4py import MPI  # pylint: disable=E0611

from src.global_optimizer import GlobalOptimizer
from src.utility import Utility


class GeneticAlgorithm(GlobalOptimizer):
    """
    Class Structure for Genetic Algorithms
    """

    def __init__(  # pylint: disable=W0102
        self,
        mutation: OrderedDict[str, float] = OrderedDict(
            [("twist", 0.2), ("angular", 0.2), ("random step", 0.2), ("etching", 0.2)]
        ),
        num_clusters: int = 16,
        num_selection: int = 0,
        preserve: bool = True,
        local_optimizer: Any = BFGS,
        calculator: Any = LennardJones,
        comm: MPI.Intracomm | None = None,
        debug: bool = False,
    ) -> None:
        """
        Genetic Algorithm Class constructor
        :param mutation: dictionary specifying the order of different mutations
                         as well as probabilities for each type of mutation
        :param num_clusters: number of clusters/configurations per generation
        :param num_selection: number of parents to be selected from each generation
        :param preserve: whether to preserve parents in new generation or not
        :param local_optimizer: optimizer used to find local minima
        :param calculator: calculator used to derive energies and potentials
        :param comm: global parallel execution communicator
        :param debug: Whether to print every operation for debugging purposes.
        """
        super().__init__(
            local_optimizer=local_optimizer,
            calculator=calculator,
            comm=comm,
            debug=debug,
        )
        self.mutation_probability = mutation
        if num_selection == 0:
            num_selection = max(int(num_clusters / 2), 2)
        self.num_selection = num_selection
        self.preserve = preserve
        self.num_clusters = num_clusters
        self.energies: List[float] = (
            []
        )  # Generate list for storing potentials of current generation
        self.cluster_list: List[Atoms] = []

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
        self.configs = []
        self.best_potential: float = float("inf")
        self.energies = []
        self.num_atoms = num_atoms
        self.atom_type = atom_type
        self.utility = Utility(self, num_atoms, atom_type)
        self.best_config: Atoms = self.utility.generate_cluster()
        self.current_cluster = self.utility.generate_cluster(initial_configuration)
        self.cluster_list = []
        for _ in range(self.num_clusters):
            clus = self.utility.generate_cluster()
            self.cluster_list.append(clus)
        if self.comm is None:
            self.logfile = "../log.txt"
        else:
            self.logfile = "./log.txt"

    def iteration(self) -> None:
        """
        Performs single iteration of the Genetic Algorithm. That is first perform local optimization and save relevant
        information from discovered local minima, then perform the genetic operators (selection, crossover, mutation)
        to create a new generation.
        :return: None, since everything is store in class fields.
        """
        if self.debug:
            print(f"Iteration {self.current_iteration}", flush=True)
        self.energies = []
        if self.comm is not None:  # pragma: no cover
            for index, cluster in enumerate(self.cluster_list):
                receiver = index % (self.comm.Get_size() - 1) + 1
                if self.debug:
                    print(
                        f"Sending to {receiver} from {self.comm.Get_rank()}.",
                        flush=True,
                    )
                self.comm.Send([cluster.positions, MPI.DOUBLE], dest=receiver, tag=1)
            self.cluster_list = []
            for _ in range(self.num_clusters):
                pos = np.empty((self.num_atoms, 3), dtype=np.float64)
                if self.debug:
                    print(f"{self.comm.Get_rank()} receiving.", flush=True)
                self.comm.Recv([pos, MPI.DOUBLE], tag=2, source=MPI.ANY_SOURCE)
                clus = self.utility.generate_cluster(pos)
                self.cluster_list.append(clus)

        else:
            for index, cluster in enumerate(self.cluster_list):
                if self.debug:
                    print(
                        f"Cluster {index+1}/{self.num_clusters} begins local optimization.",
                        flush=True,
                    )
                opt = self.local_optimizer(cluster, logfile=self.logfile)
                t = time.time()
                opt.run(steps=20000)  # Local optimization
                if self.debug:
                    print(
                        f"Cluster {index+1}/{self.num_clusters} successfully optimized for {time.time()-t} s.",
                        flush=True,
                    )
        for cluster in self.cluster_list:
            self.energies.append(
                cluster.get_potential_energy()  # type: ignore
            )  # Compute potential energy
        parents = self.selection()  # Perform selection
        children = self.generate_children(parents)  # Generate children configurations
        for child in children:  # Add children to current generation
            clus = self.utility.generate_cluster(child)
            self.cluster_list.append(
                clus
            )  # Add child to current list of configurations
        for cluster in enumerate(self.cluster_list):  # type: ignore
            if self.debug:
                print(
                    f"Mutating cluster {cluster[0]+1}/{self.num_clusters}", flush=True
                )
            self.mutation(cluster[1])  # Perform mutation

    def is_converged(self) -> bool:
        """
        Checks if convergence criteria is satisfied.
        :return: True if convergence criteria is met, otherwise False.
        """
        if self.current_iteration < self.conv_iterations:
            return False
        ret = True
        cur = self.potentials[self.current_iteration - 1]
        for i in range(
            self.current_iteration - self.conv_iterations, self.current_iteration - 1
        ):
            ret &= bool(abs(cur - self.potentials[i]) <= 10**-6)
        if ret and self.debug:
            print(f"Converged at iteration {self.current_iteration}.", flush=True)
        return ret

    def selection(self) -> List[Atoms]:
        """
        Performs selection by taking half the population with the best fitness function (potential energy) values.
        :return: None, since relevant class lists are used for storing, thus they are only updated.
        """
        pairs = list(
            zip(self.energies, self.cluster_list)
        )  # Zip clusters, potentials and optimizers
        pairs.sort(key=lambda x: x[0])  # Sort clusters on potentials
        if self.best_potential > pairs[0][0]:
            self.best_potential = pairs[0][0]
            self.best_config = pairs[0][1].copy()  # type: ignore
        self.potentials.append(self.best_potential)
        self.configs.append(self.best_config)
        if self.preserve:
            self.cluster_list = [
                pair[1] for pair in pairs[: self.num_selection]
            ]  # Update current clusters to contain only selected
        return [pair[1] for pair in pairs[: self.num_selection]]

    def generate_children(
        self, parents: List[Atoms]
    ) -> List[np.ndarray[Tuple[Any, Literal[3]], np.dtype[np.float64]]]:
        """
        Randomly selects two clusters as parents and generates at most two children out of them
        :param parents: List of parents to generate children from
        :return: List of atomic configurations (positions) of the children
        """
        crossover: List[Any] = []  # List of children atomic positions
        while len(crossover) + len(self.cluster_list) < self.num_clusters:
            i = np.random.randint(0, len(parents))
            parent1 = parents[i]  # Choose random candidate as parent
            j = np.random.randint(0, len(parents))
            while j == i:
                j = np.random.randint(0, len(parents))
            parent2 = parents[j]  # Choose another random parent
            group = self.crossover(
                parent1, parent2
            )  # Generate atomic configuration of two children
            child = []
            for atom in group:
                child.append(atom.position)  # Extract only atomic positions of child
            crossover.append(child)
        return crossover

    def crossover(self, cluster1: Atoms, cluster2: Atoms) -> List[Atom]:
        """
        Aligns clusters, then takes a random plane to split each cluster in two and merges opposing parts.
        :param cluster1: One of the parent clusters.
        :param cluster2: The other of the parent clusters.
        :return: Two lists of atoms, defining the atomic positions of two children, generated by crossing two parents.
        """
        # Align clusters
        self.utility.align_cluster(cluster1)
        self.utility.align_cluster(cluster2)

        # Generate four lists, two per cluster
        group11: List[Any] = []
        group12: List[Any] = []
        group21: List[Any] = []
        group22: List[Any] = []
        while True:
            # Generate 3 points to define the random plane which will split the clusters
            p1 = np.random.rand(3) - 0.5
            p2 = np.random.rand(3) - 0.5
            p3 = np.random.rand(3) - 0.5
            # Split the clusters
            group11, group12, _ = self.utility.split_cluster(cluster1, p1, p2, p3)
            group21, group22, _ = self.utility.split_cluster(cluster2, p1, p2, p3)

            child_1 = group11 + group22
            child_2 = group12 + group21

            group1 = []
            group2 = []

            for atom in child_1:
                group1.append(atom.position)
            for atom in child_2:
                group2.append(atom.position)

            if (
                self.utility.configuration_validity(np.array(group1))
                and len(group1) == self.num_atoms
            ):
                return child_1
            if (
                self.utility.configuration_validity(np.array(group2))
                and len(group2) == self.num_atoms
            ):
                return child_2

    def mutation(self, cluster: Atoms) -> None:
        """
        Perform mutation on a cluster.
        :param cluster: Cluster to be mutated.
        :return: None, since clusters are dynamically binding object in the scope of the program.
        """
        for i in self.mutation_probability:
            if i == "twist":
                if np.random.rand() <= self.mutation_probability[i]:
                    if self.debug:
                        print("Twist", flush=True)
                    self.utility.twist(cluster)  # Perform twist mutation
            elif i == "angular":
                if np.random.rand() <= self.mutation_probability[i]:
                    if self.debug:
                        print("Angular", flush=True)
                    self.utility.angular_movement(
                        cluster
                    )  # Perform angular movement mutation
            elif i == "random step":
                if np.random.rand() <= self.mutation_probability[i]:
                    if self.debug:
                        print("Random step", flush=True)
                    self.utility.random_step(cluster)  # Perform random step mutation
            elif i == "etching":
                etching = np.random.rand()
                if etching <= self.mutation_probability[i]:
                    if etching < self.mutation_probability[i] / 2:
                        if self.debug:
                            print("Etching (-)", flush=True)
                        self.utility.etching_subtraction(
                            cluster
                        )  # Perform etching mutation (-)
                    else:
                        if self.debug:
                            print("Etching (+)", flush=True)
                        self.utility.etching_addition(
                            cluster
                        )  # Perform etching mutation (+)
