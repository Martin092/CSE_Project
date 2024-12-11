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
        atoms: int = 30,
        atom_type: str = "C",
        calculator: Any = LennardJones,
        comm: MPI.Intracomm | None = None,
        parallel: bool = False,
    ) -> None:
        """
        Genetic Algorithm Class constructor
        :param mutation: dictionary specifying the order of different mutations
                         as well as probabilities for each type of mutation
        :param num_clusters: number of clusters/configurations per generation
        :param num_selection: number of parents to be selected from each generation
        :param preserve: whether to preserve parents in new generation or not
        :param local_optimizer: optimizer used to find local minima
        :param atoms: number of atoms per configuration
        :param atom_type: the chemical element of atoms
        :param calculator: calculator used to derive energies and potentials
        :param comm: global parallel execution communicator
        :param parallel: whether to execute in parallel or sequentially
        """
        super().__init__(
            num_clusters=num_clusters,
            local_optimizer=local_optimizer,
            atoms=atoms,
            atom_type=atom_type,
            calculator=calculator,
            comm=comm,
        )
        self.mutation_probability = mutation
        if num_selection == 0:
            num_selection = max(int(num_clusters / 2), 2)
        self.num_selection = num_selection
        self.preserve = preserve
        self.best_potentials: List[float] = []
        self.best_configs: List[Atoms] = []
        self.best_config: Atoms = Atoms(
            self.atom_type + str(self.atoms), positions=np.random.rand(self.atoms, 3)
        )  # type: ignore
        self.best_potential: float = float("inf")
        self.potentials: List[float] = (
            []
        )  # Generate list for storing potentials of current generation
        self.parallel = parallel

    def iteration(self) -> None:
        """
        Performs single iteration of the Genetic Algorithm. That is first perform local optimization and save relevant
        information from discovered local minima, then perform the genetic operators (selection, crossover, mutation)
        to create a new generation.
        :return: None, since everything is store in class fields.
        """
        print(f"Iteration {self.current_iteration}")
        self.potentials = []
        if self.parallel:
            for index, cluster in enumerate(self.cluster_list):
                receiver = index % (self.comm.Get_size() - 1) + 1  # type: ignore
                print(f"Sending to {receiver} from {self.comm.Get_rank()}.")  # type: ignore
                self.comm.Send([cluster.positions, MPI.DOUBLE], dest=receiver, tag=1)  # type: ignore
            self.cluster_list = []
            for _ in range(self.num_clusters):
                pos = np.empty((self.atoms, 3))
                print(f"{self.comm.Get_rank()} receiving.")  # type: ignore
                self.comm.Recv([pos, MPI.DOUBLE], tag=2, source=MPI.ANY_SOURCE)  # type: ignore
                clus = Atoms(  # type: ignore
                    self.atom_type + str(self.atoms),
                    positions=pos,
                    cell=np.array(
                        [
                            [self.box_length, 0, 0],
                            [0, self.box_length, 0],
                            [0, 0, self.box_length],
                        ]
                    ),
                )
                clus.calc = self.calculator()
                self.cluster_list.append(clus)

        if not self.parallel:
            for index, cluster in enumerate(self.cluster_list):
                print(
                    f"Cluster {index+1}/{self.num_clusters} begins local optimization."
                )
                opt = self.local_optimizer(cluster, logfile="../log.txt")
                t = time.time()
                opt.run(steps=20000)  # Local optimization
                print(
                    f"Cluster {index+1}/{self.num_clusters} successfully optimized for {time.time()-t} s."
                )
        for cluster in self.cluster_list:
            self.potentials.append(
                cluster.get_potential_energy()
            )  # Compute potential energy
        self.append_history()
        parents = self.selection()  # Perform selection
        children = self.generate_children(parents)  # Generate children configurations
        for child in children:  # Add children to current generation
            clus = Atoms(  # type: ignore
                self.atom_type + str(self.atoms),
                positions=child,
                cell=np.array(
                    [
                        [self.box_length, 0, 0],
                        [0, self.box_length, 0],
                        [0, 0, self.box_length],
                    ]
                ),
            )  # Create a child object
            clus.calc = self.calculator()  # Assign energy calculator
            self.cluster_list.append(
                clus
            )  # Add child to current list of configurations
        for cluster in enumerate(self.cluster_list):
            print(f"Mutating cluster {cluster[0]+1}/{self.num_clusters}")
            self.mutation(cluster[1])  # Perform mutation

    def is_converged(self, conv_iters: int = 10) -> bool:
        """
        Checks if convergence criteria is satisfied.
        :param conv_iters: Number of iterations to be considered.
        :return: True if convergence criteria is met, otherwise False.
        """
        if self.current_iteration < conv_iters:
            return False
        ret = True
        cur = self.best_potentials[self.current_iteration - 1]
        for i in range(self.current_iteration - conv_iters, self.current_iteration - 1):
            ret &= bool(abs(cur - self.best_potentials[i]) <= 10**-6)
        if ret:
            for cluster in self.best_configs:
                cluster.center()  # type: ignore
            self.history[0] = self.best_configs
        return ret

    def selection(self) -> List[Atoms]:
        """
        Performs selection by taking half the population with the best fitness function (potential energy) values.
        :return: None, since relevant class lists are used for storing, thus they are only updated.
        """
        pairs = list(
            zip(self.potentials, self.cluster_list, self.optimizers)
        )  # Zip clusters, potentials and optimizers
        pairs.sort(key=lambda x: x[0])  # Sort clusters on potentials
        if self.best_potential > pairs[0][0]:
            self.best_potential = pairs[0][0]
            self.best_config = pairs[0][1].copy()
        self.best_potentials.append(self.best_potential)
        self.best_configs.append(self.best_config)
        if self.preserve:
            self.cluster_list = [
                pair[1] for pair in pairs[: self.num_selection]
            ]  # Update current clusters to contain only selected
            self.optimizers = [
                pair[2] for pair in pairs[: self.num_selection]
            ]  # Update current optimizers to contain only selected
        return [pair[1] for pair in pairs[: self.num_selection]]

    def generate_children(
        self, parents: List[Atoms]
    ) -> List[List[np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]]]]:
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
                and len(group1) == self.atoms
            ):
                return child_1
            if (
                self.utility.configuration_validity(np.array(group2))
                and len(group2) == self.atoms
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
                    print("Twist")
                    self.utility.twist(cluster)  # Perform twist mutation
            elif i == "angular":
                if np.random.rand() <= self.mutation_probability[i]:
                    print("Angular")
                    self.utility.angular_movement(cluster)  # Perform angular mutation
            elif i == "random step":
                if np.random.rand() <= self.mutation_probability[i]:
                    print("Random Step")
                    self.utility.random_step(cluster)
            elif i == "etching":
                etching = np.random.rand()
                if etching <= self.mutation_probability[i]:
                    if etching < self.mutation_probability[i] / 2:
                        print("Etching (-)")
                        self.utility.etching_subtraction(
                            cluster
                        )  # Perform etching mutation (-)
                    else:
                        print("Etching (+)")
                        self.utility.etching_addition(
                            cluster
                        )  # Perform etching mutation (+)

    def best_energy(self) -> float:
        """
        Get the best energy found by the algorithm.
        :return: The minimal energy found by the algorithm.
        """
        return self.best_potential

    def best_cluster(self) -> Atoms:
        """
        Get the best cluster configuration found by the algorithm.
        :return: The minimal energy configuration found by the algorithm.
        """
        return self.best_config

    def potentials_history(self, index: int = 0) -> List[float]:
        """
        Get history of the best potentials per iteration.
        :return: List of the best potential energies per iteration.
        """
        return self.best_potentials
