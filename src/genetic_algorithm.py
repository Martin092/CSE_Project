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
        mutation: OrderedDict[str, float],
        num_clusters: int,
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
        self.mutation_probability = (
            mutation  # Ordered dictionary of mutations and their probabilities
        )
        if num_selection == 0:
            num_selection = max(int(num_clusters / 2), 2)
        self.num_selection = (
            num_selection  # Number of parents to select from each generation
        )
        self.preserve = preserve  # Whether to preserve parents in future generation
        self.num_clusters = num_clusters  # Number of clusters in generation
        self.energies: List[float] = (
            []
        )  # Generate list for storing potentials of current generation
        self.cluster_list: List[Atoms] = (
            []
        )  # Generate list for storing configurations of current generation

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
        self.current_iteration = 0  # Current iteration/generation set to 0
        self.configs = []  # List for storing the best configuration per generation
        self.best_potential: float = float("inf")
        self.energies = []  # List for storing potentials of current generation
        self.potentials = []  # List for storing the best potentials per generation
        self.num_atoms = num_atoms  # Number of atoms in cluster
        self.atom_type = atom_type  # Atomic type of cluster
        self.utility = Utility(self, num_atoms, atom_type)  # set up Utility object
        self.best_config: Atoms = (
            self.utility.generate_cluster()
        )  # Random setup configuration
        self.current_cluster = self.utility.generate_cluster(
            initial_configuration
        )  # Unused by GA
        self.cluster_list = []  # List for storing configurations of current generation
        for _ in range(self.num_clusters):  # Generate initial generation configurations
            clus = (
                self.utility.generate_cluster()
            )  # Generate Atoms object with random positions
            self.cluster_list.append(
                clus
            )  # Add object to list of configurations in generation
        if (
            self.comm is None
        ):  # If executed sequentially, parent directory is the project root
            self.logfile = "../log.txt"
        else:  # If executed in parallel, root directory is the project directory
            self.logfile = "./log.txt"  # pragma: no cover

    def iteration(self) -> None:
        """
        Performs single iteration of the Genetic Algorithm. That is first perform local optimization and save relevant
        information from discovered local minima, then perform the genetic operators (selection, crossover, mutation)
        to create a new generation.
        :return: None, since everything is store in class fields.
        """
        if self.debug:
            print(f"Iteration {self.current_iteration}", flush=True)
        self.energies = []  # Empty list of potentials from previous generation
        if self.comm is None:  # If executing sequentially
            for index, cluster in enumerate(self.cluster_list):
                if self.debug:
                    print(
                        f"Cluster {index+1}/{self.num_clusters} begins local optimization.",
                        flush=True,
                    )
                opt = self.local_optimizer(
                    cluster, logfile=self.logfile
                )  # Set up local optimizer
                t = time.time()  # Save start time
                opt.run(steps=20000)  # Perform local optimization
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
            clus = self.utility.generate_cluster(
                child
            )  # Generate Atoms object for each child
            self.cluster_list.append(
                clus
            )  # Add child to current list of configurations
        if self.comm is not None:  # pragma: no cover
            # If executing in parallel
            ranks = self.comm.Get_size() - 1  # Number of worker processes
            size = int(
                self.num_clusters / ranks
            )  # Number of clusters per worker process
            send: List[Any] = []  # List of clusters to send to worker processes
            for i in range(ranks):
                send.append(
                    []
                )  # For each work process, generate a list of clusters to send
            for index, cluster in enumerate(self.cluster_list):
                receiver = index % (
                    self.comm.Get_size() - 1
                )  # Compute receiver rank per cluster
                send[receiver].append(
                    cluster.positions
                )  # Append the atomic configuration of cluster
            for receiver in range(ranks):
                if self.debug:
                    print(
                        f"Sending to {receiver+1} from {self.comm.Get_rank()}.",
                        flush=True,
                    )
                # Distribute the cluster configurations to worker processes
                self.comm.Send(
                    [np.array(send[receiver]), MPI.DOUBLE], dest=receiver + 1, tag=1
                )
            self.cluster_list = []  # Empty cluster list
            for _ in range(ranks):  # For each rank
                pos = np.empty(
                    (size, self.num_atoms, 3), dtype=np.float64
                )  # Clusters to receive
                if self.debug:
                    print(f"{self.comm.Get_rank()} receiving.", flush=True)
                # Receive cluster configurations from worker processes
                self.comm.Recv([pos, MPI.DOUBLE], tag=2, source=MPI.ANY_SOURCE)
                for (
                    i
                ) in (
                    pos
                ):  # For each received cluster, generate Atoms object from configuration
                    clus = self.utility.generate_cluster(i)  # type: ignore
                    self.cluster_list.append(
                        clus
                    )  # Append Atoms object to cluster list
        else:  # If executing sequentially
            for index, cluster in enumerate(self.cluster_list):
                if self.debug:
                    print(f"Mutating cluster {index+1}/{self.num_clusters}", flush=True)
                self.mutation(cluster)  # Perform mutation

    def is_converged(self) -> bool:
        """
        Checks if convergence criteria is satisfied.
        :return: True if convergence criteria is met, otherwise False.
        """
        if self.current_iteration < self.conv_iterations:
            # If convergence iterations not reached, don't even check
            return False
        ret = True  # Set up value to true
        cur = self.potentials[
            self.current_iteration - 1
        ]  # Get the best potential of last iteration
        for i in range(
            self.current_iteration - 1 - self.conv_iterations,
            self.current_iteration - 1,
        ):  # Check the last conv_iterations generations
            ret &= bool(
                abs(cur - self.potentials[i]) <= 10**-6
            )  # If one doesn't match, False
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
        )  # Zip potentials and clusters
        pairs.sort(key=lambda x: x[0])  # Sort clusters on potentials
        if self.best_potential > pairs[0][0]:  # If better potential is found, update
            self.best_potential = pairs[0][0]
            self.best_config = pairs[0][1].copy()  # type: ignore
        self.potentials.append(
            self.best_potential
        )  # Append the best potential of current generation
        self.configs.append(
            self.best_config
        )  # Append the best configuration of current generation
        if self.preserve:  # If selected parents should be preserved
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
            # Ensure the needed number of configurations is reached
            i = np.random.randint(0, len(parents))
            parent1 = parents[i]  # Choose random candidate as parent
            j = np.random.randint(0, len(parents))
            while j == i:  # Ensure different parents are selected at random
                j = np.random.randint(0, len(parents))
            parent2 = parents[j]  # Choose another random parent
            group = self.crossover(
                parent1, parent2
            )  # Generate atomic configuration of two children
            child = []  # Set up list for atomic positions of child
            for atom in group:
                child.append(atom.position)  # Extract only atomic positions of child
            crossover.append(child)  # Append child configuration to list
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

            # Group children atomic position lists
            child_1 = group11 + group22
            child_2 = group12 + group21

            # Set up children configurations lists
            group1 = []
            group2 = []

            # For each child, append the atomic positions
            for atom in child_1:
                group1.append(atom.position)
            for atom in child_2:
                group2.append(atom.position)

            if (
                self.utility.configuration_validity(np.array(group1))
                and len(group1) == self.num_atoms
            ):  # Check if children generation is a valid configuration
                return child_1
            if (
                self.utility.configuration_validity(np.array(group2))
                and len(group2) == self.num_atoms
            ):  # Check if children generation is a valid configuration
                return child_2

    def mutation(self, cluster: Atoms, max_local_steps: int = 20000) -> None:
        """
        Perform mutation on a cluster.
        :param cluster: Cluster to be mutated.
        :param max_local_steps: Maximum number of steps for the local optimizer.
        :return: None, since clusters are dynamically binding object in the scope of the program.
        """
        for i in self.mutation_probability:  # Iterate over mutation Ordered Dictionary
            if i == "twist":  # If key is twist
                if (
                    np.random.rand() <= self.mutation_probability[i]
                ):  # If drawn probability is sufficient
                    if self.debug:
                        print("Twist", flush=True)
                    self.utility.twist(cluster)  # Perform twist mutation
            elif i == "random displacement":  # If key is random displacement
                self.utility.random_displacement(cluster, self.mutation_probability[i])
            elif i == "angular":  # If key is angular
                if (
                    np.random.rand() <= self.mutation_probability[i]
                ):  # If drawn probability is sufficient
                    if self.debug:
                        print("Angular", flush=True)
                    self.utility.angular_movement(
                        cluster
                    )  # Perform angular movement mutation
            elif i == "random step":  # If key is random step
                if (
                    np.random.rand() <= self.mutation_probability[i]
                ):  # If drawn probability is sufficient
                    if self.debug:
                        print("Random step", flush=True)
                    self.utility.random_step(
                        cluster, max_local_steps
                    )  # Perform random step mutation
            elif i == "etching":  # If key is etching
                etching = np.random.rand()  # Draw probability
                if (
                    etching <= self.mutation_probability[i]
                ):  # If drawn probability is sufficient
                    if etching < self.mutation_probability[i] / 2:  # But lower half
                        if self.debug:
                            print("Etching (-)", flush=True)
                        self.utility.etching_subtraction(
                            cluster, max_local_steps
                        )  # Perform etching mutation (-)
                    else:  # But upper half
                        if self.debug:
                            print("Etching (+)", flush=True)
                        self.utility.etching_addition(
                            cluster, max_local_steps
                        )  # Perform etching mutation (+)
