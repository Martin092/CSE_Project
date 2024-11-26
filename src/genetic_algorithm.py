"""Class Structure for Genetic Algorithms"""

from typing import List, Tuple, Literal, Any
from ase import Atoms, Atom
from ase.optimize import BFGS
from ase.calculators.lj import LennardJones
import numpy as np
from src.global_optimizer import GlobalOptimizer


class GeneticAlgorithm(GlobalOptimizer):
    """
    Class Structure for Genetic Algorithms
    """

    def __init__(
        self,
        mutation_probability: float = 0.2,
        local_optimizer: Any = BFGS,
        atoms: int = 30,
        atom_type: str = "C",
        calculator: Any = LennardJones,
        num_clusters: int = 16,
    ) -> None:
        """
        Genetic Algorithm Class constructor
        :param mutation_probability: probability to perform mutation
        :param local_optimizer: optimizer used to find local minima
        :param atoms: number of atoms per configuration
        :param atom_type: the chemical element of atoms
        :param calculator: calculator used to derive energies and potentials
        :param num_clusters: number of clusters/configurations per generation
        """
        super().__init__(
            num_clusters=num_clusters,
            local_optimizer=local_optimizer,
            atoms=atoms,
            atom_type=atom_type,
            calculator=calculator,
        )
        self.mutation_probability = mutation_probability
        self.potentials: List[Any] = (
            []
        )  # Generate list for storing potentials of current generation

    def iteration(self) -> None:
        """
        Performs single iteration of the Genetic Algorithm. That is first perform local optimization and save relevant
        information from discovered local minima, then perform the genetic operators (selection, crossover, mutation)
        to create a new generation.
        :return: None, since everything is store in class fields.
        """
        for index, cluster in enumerate(self.cluster_list):
            self.optimizers[index].run(fmax=0.1)  # Local optimization
            self.history[self.current_iteration].append(cluster)  # Save local minima
            self.potentials.append(
                cluster.get_potential_energy()
            )  # Compute potential energy
        self.selection()  # Perform selection
        children = self.generate_children()  # Generate children configurations
        for child in children:  # Add children to current generation
            clus = Atoms(  # type: ignore
                self.atom_type + str(self.atoms), positions=child
            )  # Create a child object
            clus.calc = self.calculator()  # Assign energy calculator
            self.cluster_list.append(
                clus
            )  # Add child to current list of configurations
            opt = self.local_optimizer(
                clus, logfile="log.txt"
            )  # Create a local optimizer object
            self.optimizers.append(
                opt
            )  # Add local optimizer object to current list of optimizers
        for cluster in self.cluster_list:
            self.mutation(cluster)  # Perform mutation

    def is_converged(self) -> bool:
        """
        Checks if convergence criteria is satisfied
        :return: True if convergence criteria is met, otherwise False
        """
        return False  # TODO: implement

    def selection(self) -> None:
        """
        Performs selection by taking half the population with the best fitness function (potential energy) values.
        :return: None, since relevant class lists are used for storing, thus they are only updated.
        """
        pairs = list(
            zip(self.potentials, self.cluster_list, self.optimizers)
        )  # Zip clusters, potentials and optimizers
        pairs.sort(key=lambda x: x[0])  # Sort clusters on potentials
        midpoint = (
            len(pairs) + 1
        ) // 2  # Determine the number of clusters to be selected
        self.cluster_list = [
            pair[1] for pair in pairs[:midpoint]
        ]  # Update current clusters to contain only selected
        self.optimizers = [
            pair[2] for pair in pairs[:midpoint]
        ]  # Update current optimizers to contain only selected

    def generate_children(
        self,
    ) -> List[List[np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]]]]:
        """
        Randomly selects two clusters as parents and generates at most two children out of them
        :return: List of atomic configurations (positions) of the children
        """
        crossover: List[Any] = []  # List of children atomic positions
        while len(crossover) + len(self.cluster_list) < self.num_clusters:
            i = np.random.randint(0, len(self.cluster_list))
            parent1 = self.cluster_list[i]  # Choose random candidate as parent
            j = np.random.randint(0, len(self.cluster_list))
            while j == i:
                j = np.random.randint(0, len(self.cluster_list))
            parent2 = self.cluster_list[j]  # Choose another random parent
            group1, group2 = self.crossover(
                parent1, parent2
            )  # Generate atomic configuration of two children
            child1 = []
            for atom in group1:
                child1.append(atom.position)  # Extract only atomic positions of child
            crossover.append(child1)
            if (
                len(self.cluster_list) + len(crossover) != self.num_clusters
            ):  # Take second child only if necessary
                child2 = []
                for atom in group2:
                    child2.append(
                        atom.position
                    )  # Extract only atomic positions of child
                crossover.append(child2)
        return crossover

    def crossover(
        self, cluster1: Atoms, cluster2: Atoms
    ) -> Tuple[List[Atom], List[Atom]]:
        """
        Aligns clusters, then takes a random plane to split each cluster in two and merges opposing parts.
        :param cluster1: One of the parent clusters.
        :param cluster2: The other of the parent clusters.
        :return: Two lists of atoms, defining the atomic positions of two children, generated by crossing two parents.
        """
        # Align clusters
        self.disturber.align_cluster(cluster1)
        self.disturber.align_cluster(cluster2)

        # Generate four lists, two per cluster
        group11: List[Any] = []
        group12: List[Any] = []
        group21: List[Any] = []
        group22: List[Any] = []
        while len(group11) + len(group22) != len(
            cluster1.positions
        ):  # Make sure split is even between different parts
            # Generate 3 points to define the random plane which will split the clusters
            p1 = np.random.rand(3)
            p2 = np.random.rand(3)
            p3 = np.random.rand(3)
            # Split the clusters
            group11, group12, _ = self.disturber.split_cluster(cluster1, p1, p2, p3)
            group21, group22, _ = self.disturber.split_cluster(cluster2, p1, p2, p3)
        return (
            group11 + group22,
            group12 + group21,
        )  # Return crossed parts of parent clusters

    def mutation(self, cluster: Atoms) -> None:
        """
        Perform mutation on a cluster.
        :param cluster: Cluster to be mutated.
        :return: None, since clusters are dynamically binding object in the scope of the program.
        """
        if np.random.rand() <= self.mutation_probability:
            self.disturber.twist(cluster)  # Perform twist mutation
        for i in range(self.atoms):
            if np.random.rand() <= self.mutation_probability / self.atoms:
                cluster.positions[i] += (
                    np.random.rand(3) - 0.5
                )  # Perform single atom displacement mutation
