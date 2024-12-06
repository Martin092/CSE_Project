"""Genetic Algorithms module"""

import time
from typing import List, Tuple, Literal, Any
from ase import Atoms, Atom
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from ase.calculators.lj import LennardJones
import numpy as np
from matplotlib import pyplot as plt

from src.global_optimizer import GlobalOptimizer
from src.oxford_database import get_cluster_energy


class GeneticAlgorithm(GlobalOptimizer):
    """
    Class Structure for Genetic Algorithms
    """

    def __init__(
        self,
        mutation_probability: float = 0.2,
        num_clusters: int = 16,
        local_optimizer: Any = BFGS,
        atoms: int = 30,
        atom_type: str = "C",
        calculator: Any = LennardJones,
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
        self.best_potentials: List[float] = []
        self.best_configs: List[Atoms] = []
        self.best_config: Atoms = Atoms(
            self.atom_type + str(self.atoms), positions=np.random.rand(self.atoms, 3)
        )  # type: ignore
        self.best_potential: float = float("inf")
        self.potentials: List[float] = (
            []
        )  # Generate list for storing potentials of current generation

    def iteration(self) -> None:
        """
        Performs single iteration of the Genetic Algorithm. That is first perform local optimization and save relevant
        information from discovered local minima, then perform the genetic operators (selection, crossover, mutation)
        to create a new generation.
        :return: None, since everything is store in class fields.
        """
        print(f"Iteration {self.current_iteration}")
        self.potentials = []
        for index, cluster in enumerate(self.cluster_list):
            print(f"Cluster {index}/{self.num_clusters} begins local optimization.")
            t = time.time()
            self.optimizers[index].run()  # Local optimization
            print(
                f"Cluster {index}/{self.num_clusters} successfully optimized for {time.time()-t} s."
            )
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
        return ret

    def selection(self) -> None:
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
        self.utility.align_cluster(cluster1)
        self.utility.align_cluster(cluster2)

        # Generate four lists, two per cluster
        group11: List[Any] = []
        group12: List[Any] = []
        group21: List[Any] = []
        group22: List[Any] = []
        while len(group11) + len(group22) != len(
            cluster1.positions
        ):  # Make sure split is even between different parts
            # Generate 3 points to define the random plane which will split the clusters
            p1 = np.random.rand(3) - 0.5
            p2 = np.random.rand(3) - 0.5
            p3 = np.random.rand(3) - 0.5
            # Split the clusters
            group11, group12, _ = self.utility.split_cluster(cluster1, p1, p2, p3)
            group21, group22, _ = self.utility.split_cluster(cluster2, p1, p2, p3)

            child_1 = group11 + group22
            child_2 = group12 + group21
        return child_1, child_2  # Return crossed parts of parent clusters

    def mutation(self, cluster: Atoms) -> None:
        """
        Perform mutation on a cluster.
        :param cluster: Cluster to be mutated.
        :return: None, since clusters are dynamically binding object in the scope of the program.
        """
        if np.random.rand() <= self.mutation_probability:
            print("Twist")
            self.utility.twist(cluster)  # Perform twist mutation
        if np.random.rand() <= self.mutation_probability:
            print("Angular")
            self.utility.angular_movement(cluster)  # Perform angular mutation
        etching = np.random.rand()
        if etching <= self.mutation_probability:
            if etching < self.mutation_probability / 2:
                print("Etching (-)")
                self.utility.etching_subtraction(
                    cluster
                )  # Perform etching mutation (-)
            else:
                print("Etching (+)")
                self.utility.etching_addition(cluster)  # Perform etching mutation (+)

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

    def benchmark_run(
        self, indices: List[int], num_iterations: int, conv_iters: int = 10
    ) -> None:
        """
        Benchmark execution of Genetic Algorithm.
        Measures the execution times, saves the best configurations history and plots the best potentials.
        :param indices: Cluster indices for LJ tests.
        :param num_iterations: Max number of iterations per execution.
        :param conv_iters: Number of iterations to conclude convergence.
        :return: None.
        """
        times = []
        convergence = []
        for lj in indices:
            self.atoms = lj
            self.run(num_iterations, conv_iters)

            best_cluster = self.best_config
            print(f"Best energy found: {self.best_potential}")
            write("clusters/minima_optimized.xyz", best_cluster)

            best = get_cluster_energy(lj, self.atom_type)

            if self.best_potential > best and self.best_potential - best < 0.001:
                print("Best energy matches the database")
            elif self.best_potential < best:
                print("GROUNDBREAKING!!!")
            else:
                print(f"Best energy in database is {best}.")

            traj = Trajectory("clusters/minima_progress.traj", mode="w")  # type: ignore
            for cluster in self.best_configs:
                traj.write(cluster)

            times.append(self.execution_time)
            convergence.append(self.current_iteration)
            print(
                f"Time taken: {int(np.floor_divide(self.execution_time, 60))} min {int(self.execution_time)%60} sec"
            )

            plt.plot(self.best_potentials)
            plt.title(f"Execution on LJ{lj}")
            plt.xlabel("Iteration")
            plt.ylabel("Potential Energy")
            plt.show()

        for k in enumerate(indices):
            print(
                f"LJ {k[1]}: {convergence[k[0]]} iterations for "
                f"{int(np.floor_divide(times[k[0]], 60))} min {int(times[k[0]])%60} sec"
            )
