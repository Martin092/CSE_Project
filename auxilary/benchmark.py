"""
A class that provides utility methods for easily
comparing the performance of two global optimization algorithms
"""

from typing import List

import numpy as np
from ase.io import write
from ase.optimize import BFGS
from matplotlib import pyplot as plt

from src.genetic_algorithm import GeneticAlgorithm
from src.global_optimizer import GlobalOptimizer
from src.basin_hopping_optimizer import BasinHoppingOptimizer
from src.minima_hopping_optimizer import MinimaHoppingOptimizer
from auxilary.oxford_database import get_cluster_energy


class Benchmark:
    """
    Class that provides utility methods for easily comparing
    the performance of global optimization algorithms
    """

    def __init__(self, optimizer: GlobalOptimizer):
        self.optimizer = optimizer

    def compare_to_oxford(self) -> float:
        """
        Returns the difference between the actual global minima and the one found
        by the algorithm
        """
        actual = get_cluster_energy(bh.atoms, bh.atom_type)
        energy = self.optimizer.best_energy()
        return actual - energy

    def plot_energies(self) -> None:
        """
        Plots the energy values over the course of the entire run
        """
        energies = np.array([])
        for clus in self.optimizer.history[0]:
            clus.calc = self.optimizer.calculator()
            energies = np.append(energies, clus.get_potential_energy())

        plt.plot(energies)
        plt.title(f"Energy levels discovered for LJ{self.optimizer.atoms}")
        plt.xlabel("Iteration")
        plt.ylabel("Energy")
        plt.show()

    def get_time(self) -> float:
        """
        Returns the time for the last run of the algorithm
        """
        return self.optimizer.execution_time

    def benchmark_run(self, indices: List[int], num_iterations: int) -> None:
        """
        TODO: Write this.
        """
        times = []
        convergence = []
        for lj in indices:
            self.optimizer.atoms = lj
            self.optimizer.run(num_iterations)

            best_cluster = self.optimizer.best_cluster()
            print(f"Best energy found: {self.optimizer.best_energy()}")
            print(
                f"Actual best energy is {get_cluster_energy(self.optimizer.atoms, self.optimizer.atom_type)}"
            )
            write("clusters/minima_optimized.xyz", best_cluster)

            times.append(self.optimizer.execution_time)
            convergence.append(self.optimizer.current_iteration)
            print(
                f"Time taken: {int(np.floor_divide(self.optimizer.execution_time, 60))} "
                f"min {int(self.optimizer.execution_time)%60} sec"
            )
            print(f"Stopped at {self.optimizer.current_iteration}")
            best_potentials = self.optimizer.potentials_history()
            plt.plot(best_potentials)
            plt.title(f"Execution on LJ{lj}")
            plt.xlabel("Iteration")
            plt.ylabel("Potential Energy")
            plt.show()

        for k in enumerate(indices):
            print(
                f"LJ {k[1]}: {convergence[k[0]]} iterations for "
                f"{int(np.floor_divide(times[k[0]], 60))} min {int(times[k[0]])%60} sec"
            )


bh = BasinHoppingOptimizer(local_optimizer=BFGS, atoms=13, atom_type="C")
mh = MinimaHoppingOptimizer(
    num_clusters=1,
    atoms=13,
    atom_type="C",
    temperature=300,
)
ga = GeneticAlgorithm(num_clusters=4, atoms=13)

b = Benchmark(bh)

b.benchmark_run([38], 1000)

print("---------------")
