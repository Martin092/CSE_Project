"""
A class that provides utility methods for easily
comparing the performance of two global optimization algorithms
"""

from typing import List
import os

import numpy as np
from ase.io import write
from matplotlib import pyplot as plt

from src.global_optimizer import GlobalOptimizer
from auxiliary.oxford_database import get_cluster_energy


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
        actual = get_cluster_energy(self.optimizer.atoms, self.optimizer.atom_type)
        energy = self.optimizer.best_energy()
        return actual - energy

    def plot_energies(self) -> None:
        """
        Plots the energy values over the course of the entire run
        """
        energies: list[float] = []
        for clus in self.optimizer.history[0]:
            clus.calc = self.optimizer.calculator()
            energies.append(clus.get_potential_energy())

        energies = self.optimizer.potentials_history()
        plt.plot(energies)
        plt.scatter(
            self.optimizer.utility.big_jumps,
            [energies[i] for i in self.optimizer.utility.big_jumps],
            c="red",
        )
        plt.title(f"Execution on LJ{self.optimizer.atoms}")
        plt.xlabel("Iteration")
        plt.ylabel("Potential Energy")
        plt.show()

    def get_time(self) -> float:
        """
        Returns the time for the last run of the algorithm
        """
        return self.optimizer.execution_time

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
        if not os.path.exists("../data/optimizer"):
            os.mkdir("../data")
            os.mkdir("../data/optimizer")
        for lj in indices:
            self.optimizer.atoms = lj
            self.optimizer.run(num_iterations, conv_iters)

            best_cluster = self.optimizer.best_cluster()
            print(f"Best energy found: {self.optimizer.best_energy()}")
            write(f"../data/optimizer/LJ{lj}.xyz", best_cluster)

            best = get_cluster_energy(lj, self.optimizer.atom_type)

            if (
                self.optimizer.best_energy() > best
                and self.optimizer.best_energy() - best < 0.001
            ):
                print("Best energy matches the database")
            elif self.optimizer.best_energy() < best:
                print("GROUNDBREAKING!!!")
            else:
                print(f"Best energy in database is {best}.")

            self.optimizer.write_trajectory(f"../data/optimizer/LJ{lj}.traj")

            times.append(self.optimizer.execution_time)
            convergence.append(self.optimizer.current_iteration)
            print(
                f"Time taken: {int(np.floor_divide(self.optimizer.execution_time, 60))} "
                f"min {int(self.optimizer.execution_time)%60} sec"
            )
            print(f"Stopped at {self.optimizer.current_iteration}")
            if len(self.optimizer.utility.big_jumps) != 0:
                print(f"Big jumps were made at {self.optimizer.utility.big_jumps}")

            self.plot_energies()

        for k in enumerate(indices):
            print(
                f"LJ {k[1]}: {convergence[k[0]]} iterations for "
                f"{int(np.floor_divide(times[k[0]], 60))} min {int(times[k[0]])%60} sec"
            )
