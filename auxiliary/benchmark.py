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
from auxiliary.cambridge_database import get_cluster_energy


class Benchmark:
    """
    Class that provides utility methods for easily comparing
    the performance of global optimization algorithms
    """

    def __init__(self, optimizer: GlobalOptimizer):
        self.optimizer = optimizer

    def plot_energies(self) -> None:
        """
        Plots the energy values over the course of the entire run
        """
        energies = self.optimizer.potentials
        plt.plot(energies)
        plt.scatter(
            self.optimizer.utility.big_jumps,  # type: ignore
            [energies[i] for i in self.optimizer.utility.big_jumps],  # type: ignore
            c="red",
        )
        plt.title(f"Execution on LJ{self.optimizer.num_atoms}")
        plt.xlabel("Iteration")
        plt.ylabel("Potential Energy")
        plt.savefig(f"../data/optimizer/LJ{self.optimizer.num_atoms}.png")
        plt.show()

    def benchmark_run(self, indices: List[int], num_iterations: int) -> None:
        """
        Benchmark execution of Genetic Algorithm.
        Measures the execution times, saves the best configurations history and plots the best potentials.
        :param indices: Cluster indices for LJ tests.
        :param num_iterations: Max number of iterations per execution.
        :return: None.
        """
        times = []
        convergence = []
        if not os.path.exists("../data/optimizer"):
            os.mkdir("../data")
            os.mkdir("../data/optimizer")
        for lj in indices:
            self.optimizer.run(lj, "C", num_iterations)

            best_cluster = self.optimizer.best_config
            print(f"Best energy found: {self.optimizer.best_potential}")
            best_cluster.center()  # type: ignore
            write(f"../data/optimizer/LJ{lj}.xyz", best_cluster)  # type: ignore

            best = get_cluster_energy(lj, self.optimizer.atom_type)

            if (
                self.optimizer.best_potential > best
                and self.optimizer.best_potential - best < 0.001
            ):
                print("Best energy matches the database!")
            elif self.optimizer.best_potential < best:
                print("GROUNDBREAKING!!!")
            else:
                print(f"Suboptimal. Best energy in database is {best}.")

            self.optimizer.write_trajectory(f"../data/optimizer/LJ{lj}.traj")

            times.append(self.optimizer.execution_time)
            convergence.append(self.optimizer.current_iteration)
            print(
                f"Time taken: {int(np.floor_divide(self.optimizer.execution_time, 60))} "
                f"min {int(self.optimizer.execution_time)%60} sec"
            )
            print(f"Stopped/Converged at iteration {self.optimizer.current_iteration}.")
            if len(self.optimizer.utility.big_jumps) != 0:  # type: ignore
                print(f"Big jumps were made at {self.optimizer.utility.big_jumps}")  # type: ignore

            self.plot_energies()

        for k in enumerate(indices):
            print(
                f"LJ {k[1]}: {convergence[k[0]]} iterations for "
                f"{int(np.floor_divide(times[k[0]], 60))} min {int(times[k[0]])%60} sec"
            )
