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
            self.optimizer.utility.big_jumps,
            [energies[i] for i in self.optimizer.utility.big_jumps],
            c="red",
        )
        plt.title(f"Execution on LJ{self.optimizer.num_atoms}")
        plt.xlabel("Iteration")
        plt.ylabel("Potential Energy")
        plt.savefig(f"../data/optimizer/LJ{self.optimizer.num_atoms}.png")

    def benchmark_run(
        self, indices: List[int], num_iterations: int, conv_iters: int = 10
    ) -> None:
        """
        Benchmark execution of Global Optimizer for LJ clusters.
        Measures the execution times, saves the best configurations history and plots the best potentials.
        :param indices: Cluster indices for LJ tests.
        :param num_iterations: Max number of iterations per execution.
        :param conv_iters: Number of iterations to be considered in the convergence criteria.
        :return: None.
        """
        times = []
        minima = []
        energy = []
        database = []
        convergence = []
        if not os.path.exists("../data/optimizer"):
            os.mkdir("../data")
            os.mkdir("../data/optimizer")
        for lj in indices:
            self.optimizer.run(lj, "C", num_iterations, conv_iters)

            best_cluster = self.optimizer.best_config
            best_cluster.center()  # type: ignore
            write(f"../data/optimizer/LJ{lj}.xyz", best_cluster)

            best = get_cluster_energy(lj, self.optimizer.atom_type)

            if (
                self.optimizer.best_potential > best
                and self.optimizer.best_potential - best < 0.001
            ):
                minima.append(1)
            elif self.optimizer.best_potential < best:
                minima.append(2)
            else:
                minima.append(3)

            self.optimizer.utility.write_trajectory(f"../data/optimizer/LJ{lj}.traj")

            times.append(self.optimizer.execution_time)
            convergence.append(self.optimizer.current_iteration)
            database.append(best)
            energy.append(self.optimizer.best_potential)

            self.plot_energies()

        for k in range(len(indices)):  # pylint: disable=C0200
            print(
                f"LJ {indices[k]}: {convergence[k]} iterations for "
                f"{int(np.floor_divide(times[k], 60))} min {int(times[k])%60} sec"
            )
            if minima[k] == 0:
                print(
                    f"Didn't find global minimum. Found {energy[k]}, but global minimum is {database[k]}."
                )
            elif minima[k] == 1:
                print("Found global minimum from database.")
            else:
                print(
                    f"Found new global minimum. Found {energy[k]}, but database minimum is {database[k]}."
                )
