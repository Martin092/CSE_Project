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
