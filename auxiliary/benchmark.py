"""
A class that provides utility methods for easily
comparing the performance of two global optimization algorithms
"""

from typing import List
import os

import numpy as np
from ase.io import write
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator

from src.global_optimizer import GlobalOptimizer
from src.basin_hopping_optimizer import BasinHoppingOptimizer
from src.genetic_algorithm import GeneticAlgorithm
from auxiliary.cambridge_database import get_cluster_energy


class Benchmark:
    """
    Class that provides utility methods for easily comparing
    the performance of global optimization algorithms
    """

    def __init__(self, optimizer: GlobalOptimizer, log: str | None = None):
        self.optimizer = optimizer
        self.log = log
        self.algorithm = ""
        if isinstance(self.optimizer, BasinHoppingOptimizer):
            self.algorithm = "bh_seq"
        elif isinstance(self.optimizer, GeneticAlgorithm):
            self.algorithm = "ga_seq"

    def plot_energies(self) -> None:
        """
        Plots the energy values over the course of the entire run
        """
        energies = self.optimizer.potentials
        plt.plot(energies)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
        ticks = list(plt.gca().get_xticks())[1:-1]
        if ticks[-1] != len(energies) - 1:
            ticks.append(len(energies) - 1)
        plt.gca().set_xticks(ticks)
        if isinstance(self.optimizer, BasinHoppingOptimizer):
            plt.scatter(
                self.optimizer.utility.big_jumps,
                [energies[i] for i in self.optimizer.utility.big_jumps],
                c="red",
            )
        plt.title(f"Execution on LJ{self.optimizer.utility.num_atoms}")
        plt.xlabel("Iteration")
        plt.ylabel("Potential Energy")
        plt.tight_layout()
        if self.optimizer.comm is None:
            plt.savefig(
                f"../data/optimizer/LJ{self.optimizer.utility.num_atoms}_{self.algorithm}.png"
            )
        else:
            plt.savefig(
                f"./data/optimizer/LJ{self.optimizer.utility.num_atoms}_{self.algorithm}.png"
            )
        plt.show()
        plt.close()

    def benchmark_run(
        self,
        indices: List[int],
        num_iterations: int,
        conv_iterations: int = 10,
        seed: int | None = None,
    ) -> None:
        """
        Benchmark execution of Global Optimizer for LJ clusters.
        Measures the execution times, saves the best configurations history and plots the best potentials.
        :param indices: Cluster indices for LJ tests.
        :param num_iterations: Max number of iterations per execution.
        :param conv_iterations: Number of iterations to be considered in the convergence criteria.
        :param seed: Seed for which to perform the benchmark
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
            self.optimizer.run(
                "C" + str(lj), num_iterations, conv_iterations, seed=seed
            )

            best_cluster = self.optimizer.best_config
            best_cluster.center()  # type: ignore
            write(f"../data/optimizer/LJ{lj}_{self.algorithm}.xyz", best_cluster)

            best = get_cluster_energy(lj)

            if abs(self.optimizer.best_potential - best) < 0.000001 * lj * lj:
                minima.append(1)
            elif self.optimizer.best_potential < best:
                minima.append(2)
            else:
                minima.append(0)

            self.optimizer.utility.write_trajectory(
                f"../data/optimizer/LJ{lj}_{self.algorithm}.traj"
            )

            times.append(self.optimizer.execution_time)
            convergence.append(self.optimizer.current_iteration)
            database.append(best)
            energy.append(self.optimizer.best_potential)

            self.plot_energies()

        for k in range(len(indices)):  # pylint: disable=C0200
            self.print_log(
                f"LJ {indices[k]}: {convergence[k] - 1} iterations for "
                + f"{int(np.floor_divide(times[k], 60))} min {int(times[k])%60} sec"
            )
            if minima[k] == 0:
                self.print_log(
                    f"Didn't find global minimum. Found {energy[k]}, but global minimum is {database[k]}."
                )
            elif minima[k] == 1:
                self.print_log("Found global minimum from database.")
            else:
                self.print_log(
                    f"Found new global minimum. Found {energy[k]}, but database minimum is {database[k]}."
                )

            self.print_log("\n")

    def print_log(self, content: str) -> None:
        """
        Prints the contents to a file or to the console
        :param content: The string that should be printed
        """
        if self.log is None:
            print(content)
        else:
            with open(self.log, "a", encoding="utf-8") as f:
                f.write(content + "\n")
                f.flush()
