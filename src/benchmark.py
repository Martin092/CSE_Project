import time

import numpy as np
from ase.optimize import BFGS
from matplotlib import pyplot as plt

from src.global_optimizer import GlobalOptimizer
from src.basin_hopping_optimizer import BasinHoppingOptimizer
from src.oxford_database import get_cluster_energy

class Benchmark:
    def __init__(self, optimizer: GlobalOptimizer, max_iterations: int):
        self.optimizer = optimizer
        self.max_iterations = max_iterations
        self.run()

    def set_max_iterations(self, max_iterations: int):
        self.max_iterations = max_iterations
        self.run()

    def run(self):
        print(f"Running with {self.max_iterations} max iterations")
        self.optimizer.run(self.max_iterations)

    def compare_to_oxford(self):
        actual = get_cluster_energy(bh.atoms, bh.atom_type)
        energy, cluster = self.optimizer.best_energy(0)

        print(actual)
        print(energy)
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


bh = BasinHoppingOptimizer(local_optimizer=BFGS, atoms=13, atom_type="Fe")

benchmarker = Benchmark(bh, 600)

print(benchmarker.compare_to_oxford())
benchmarker.plot_energies()