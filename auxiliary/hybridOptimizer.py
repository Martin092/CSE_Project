import sys
import time
from ase import Atoms
from ase.optimize import FIRE
from ase.io import write
import numpy as np
from typing import Any
from ase.calculators.lj import LennardJones

sys.path.append("./")

from src.global_optimizer import GlobalOptimizer
from src.basin_hopping_optimizer import BasinHoppingOptimizer
from src.minima_hopping_optimizer import MinimaHoppingOptimizer
from src.genetic_algorithm import GeneticAlgorithm
from auxiliary.oxford_database import get_cluster_energy

# define some new optimizer method, using the existing methods

class HybridOptimizer(GlobalOptimizer):
    def __init__(
        self,
        atoms: int,
        atom_type: str,
        local_optimizer: Any = FIRE,
        calculator: Any = LennardJones,
        num_clusters: int = 1,
        temperature: float = 300,

    ) -> None:

        super().__init__(
            num_clusters=num_clusters,
            local_optimizer=local_optimizer,
            atoms=atoms,
            atom_type=atom_type,
            calculator=calculator
        )

        self.temperature = temperature

        self.minima_optimizer = MinimaHoppingOptimizer(           
            num_clusters=num_clusters,
            local_optimizer=local_optimizer,
            atoms=atoms,
            atom_type=atom_type,
            calculator=calculator,
            temperature=temperature
        )
        self.basin_optimizer = BasinHoppingOptimizer(           
            num_clusters=num_clusters,
            local_optimizer=local_optimizer,
            atoms=atoms,
            atom_type=atom_type,
            calculator=calculator
        )
        # self.genetic_algorithm = GeneticAlgorithm()


    def is_converged(self, conv_iters: int = 10) -> bool:
        """
        Checks if convergence criteria is satisfied. The convergence check of basin hopping is inherited.
        :param conv_iters: Number of iterations to be considered.
        :return: True if convergence criteria is met, otherwise False.
        """
        return BasinHoppingOptimizer.is_converged(self, conv_iters)


    def iteration(self) -> None:
        """
        Performs a test iteration, using both basin and minima hopping iterations
        :return: None.
        """

        print(f"Iteration {self.current_iteration}")

        if (self.current_iteration + 1) % 10 != 0:
            self.minima_optimizer = MinimaHoppingOptimizer(           
                num_clusters=self.num_clusters,
                local_optimizer=self.local_optimizer,
                atoms=self.atoms,
                atom_type=self.atom_type,
                calculator=self.calculator,
                temperature=self.temperature
            )
            self.minima_optimizer.setup()
            # self.minima_optimizer.cluster_list[0] = self.cluster_list[-1].copy()

        if (self.current_iteration + 1) % 10 != 0:
            self.minima_optimizer.iteration()

            self.cluster_list = self.minima_optimizer.cluster_list.copy()

        else:
            self.basin_optimizer = BasinHoppingOptimizer(           
                num_clusters=self.num_clusters,
                local_optimizer=self.local_optimizer,
                atoms=self.atoms,
                atom_type=self.atom_type,
                calculator=self.calculator
            )
            self.basin_optimizer.setup()
            # self.basin_optimizer.cluster_list[0] = self.cluster_list[-1].copy()

            self.basin_optimizer.iteration()

            self.cluster_list = self.basin_optimizer.cluster_list.copy()

        for index, clus in enumerate(self.cluster_list):
            self.history[index].append(clus.copy())


if __name__ == "__main__":
    hybridMethod = HybridOptimizer(atoms=13, atom_type="Fe", temperature=300)

    hybridMethod.run(100)

    energy, cluster = hybridMethod.best_energy_cluster()
    print(f"Result: {energy}")
    print(f"Actual: {get_cluster_energy(hybridMethod.atoms, hybridMethod.atom_type)}")
