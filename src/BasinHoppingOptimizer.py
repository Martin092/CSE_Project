from GlobalOptimizer import GlobalOptimizer
from ase.optimize import BFGS
from ase.calculators.lj import LennardJones
import numpy as np
from ase.io import write
from typing import Any
from Disturber import Disturber


class BasinHoppingOptimizer(GlobalOptimizer):
    def __init__(
        self, localOptimizer: Any, atoms: int, atom_type: str, calculator: Any = LennardJones, num_clusters: int = 1
    ) -> None:
        super().__init__(  # type: ignore
            num_clusters=num_clusters,
            localOptimizer=localOptimizer,
            atoms=atoms,
            atom_type=atom_type,
            calculator=calculator,
        )
        self.last_energy = self.clusterList[0].get_potential_energy()

    def iteration(self):  # type: ignore
        for index, cluster in enumerate(self.clusterList):
            self.last_energy = self.clusterList[index].get_potential_energy()

            energies = self.clusterList[index].get_potential_energies()
            min_energy = min(energies)
            max_energy = max(energies)

            # self.disturber.random_step(cluster)
            if abs(min_energy - max_energy) < 1.5:
                self.disturber.random_step(cluster)  # type: ignore
            else:
                self.disturber.angular_movement(cluster)

            self.optimizers[index].run(fmax=0.2)
            self.history[index].append(cluster)

    def isConverged(self):  # type: ignore
        if self.currentIteration < 2:
            return False

        # TODO this takes in only one cluster into account, use all of them
        current = self.clusterList[0].get_potential_energy()
        return abs(current - self.last_energy) < 2e-6

    def setup(self):  # type: ignore
        pass


bh = BasinHoppingOptimizer(localOptimizer=BFGS, atoms=13, atom_type="Fe")  # type: ignore
print(bh.boxLength)
write("clusters/basin_before.xyz", bh.clusterList[0])
bh.run(1000)

min_energy = float("inf")
best_cluster = None
for cluster in bh.history[0]:
    cluster.calc = bh.calculator()
    curr_energy = cluster.get_potential_energy()
    if curr_energy < min_energy:
        min_energy = curr_energy
        best_cluster = cluster

print(min_energy)

write("clusters/basin_optimized.xyz", best_cluster)  # type: ignore
