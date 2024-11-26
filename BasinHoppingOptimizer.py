from GlobalOptimizer import GlobalOptimizer
from ase.optimize import BFGS
from ase.calculators.lj import LennardJones
import numpy as np
from ase.io import write
from Disturber import Disturber

class BasinHoppingOptimizer(GlobalOptimizer):
    def __init__(self, local_optimizer, atoms, atom_type, calculator=LennardJones, num_clusters=1):
        super().__init__(num_clusters=num_clusters, local_optimizer=local_optimizer, atoms=atoms, atom_type=atom_type, calculator=calculator)
        self.last_energy = float('inf')

    def iteration(self):
        if self.iteration == 0:
            self.last_energy = self.clusterList[0].get_potential_energy()

        for index, cluster in enumerate(self.clusterList):
            self.last_energy = self.clusterList[index].get_potential_energy()

            energies = self.clusterList[index].get_potential_energies()
            min_energy = min(energies)
            max_energy = max(energies)

            # self.disturber.random_step(cluster)
            if abs(min_energy - max_energy) < 1.5:
                self.disturber.random_step(cluster)
            else:
                self.disturber.angular_movement(cluster)

            self.optimizers[index].run(fmax=0.2)
            self.history[index].append(cluster)

    def is_converged(self):
        if self.currentIteration < 2:
            return False

        # TODO this takes in only one cluster into account, use all of them
        current = self.clusterList[0].get_potential_energy()
        return abs(current - self.last_energy) < 2e-6



bh = BasinHoppingOptimizer(local_optimizer=BFGS, atoms=13, atom_type='Fe')
print(bh.boxLength)

bh.run(1000)

min_energy = float('inf')
best_cluster = None
for cluster in bh.history[0]:
    cluster.calc = bh.calculator()
    curr_energy = cluster.get_potential_energy()
    if curr_energy < min_energy:
        min_energy = curr_energy
        best_cluster = cluster

print(min_energy)

write('clusters/basin_optimized.xyz', best_cluster)
