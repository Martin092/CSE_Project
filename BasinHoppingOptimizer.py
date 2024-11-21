from GlobalOptimizer import GlobalOptimizer
from ase.optimize import BFGS
from ase.calculators.lj import LennardJones
import numpy as np
from ase.io import write
from Disturber import Disturber

class BasinHoppingOptimizer(GlobalOptimizer):
    def __init__(self, localOptimizer, atoms, atom_type, calculator=LennardJones, num_clusters=1):
        super().__init__(num_clusters=num_clusters, localOptimizer=localOptimizer, atoms=atoms, atom_type=atom_type, calculator=calculator)

    def iteration(self):
        for index, cluster in enumerate(self.clusterList):
            self.disturber.random_step(cluster, self.boxLength)

            dists = self.clusterList[0].get_potential_energies()
            print(np.max(dists))

            self.optimizers[index].run(fmax=0.4)
            self.history[index].append(cluster)

    def isConverged(self):
        if self.currentIteration < 2:
            return False

        # TODO this takes in only one cluster into account, use all of them
        current = self.clusterList[0].get_potential_energy()
        self.history[0][len(self.history) - 1].calc = self.calculator()
        last = self.history[0][len(self.history[0]) - 1].get_potential_energy()

        print(f"hist: {len(self.history[0])}")
        print(f"current: {current}; last: {last}")
        print(f"diff: {abs(current - last)}")
        return abs(current - last) < 0.2

    def setup(self):
        pass


bh = BasinHoppingOptimizer(localOptimizer=BFGS, atoms=19, atom_type='Fe')
print(bh.boxLength)
write('clusters/basin_before.xyz', bh.clusterList[0])
bh.run(100000)
write('clusters/basin_optimized.xyz', bh.clusterList[0])
