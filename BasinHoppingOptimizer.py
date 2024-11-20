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
            Disturber.random_step(cluster, self.boxLength)
            self.optimizers[index].run(fmax=0.2)
            self.history[index].append(cluster)

    def isConverged(self):
        pass

    def setup(self):
        pass


bh = BasinHoppingOptimizer(localOptimizer=BFGS, atoms=13, atom_type='Fe')
print(bh.boxLength)
write('clusters/basin_before.xyz', bh.clusterList[0])
bh.run(40)
write('clusters/basin_optimized.xyz', bh.clusterList[0])
