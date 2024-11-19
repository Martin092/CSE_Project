from GlobalOptimizer import GlobalOptimizer
from ase.optimize import BFGS
from ase.calculators.lj import LennardJones

class BasinHoppingOptimizer(GlobalOptimizer):
    def __init__(self, localOptimizer, atoms, atom_type, calculator=LennardJones, num_clusters=1):
        super().__init__(num_clusters=num_clusters, localOptimizer=localOptimizer, atoms=atoms, atom_type=atom_type, calculator=calculator)

    def iteration(self):
        pass

    def isConverged(self):
        pass

    def setup(self):
        pass


BasinHoppingOptimizer(localOptimizer=BFGS, atoms=13, atom_type='Fe')

