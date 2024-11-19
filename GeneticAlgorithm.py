from GlobalOptimizer import GlobalOptimizer
from ase.optimize import BFGS
from ase.calculators.lj import LennardJones
from Disturber import Disturber


class GeneticAlgorithm(GlobalOptimizer):
    def __init__(self, localOptimizer=BFGS, atoms=30, atom_type='C', calculator=LennardJones, num_clusters=16):
        super().__init__(num_clusters=num_clusters,
                         localOptimizer=localOptimizer,
                         atoms=atoms,
                         atom_type=atom_type,
                         calculator=calculator)

    def iteration(self):
        pass

    def isConverged(self):
        pass

    def setup(self):
        pass

    @staticmethod
    def crossover(cluster1, cluster2):
        Disturber.align_cluster(cluster1)
        Disturber.align_cluster(cluster2)
        group11, group12, _ = Disturber.split_cluster(cluster1)
        group21, group22, _ = Disturber.split_cluster(cluster2)
        return len(group11) + len(group22), len(group12) + len(group21)
