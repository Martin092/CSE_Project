import numpy as np
from GlobalOptimizer import GlobalOptimizer
from ase import Atoms
from ase.optimize import BFGS
from ase.calculators.lj import LennardJones


class GeneticAlgorithm(GlobalOptimizer):
    def __init__(self, mutation_probability: float = 0.2, localOptimizer=BFGS, atoms=30,
                 atom_type='C', calculator=LennardJones, num_clusters=16):
        super().__init__(num_clusters=num_clusters,
                         localOptimizer=localOptimizer,
                         atoms=atoms,
                         atom_type=atom_type,
                         calculator=calculator)
        self.mutation_probability = mutation_probability
        self.potentials = []

    def iteration(self):
        self.history.append([])
        for index, cluster in enumerate(self.clusterList):
            self.optimizers[index].run(fmax=0.1)  # Local optimization
            self.history[self.currentIteration].append(cluster)  # Save local minima
            self.potentials.append(cluster.get_potential_energy())  # Compute potential energy
        pairs = list(zip(self.potentials, self.clusterList, self.optimizers))  # Zip clusters, potentials and optimizers
        pairs.sort(key=lambda x: x[0])  # Sort clusters on potentials
        midpoint = (len(pairs) + 1) // 2  # Determine the number of clusters to be selected
        self.clusterList = [pair[1] for pair in pairs[:midpoint]]  # Update current clusters to contain only selected
        self.optimizers = [pair[2] for pair in pairs[:midpoint]]  # Update current optimizers to contain only selected
        crossover = []  # List of children atomic positions
        while len(crossover) + len(self.clusterList) < self.num_clusters:
            i = np.random.randint(0, len(self.clusterList))
            parent1 = self.clusterList[i]
            i = np.random.randint(0, len(self.clusterList))
            parent2 = self.clusterList[i]
            group1, group2 = self.crossover(parent1, parent2)  # generate children
            child1 = []
            for atom in group1:
                child1.append(atom.position)
            crossover.append(child1)
            if len(self.clusterList) + len(crossover) != self.num_clusters:
                child2 = []
                for atom in group2:
                    child2.append(atom.position)
                crossover.append(child2)
        for child in crossover:  # Add children to cluster list
            clus = Atoms(self.atom_type + str(self.atoms), positions=child)
            clus.calc = self.calculator()
            self.clusterList.append(clus)
            opt = self.localOptimizer(clus, logfile='log.txt')
            self.optimizers.append(opt)
        for cluster in self.clusterList:
            if np.random.rand() <= self.mutation_probability:
                self.disturber.twist(cluster)
            for i in range(self.atoms):
                if np.random.rand() <= self.mutation_probability/self.atoms:
                    cluster.positions[i] += (np.random.rand(3) - 0.5)

    def isConverged(self):
        return False

    def crossover(self, cluster1, cluster2):
        self.disturber.align_cluster(cluster1)
        self.disturber.align_cluster(cluster2)
        group11 = []
        group12 = []
        group21 = []
        group22 = []
        while len(group11)+len(group22) != len(cluster1.positions):
            p1 = np.random.rand(3)
            p2 = np.random.rand(3)
            p3 = np.random.rand(3)
            group11, group12, _ = self.disturber.split_cluster(cluster1, p1, p2, p3)
            group21, group22, _ = self.disturber.split_cluster(cluster2, p1, p2, p3)
        return group11 + group22, group12 + group21
