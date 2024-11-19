from abc import ABC, abstractmethod
from ase import Atoms
import numpy as np

class GlobalOptimizer(ABC):

    def __init__(self, num_clusters: int, localOptimizer, atoms: int, atom_type: str, calculator):
        self.history = np.array([])
        self.clusterList = []
        self.localOptimizer = localOptimizer
        self.currentIteration = 0
        self.atoms = atoms
        self.covalentRadius = 1.0
        self.boxLength = 2*self.covalentRadius * (1/2 + ((3*self.atoms)/(4*np.pi*np.sqrt(2)))**(1/3))
        self.atom_type = atom_type
        self.calculator = calculator

        for i in range(num_clusters):
            positions = ( np.random.rand(self.atoms, 3) - 0.5 ) *  self.boxLength * 1.5 #1.5 is a magic number
            # In the future, instead of number of atoms, we ask the user to choose how many atoms they want for each atom type.
            clus = Atoms(self.atom_type + str(self.atoms), positions=positions)
            clus.calc = calculator()
            self.clusterList.append(clus)
        # Disturber

    @abstractmethod
    def iteration(self):
        pass

    @abstractmethod
    def isConverged(self):
        pass

    @abstractmethod
    def setup(self):
        pass

    def run(self, maxIterations):
        self.setup()

        while self.currentIteration < maxIterations or not self.isConverged():
            self.iteration()
