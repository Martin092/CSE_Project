from abc import ABC, abstractmethod
import numpy as np

class GlobalOptimizer(ABC):

    def __init__(self, clusters, localOptimizer):
        self.history = np.array([])
        self.clusters = clusters
        self.localOptimizer = localOptimizer
        self.currentIteration = 0
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
