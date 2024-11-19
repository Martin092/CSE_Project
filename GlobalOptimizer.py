from abc import ABC, abstractmethod
import numpy as np

class GlobalOptimizer(ABC):
    def __init__(self, clusters, localOptimizer):
        self.history = np.array([])
        self.clusters = clusters
        self.localOptimizer = localOptimizer
        self.currentIteration = 0
        # Disturber
