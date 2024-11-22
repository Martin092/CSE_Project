from GlobalOptimizer import GlobalOptimizer
from Disturber import Disturber


class MinimaHoppingOptimizer(GlobalOptimizer):

    def __init__(self, clusters: int, localOptimizer, atoms: int,
                 alpha_r: float, alpha_a: float,
                 beta_S: float, beta_O: float, beta_N: float,
                 temperature: float, E_diff: float
                 ):
        super(MinimaHoppingOptimizer, self).__init__(clusters, localOptimizer, atoms)
        self.alpha_r = alpha_r
        self.alpha_a = alpha_a
        self.beta_S = beta_S
        self.beta_O = beta_O
        self.beta_N = beta_N
        self.temperature = temperature
        self.E_diff = E_diff
        self.disturber = Disturber()
        self.m_cur = None


    def iteration(self):
        for cluster in self.clusters:
            # TODO: Initialize velocities of atoms based on Maxwell Boltzmann
            self.disturber.md(cluster, self.temperature) #TODO: Change MD run to run until we find some number of minima, this number will be a hyperparameter
            #TODO: Locally Optimize #self.localOptimizer
            self.localOptimizer
            #TODO: Check results, edit temperature and E_diff accordingly, see reference code and flowchart



