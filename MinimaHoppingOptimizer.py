from ase.optimize.minimahopping import ComparePositions
from GlobalOptimizer import GlobalOptimizer
from Disturber import Disturber
import numpy as np


class MinimaHoppingOptimizer(GlobalOptimizer):

    def __init__(self, clusters: int, localOptimizer, atoms: int, atom_type: str, calculator,
                 alpha_r: float = 1.02, alpha_a: float = 1 / 1.02,
                 beta_S: float = 1.05, beta_O: float = 1.05, beta_N: float = 1 / 1.05,
                 temperature: float, E_diff: float, mdmin: int
                 ):
        super(MinimaHoppingOptimizer, self).__init__(clusters, localOptimizer, atoms, atom_type, calculator)
        self.alpha_r = alpha_r
        self.alpha_a = alpha_a
        self.beta_S = beta_S
        self.beta_O = beta_O
        self.beta_N = beta_N
        self.temperature = temperature
        self.E_diff = E_diff
        self.mdmin = mdmin
        self.m_cur = [None] * clusters
        self.minima_history = []

    def iteration(self):
        """
        Runs a single iteration of Minima hopping
        :return:
        """
        for i, cluster in enumerate(self.clusterList):
            self.disturber.md(cluster, self.temperature, self.mdmin)
            cluster.set_momenta(np.zeros(cluster.get_momenta().shape))

            with self.localOptimizer(cluster, logfile='log.txt') as opt:
                opt.run(fmax=0.02)

            self.check_results(cluster, i)

    def check_results(self, m, i):
        """
        Checks the outcome of a minima hopping run and changes temperature and E_diff variables depending on the outocme
        :param m: Minima we found
        :param i: Current index in the cluster list
        :return:
        """
        if self.m_cur[i] is not None:  # Case 1: We did not find a new minimum
            if self.compare_clusters(self.m_cur[i], m):
                self.temperature *= self.beta_S
                return

        if m.get_potential_energy() - self.m_cur[
            i].get_potential_energy() < self.E_diff:  # Check if energy has decreased enough to be considered a new minimum
            self.E_diff *= self.alpha_a
            self.m_cur = m
        else:
            self.E_diff *= self.alpha_r

        if m in self.minima_history:  # Is this a minima we've seen before? Change temperature accordingly
            self.temperature *= self.beta_O
        else:
            self.minima_history.append(m)
            self.temperature *= self.beta_N

