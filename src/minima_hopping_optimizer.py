"""
Implementation of the Minima Hopping Algorithm
"""

from typing import List, Any
import numpy as np

from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.optimize import QuasiNewton
from src.global_optimizer import GlobalOptimizer


class MinimaHoppingOptimizer(GlobalOptimizer):
    """
    Implementation of the Minima Hopping Optimizer
    """

    def __init__(
        self,
        temperature: float = 300,
        e_diff: float = 1,
        mdmin: int = 4,
        atol: float = 5e-6,
        local_optimizer: Any = QuasiNewton,
        calculator: Any = LennardJones,
        debug: bool = False,
    ):
        super().__init__(local_optimizer, calculator, debug=debug)
        self.alpha_r = 1.02
        self.alpha_a = 1 / 1.02
        self.beta_s = 1.05
        self.beta_o = 1.05
        self.beta_n = 1 / 1.05
        self.temperature = temperature
        self.e_diff = e_diff
        self.mdmin = mdmin
        self.atol = atol
        self.m_cur = None
        self.minima_history: List[Atoms] = []

    def iteration(self) -> None:
        """
        Runs a single iteration of Minima hopping
        :return: None
        """
        self.utility.md(self.current_cluster, self.temperature, self.mdmin)
        self.current_cluster.set_momenta(  # type: ignore
            np.zeros(self.current_cluster.get_momenta().shape)  # type: ignore
        )

        with self.local_optimizer(self.current_cluster, logfile=self.logfile) as opt:
            opt.run()

        self.check_results()
        self.configs.append(self.current_cluster.copy())  # type: ignore
        self.potentials.append(self.current_cluster.get_potential_energy())  # type: ignore
        if self.current_cluster.get_potential_energy() < self.best_potential:  # type: ignore
            self.best_config = self.current_cluster
            self.best_potential = self.current_cluster.get_potential_energy()  # type: ignore
        # self.append_history()
        # print("Temperature: " + str(self.temperature))
        # print("Energy: " + str(cluster.get_potential_energy()))
        # print()

    def check_results(self) -> None:
        """
        Checks the outcome of a minima hopping run and changes temperature and e_diff variables depending on the outcome
        :param m: Minima we found
        :param cluster_index: Current index in the cluster list
        :return: None
        """
        if self.m_cur is not None:
            if self.utility.compare_clusters(
                self.m_cur, self.current_cluster, self.atol
            ):
                # print(self.m_cur[i].get_potential_energy())
                # print(m.get_potential_energy())
                # print("2 minima are the same")
                self.temperature *= self.beta_s
                return

            if (
                self.current_cluster.get_potential_energy()
                - self.m_cur.get_potential_energy()
                < self.e_diff
            ):  # Check if energy has decreased enough to be considered a new minimum
                # print("Energy between 2 minima has decreased enough")
                self.e_diff *= self.alpha_a
                self.m_cur = self.current_cluster.copy()
                self.m_cur.calc = self.calculator()
            else:
                self.e_diff *= self.alpha_r
        else:
            self.m_cur = self.current_cluster.copy()  # type: ignore
            self.m_cur.calc = self.calculator()  # type: ignore

        for (
            minima
        ) in (
            self.minima_history
        ):  # Is this a minima we've seen before? Change temperature accordingly
            if self.utility.compare_clusters(self.current_cluster, minima, self.atol):
                # print(minima.get_potential_energy())
                # print(m.get_potential_energy())
                # print("We've seen this minima before")
                self.temperature *= self.beta_o
                return

        # print("We've never seen this minima before")
        self.minima_history.append(self.current_cluster.copy())  # type: ignore
        self.minima_history[-1].calc = self.calculator()
        self.temperature *= self.beta_n

    def is_converged(self) -> bool:
        return False
