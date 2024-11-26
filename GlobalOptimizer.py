from abc import ABC, abstractmethod
from ase import Atoms
import numpy as np
from Disturber import Disturber
from ase.io import write, Trajectory
from AtomParameters import lj_parameters


class GlobalOptimizer(ABC):

    def __init__(self, num_clusters: int, local_optimizer, atoms: int, atom_type: str, calculator):
        self.history = []
        self.clusterList = []
        self.optimizers = []
        self.local_optimizer = local_optimizer
        self.currentIteration = 0
        self.num_clusters = num_clusters
        self.atoms = atoms
        self.covalentRadius = 1.0
        self.boxLength = 2 * self.covalentRadius * (1/2 + ((3.0 * self.atoms) / (4 * np.pi * np.sqrt(2)))**(1/3)) #More restrictive
        self.atom_type = atom_type
        self.calculator = calculator
        self.disturber = Disturber(self)
        self.sigma,self.epsilon = self.setup()
        #self.boxLength = self.sigma * np.sqrt(self.atoms) #More permissive (assumes plane packing)

    @abstractmethod
    def iteration(self):
        pass

    @abstractmethod
    def is_converged(self):
        pass

    def setup(self):
        self.currentIteration = 0
        self.history = []
        self.clusterList = []
        self.optimizers = []
        for i in range(self.num_clusters):
            positions = (np.random.rand(self.atoms, 3) - 0.5) * self.boxLength * 1.5  # 1.5 is a magic number
            # In the future, instead of number of atoms,
            # we ask the user to choose how many atoms they want for each atom type.
            clus = Atoms(self.atom_type + str(self.atoms), positions=positions)
            clus.calc = self.calculator()
            self.clusterList.append(clus)
            opt = self.local_optimizer(clus, logfile='log.txt')
            self.optimizers.append(opt)
        sigma = lj_parameters[self.atom_type]["sigma"]
        epsilon = lj_parameters[self.atom_type]["epsilon"]   
        return sigma, epsilon
    

    def run(self, max_iterations):
        self.setup()

        while self.currentIteration < max_iterations and not self.is_converged():
            self.history.append([])
            print(self.currentIteration)
            self.iteration()
            self.currentIteration += 1

    def write_to_file(self, filename: str, cluster_index=0):
        """
        Writes the cluster to a .xyz file.
        :param filename: the name of the file, does not matter if it has the .xyz extension
        :param cluster_index: which cluster will be written
        """
        filename = filename if filename[-4:] == ".xyz" else filename + ".xyz"
        write(f'clusters/{filename}', self.clusterList[cluster_index])

    def write_trajectory(self, filename: str, cluster_index=0):
        with Trajectory(filename, mode='w') as traj:
            for cluster in self.history[cluster_index]:
                traj.write(cluster)

    def append_history(self):
        """
        Appends copies of all the clusters in the clusterList to the history.
        Copies are used since clusters are passed by reference
        :return:
        """
        for i, cluster in enumerate(self.clusterList):
            self.history[i].append(cluster.copy())

    def compare_clusters(self, cluster1, cluster2):
        """
        Checks whether two clusters are equal based on their potential energy.
        This method may be changed in the future to use more sophisticated methods,
        such as overlap matrix fingerprint thresholding.
        :param cluster1: First cluster
        :param cluster2: Second cluster
        :return: boolean
        """
        return np.isclose(cluster1.get_potential_energy(), cluster2.get_potential_energy())

    def get_best_cluster_found(self, cluster_index=0):
        # TODO Make this work for multiple clusters
        min_energy = float('inf')
        best_cluster = None
        for cluster in self.history[cluster_index]:
            cluster.calc = self.calculator()
            curr_energy = cluster.get_potential_energy()
            if curr_energy < min_energy:
                min_energy = curr_energy
                best_cluster = cluster

        return best_cluster