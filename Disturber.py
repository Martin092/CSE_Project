import numpy as np
from ase.units import fs
from ase.md.langevin import Langevin


class Disturber:

    # Class with all the methods to disturb a cluster

    def __init__(self, local_optimizer):
        self.local_optimizer = local_optimizer

    def random_setup(self, cluster):
        pass

    def angular_movement(self, cluster):
        pass

    def md(self, cluster, temperature, number_of_steps):
        """
        Perform a Molecular Dynamics run using Langevin Dynamics
        :param cluster: Cluster of atoms
        :param temperature: Temperature in Kelvin
        :param number_of_steps: Number of steps to use in Molecular Dynamics
        """
        dyn = Langevin(
            cluster,
            timestep=5.0 * fs, # Feel free to mess with this parameter
            temperature_K=temperature,
            friction=0.5 / fs,  # Feel free to mess with this parameter
        )

        dyn.run(number_of_steps)

    def twist(self, cluster):
        pass

    def etching(self, cluster):
        pass

    @staticmethod
    def split_cluster(cluster, p1=np.random.rand(3), p2=np.random.rand(3), p3=np.random.rand(3)):
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        d = - np.dot(normal, p1)
        group1 = []
        group2 = []
        for atom in cluster:
            val = np.dot(normal, atom.position) + d
            if val > 0:
                group1.append(atom)
            else:
                group2.append(atom)
        return group1, group2
