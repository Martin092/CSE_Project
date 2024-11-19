from ase.units import fs
from ase.md.langevin import Langevin


class Disturber():

    # Class with all the methods to disturb a cluster

    def __init__(self, local_optimizer):
        self.local_optimizer = local_optimizer

    def random_setup(self, cluster):
        pass

    def angular_movement(self, cluster):
        pass

    def MD(self, cluster, temperature, number_of_steps):
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
