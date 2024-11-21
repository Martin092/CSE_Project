import numpy as np
from ase.units import fs
from ase.md.langevin import Langevin
from sklearn.decomposition import PCA
from reference_code.rotation_matrices import rotation_matrix
from ase import Atoms
from ase.optimize.minimahopping import PassedMinimum
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import time


class Disturber:

    # Class with all the methods to disturb a cluster

    def __init__(self, local_optimizer, global_optimizer):
        self.local_optimizer = local_optimizer
        self.global_optimizer = global_optimizer


    def random_step(self, cluster, box_size):
        """
        Moves the highest energy atom in a random direction
        :param cluster: the cluster we want to disturb
        :param box_size the size of the container containing the atoms
        :return: result is written directly to cluster, nothing is returned
        """
        energies = cluster.get_potential_energies()
        highest_index = np.argmax(energies)

        # random step from -1 to 1
        stepSize = (np.random.rand(3) - 0.5) * 2
        for i in range(15):
            step = (np.random.rand(3) - 0.5) * 2 * stepSize
            cluster.positions[highest_index] += step

            # might be good to take this out of here as it does another pass over all atoms
            cluster.positions = np.clip(cluster.positions, -box_size, box_size)


    def check_position(self, cluster, atom):
        if np.linalg.norm(atom.position) > self.global_optimizer.boxLength:
            return False
        
        for other_atom in cluster:
            if np.linalg.norm(atom.position - other_atom.position) < 0.5 * self.global_optimizer.covalentRadius:
                return False
        
        return True
    
    def check_position(self, group_static, group_moved):
        for atom in group_moved:
            if np.linalg.norm(atom.position) > self.global_optimizer.boxLength:
                return False
    
            for other_atom in group_static:
                if np.linalg.norm(atom.position - other_atom.position) < 0.5 * self.global_optimizer.covalentRadius:
                    return False
        
        return True
            

    def angular_movement(self, cluster):
        vector = np.random.rand(3)
        atom = cluster[np.random.randint(0, len(cluster))]
        angle = np.random.uniform(0, 2 * np.pi)
        atom.position = np.dot(rotation_matrix(vector, angle), atom.position)
        #Check if position is valid: check_position(self, cluster, atom)
        return cluster

    @staticmethod
    def md(cluster, temperature, mdmin, seed=int(time.time())):
        """
        Perform a Molecular Dynamics run using Langevin Dynamics
        :param cluster: Cluster of atoms
        :param temperature: Temperature in Kelvin
        :param mdmin: Number of minima to be found before MD run halts. Alternatively it will halt once we reach 10000 iterations
        """
        dyn = Langevin(
            cluster,
            timestep=5.0 * fs,  # Feel free to mess with this parameter
            temperature_K=temperature,
            friction=0.5 / fs,  # Feel free to mess with this parameter
            rng = np.random.default_rng(seed)
        )

        MaxwellBoltzmannDistribution(cluster, temperature_K=temperature)
        passed_minimum = PassedMinimum()
        mincount = 0
        energies, oldpositions = [], []
        i = 0
        while mincount < mdmin and i < 10000:
            dyn.run(1) #Run MD for 1 step
            energies.append(cluster.get_potential_energy())
            passedmin = passed_minimum(energies)
            if passedmin: #Check if we have passed a minimum
                mincount += 1 #Add this minimum to our mincount
            oldpositions.append(cluster.positions.copy())
            i += 1
        cluster.positions = oldpositions[passedmin[0]]



    def twist(self, cluster):
        #Twist doesnt have a check since it is a rotation and it wouldnt collide with already existing atoms.
        group1, group2, normal = self.split_cluster(cluster)
        choice = np.random.choice([0, 1])
        chosen_group = group1 if choice == 0 else group2

        angle = np.random.uniform(0, 2 * np.pi)
        matrix = rotation_matrix(normal, angle)

        for atom in chosen_group:
            atom.position = np.dot(matrix, atom.position)

        if choice == 0:
            group = group1.extend(chosen_group)
        else:
            group = group2.extend(chosen_group)
        
        return group

    def etching(self, cluster):
        pass


    def split_cluster(self, cluster: Atoms, p1=np.random.rand(3), p2=np.random.rand(3), p3=np.random.rand(3)):
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
        return group1, group2, normal


    def align_cluster(self, cluster: Atoms):
        cl = np.array(cluster.positions)
        center_of_mass = np.mean(cl, axis=0)
        cluster_centered = cl - center_of_mass
        pca = PCA(n_components=3)
        pca.fit(cluster_centered)
        principal_axes = pca.components_
        rotated_cluster = np.dot(cluster_centered, principal_axes.T)
        cluster.positions = rotated_cluster
        return cluster
