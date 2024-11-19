import numpy as np
from ase.units import fs
from ase.md.langevin import Langevin
from sklearn.decomposition import PCA
from reference_code.rotation_matrices import rotation_matrix
from ase import Atoms


class Disturber:

    # Class with all the methods to disturb a cluster

    def __init__(self, local_optimizer):
        self.local_optimizer = local_optimizer

    def random_setup(self, cluster):
        atom = cluster[np.random.randint(0, len(cluster))]
        position = atom.position
        new_position = np.random(-1/2* self.covalentRadius,self.covalentRadius*1/2) + position

        #Check if position is valid: check_position(self, cluster, atom)
        pass

    def check_position(self, cluster, atom):
        if np.linalg.norm(atom.position) > self.boxLength:
            return False
        
        for other_atom in cluster:
            if np.linalg.norm(atom.position - other_atom.position) < 0.5 * self.covalentRadius:
                return False
        
        return True
    
    def check_position(self, group_static, group_moved):
        for atom in group_moved:
            if np.linalg.norm(atom.position) > self.boxLength:
                return False
    
            for other_atom in group_static:
                if np.linalg.norm(atom.position - other_atom.position) < 0.5 * self.covalentRadius:
                    return False
        
        return True
            
        
    

    @staticmethod
    def angular_movement(cluster):
        vector = np.random.rand(3)
        atom = cluster[np.random.randint(0, len(cluster))]
        angle = np.random.uniform(0, 2 * np.pi)
        atom.position = np.dot(rotation_matrix(vector, angle), atom.position)
        #Check if position is valid: check_position(self, cluster, atom)
        return cluster

    @staticmethod
    def md(cluster, temperature, number_of_steps):
        """
        Perform a Molecular Dynamics run using Langevin Dynamics
        :param cluster: Cluster of atoms
        :param temperature: Temperature in Kelvin
        :param number_of_steps: Number of steps to use in Molecular Dynamics
        """
        dyn = Langevin(
            cluster,
            timestep=5.0 * fs,  # Feel free to mess with this parameter
            temperature_K=temperature,
            friction=0.5 / fs,  # Feel free to mess with this parameter
        )

        dyn.run(number_of_steps)

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

    @staticmethod
    def split_cluster(cluster: Atoms, p1=np.random.rand(3), p2=np.random.rand(3), p3=np.random.rand(3)):
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

    @staticmethod
    def align_cluster(cluster: Atoms):
        cl = np.array(cluster.positions)
        center_of_mass = np.mean(cl, axis=0)
        cluster_centered = cl - center_of_mass
        pca = PCA(n_components=3)
        pca.fit(cluster_centered)
        principal_axes = pca.components_
        rotated_cluster = np.dot(cluster_centered, principal_axes.T)
        cluster.positions = rotated_cluster
        return cluster
