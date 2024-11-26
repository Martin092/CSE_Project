import numpy as np
from ase.units import fs
from ase.md.langevin import Langevin
from sklearn.decomposition import PCA  # type: ignore
from reference_code.rotation_matrices import rotation_matrix
from GlobalOptimizer import GlobalOptimizer
from ase import Atoms, Atom
from typing import List, Literal, Tuple
import sys


class Disturber:

    # Class with all the methods to disturb a cluster

    def __init__(self, global_optimizer: GlobalOptimizer) -> None:
        self.global_optimizer = global_optimizer

    def random_step(self, cluster: Atoms, box_size: float) -> None:
        """
        Moves the highest energy atom in a random direction
        :param cluster: the cluster we want to disturb
        :param box_size the size of the container containing the atoms
        :return: result is written directly to cluster, nothing is returned
        """
        energies = cluster.get_potential_energies()  # type: ignore
        index = np.argmax(energies)
        energy_before = cluster.get_potential_energy()  # type: ignore

        # random step from -1 to 1
        step_size = (np.random.rand(3) - 0.5) * 2

        attempts = 0
        while True:
            # every 100 attempts to find a new step, increase the step size by 1.
            # NOTE: probably not the best way to go about the algorithm not finding an appropriate step but works
            attempts += 1
            if attempts % 100 == 0:
                step_size += 1

            step = (np.random.rand(3) - 0.5) * 2 * step_size
            energy_before = self.global_optimizer.clusterList[0].get_potential_energy()

            cluster.positions[index] += step
            cluster.positions = np.clip(
                cluster.positions,
                -self.global_optimizer.boxLength,
                self.global_optimizer.boxLength,
            )

            energy_after = self.global_optimizer.clusterList[0].get_potential_energy()

            # Metropolis criterion gives an acceptance probability based on temperature for each move
            if not self.metropolis_criterion(energy_before, energy_after, 1):
                cluster.positions[index] -= step
                continue
            break

    def metropolis_criterion(
        self, initial_energy: float, new_energy: float, temp: float = 0.8
    ) -> bool:
        """
        Metropolis acceptance criterion for accepting a new move based on temperature
        :param initial_energy: The energy of the cluster before the move
        :param new_energy: The energy of the cluster after the move
        :param temp: temperature at which we want the move to occur
        :return: whether the move is accepted
        """
        if (
            np.isnan(new_energy) or new_energy - initial_energy > 50
        ):  # Energy is way too high, bad move
            return False
        elif new_energy > initial_energy:
            accept_prob = np.exp(-(new_energy - initial_energy) / temp)
            # We accept each move with a probability given by the Metropolis criterion
            return bool(np.random.rand() < accept_prob)
        else:  # We went downhill, cool
            return True

    def check_atom_position(self, cluster: Atoms, atom: Atom) -> bool:
        if np.linalg.norm(atom.position) > self.global_optimizer.boxLength:
            return False
        for other_atom in cluster:
            if (
                np.linalg.norm(atom.position - other_atom.position)
                < 0.5 * self.global_optimizer.covalentRadius
            ):
                return False
        return True

    def check_group_position(
        self, group_static: List[Atom], group_moved: List[Atom]
    ) -> bool:
        for atom in group_moved:
            if np.linalg.norm(atom.position) > self.global_optimizer.boxLength:
                return False
            for other_atom in group_static:
                if (
                    np.linalg.norm(atom.position - other_atom.position)
                    < 0.5 * self.global_optimizer.covalentRadius
                ):
                    return False
        return True

    def angular_movement(self, cluster: Atoms) -> None:
        """
        Perform a rotational movement for the atom with the highest energy.
        :param cluster: The atomic cluster to modify
        """

        energies = cluster.get_potential_energies()  # type: ignore
        index = np.argmax(energies)

        initial_positions = cluster.positions.copy()
        initial_energy = cluster.get_potential_energy()  # type: ignore
        max_attempts = 500
        temperature = 1.0

        cluster.set_center_of_mass([0, 0, 0])  # type: ignore

        for attempt in range(max_attempts):
            vector = np.random.rand(3) - 0.5
            angle = np.random.uniform(0, 2 * np.pi)

            rotation = rotation_matrix(vector, angle)

            rotated_position = np.dot(rotation, cluster.positions[index])
            cluster.positions[index] = rotated_position

            cluster.positions = np.clip(
                cluster.positions,
                -self.global_optimizer.boxLength,
                self.global_optimizer.boxLength,
            )

            new_energy = cluster.get_potential_energy()  # type: ignore

            if self.metropolis_criterion(initial_energy, new_energy, temperature):
                break
            else:
                cluster.positions = initial_positions
        else:
            print("WARNING: Unable to find a valid rotational move.", file=sys.stderr)

    def md(self, cluster: Atoms, temperature: float, number_of_steps: int) -> None:
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

        dyn.run(number_of_steps)  # type: ignore

    def twist(self, cluster: Atoms) -> Atoms:
        # Twist doesn't have a check since it is a rotation, and it wouldn't collide with already existing atoms.
        group1, group2, normal = self.split_cluster(cluster)
        choice = np.random.choice([0, 1])
        chosen_group = group1 if choice == 0 else group2

        angle = np.random.uniform(0, 2 * np.pi)
        matrix = rotation_matrix(normal, angle)

        for atom in chosen_group:
            atom.position = np.dot(matrix, atom.position)

        return cluster

    def etching(self, cluster: Atoms) -> None:
        pass

    def split_cluster(
        self,
        cluster: Atoms,
        p1: np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]] = np.random.rand(3),
        p2: np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]] = np.random.rand(3),
        p3: np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]] = np.random.rand(3),
    ) -> Tuple[
        List[Atom], List[Atom], np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]]
    ]:
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        d = -np.dot(normal, p1)
        group1 = []
        group2 = []
        for atom in cluster:
            val = np.dot(normal, atom.position) + d
            if val > 0:
                group1.append(atom)
            else:
                group2.append(atom)
        return group1, group2, normal

    def align_cluster(self, cluster: Atoms) -> Atoms:
        cl = np.array(cluster.positions)
        center_of_mass = np.mean(cl, axis=0)
        cluster_centered = cl - center_of_mass
        pca = PCA(n_components=3)
        pca.fit(cluster_centered)
        principal_axes = pca.components_
        rotated_cluster = np.dot(cluster_centered, principal_axes.T)
        cluster.positions = rotated_cluster
        return cluster
