"""TODO: Write this."""

from typing import Any, List, Literal, Tuple
import time
import sys

from mpi4py import MPI
import numpy as np
from sklearn.decomposition import PCA  # type: ignore
from ase import Atoms, Atom
from ase.units import fs
from ase.md.langevin import Langevin
from ase.optimize.minimahopping import PassedMinimum
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from reference_code.rotation_matrices import rotation_matrix


class Utility:
    """
    Class with all the methods to disturb a cluster
    """

    def __init__(
        self, global_optimizer: Any, temp: float = 0.8, step: float = 1
    ) -> None:
        self.global_optimizer = global_optimizer
        self.temp = temp
        self.step = step

    def random_step(self, cluster: Atoms) -> None:
        """
        Moves the highest energy atom in a random direction
        :param cluster: the cluster we want to disturb
        :return: result is written directly to cluster, nothing is returned
        """
        energies = cluster.get_potential_energies()  # type: ignore
        index = np.argmax(energies)
        energy_before = cluster.get_potential_energy()  # type: ignore

        while True:
            step = (np.random.rand(3) - 0.5) * 2 * self.step
            energy_before = self.global_optimizer.cluster_list[0].get_potential_energy()

            cluster.positions[index] += step
            cluster.positions = np.clip(
                cluster.positions,
                -self.global_optimizer.box_length,
                self.global_optimizer.box_length,
            )

            energy_after = self.global_optimizer.cluster_list[0].get_potential_energy()

            # Metropolis criterion gives an acceptance probability based on temperature for each move
            accept = self.metropolis_criterion(energy_before, energy_after)
            self.step = self.step * (1 - 0.01 * (0.5 - accept))

            if np.random.rand() > accept:
                cluster.positions[index] -= step
                continue
            break

    def metropolis_criterion(self, initial_energy: float, new_energy: float) -> float:
        """
        Metropolis acceptance criterion for accepting a new move based on temperature
        :param initial_energy: The energy of the cluster before the move
        :param new_energy: The energy of the cluster after the move
        :param temp: temperature at which we want the move to occur
        :return: probability of accepting the move
        """
        if new_energy - initial_energy > 50:  # Energy is way too high, bad move
            return float(0)
        if np.isnan(new_energy):
            if self.global_optimizer.comm:
                self.global_optimizer.comm.Send(
                    [np.array([]), MPI.DOUBLE], dest=0, tag=1
                )
            sys.exit("NaN encountered, exiting")
        if new_energy > initial_energy:
            return float(min(1, np.exp(-(new_energy - initial_energy) / self.temp)))

        return float(1)

    def check_atom_position(self, cluster: Atoms, atom: Atom) -> bool:
        """
        TODO: Write this.
        :param cluster:
        :param atom:
        :return:
        """
        if np.linalg.norm(atom.position) > self.global_optimizer.box_length:
            return False
        for other_atom in cluster:
            if (
                np.linalg.norm(atom.position - other_atom.position)
                < 0.5 * self.global_optimizer.covalent_radius
            ):
                return False
        return True

    def check_group_position(
        self, group_static: List[Atom], group_moved: List[Atom]
    ) -> bool:
        """
        TODO: Write this.
        """
        for atom in group_moved:
            if np.linalg.norm(atom.position) > self.global_optimizer.box_length:
                return False
            for other_atom in group_static:
                if (
                    np.linalg.norm(atom.position - other_atom.position)
                    < 0.5 * self.global_optimizer.covalent_radius
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

        cluster.set_center_of_mass([0, 0, 0])  # type: ignore

        for _ in range(max_attempts):
            vector = np.random.rand(3) - 0.5
            angle = np.random.uniform(0, 2 * np.pi)

            rotation = rotation_matrix(vector, angle)

            rotated_position = np.dot(rotation, cluster.positions[index])
            cluster.positions[index] = rotated_position

            new_energy = cluster.get_potential_energy()  # type: ignore

            accept = self.metropolis_criterion(initial_energy, new_energy)
            if np.random.rand() < accept:
                break
            cluster.positions = initial_positions
        else:
            print("WARNING: Unable to find a valid rotational move.", file=sys.stderr)

    def md(
        self,
        cluster: Atoms,
        temperature: float,
        mdmin: int,
        seed: int = int(time.time()),
    ) -> None:
        """
        Perform a Molecular Dynamics run using Langevin Dynamics
        :param cluster: Cluster of atoms
        :param temperature: Temperature in Kelvin
        :param mdmin: Number of minima to be found before MD run halts.
        Alternatively it will halt once we reach 10000 iterations
        :param seed: seed for random generation, can be used for testing
        """
        dyn = Langevin(
            cluster,
            timestep=5.0 * fs,  # Feel free to mess with this parameter
            temperature_K=temperature,
            friction=0.5 / fs,  # Feel free to mess with this parameter
            rng=np.random.default_rng(seed),
        )

        MaxwellBoltzmannDistribution(cluster, temperature_K=temperature)
        passed_minimum = PassedMinimum()  # type: ignore
        mincount = 0
        energies, oldpositions = [], []
        i = 0
        while mincount < mdmin and i < 10000:
            dyn.run(1)  # type: ignore # Run MD for 1 step
            energies.append(cluster.get_potential_energy())  # type: ignore
            passedmin = passed_minimum(energies)
            if passedmin:  # Check if we have passed a minimum
                mincount += 1  # Add this minimum to our mincount
            oldpositions.append(cluster.positions.copy())
            i += 1
        print("Number of MD steps: " + str(i))
        cluster.positions = oldpositions[passedmin[0]]
        cluster.positions = np.clip(
            cluster.positions,
            -self.global_optimizer.box_length,
            self.global_optimizer.box_length,
        )

    def twist(self, cluster: Atoms) -> Atoms:
        """
        TODO: Write this.
        :param cluster:
        :return:
        """
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
        """
        TODO: Write this.
        :param cluster:
        :return:
        """

    def split_cluster(
        self,
        cluster: Atoms,
        p1: np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]] = np.random.rand(3),
        p2: np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]] = np.random.rand(3),
        p3: np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]] = np.random.rand(3),
    ) -> Tuple[
        List[Atom], List[Atom], np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]]
    ]:
        """
        TODO: Write this.
        :param cluster:
        :param p1:
        :param p2:
        :param p3:
        :return:
        """
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
        """
        TODO: Write this.
        :param cluster:
        :return:
        """
        cl = np.array(cluster.positions)
        center_of_mass = np.mean(cl, axis=0)
        cluster_centered = cl - center_of_mass
        pca = PCA(n_components=3)
        pca.fit(cluster_centered)
        principal_axes = pca.components_
        rotated_cluster = np.dot(cluster_centered, principal_axes.T)
        cluster.positions = rotated_cluster
        return cluster

    def compare_clusters(self, cluster1: Atoms, cluster2: Atoms) -> np.bool:
        """
        Checks whether two clusters are equal based on their potential energy.
        This method may be changed in the future to use more sophisticated methods,
        such as overlap matrix fingerprint thresholding.
        :param cluster1: First cluster
        :param cluster2: Second cluster
        :return: boolean
        """
        return np.isclose(cluster1.get_potential_energy(), cluster2.get_potential_energy())  # type: ignore
