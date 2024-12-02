"""Utility methods module"""

from typing import Any, List, Literal, Tuple
import time
import sys

from mpi4py import MPI
import numpy as np
from sklearn.decomposition import PCA  # type: ignore
from scipy.spatial.distance import pdist  # type: ignore
from ase import Atoms, Atom
from ase.units import fs
from ase.optimize import BFGS
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
            energy_before = cluster.get_potential_energy()  # type: ignore

            cluster.positions[index] += step
            cluster.positions = np.clip(
                cluster.positions,
                -self.global_optimizer.box_length,
                self.global_optimizer.box_length,
            )

            energy_after = cluster.get_potential_energy()  # type: ignore

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
        while True:
            group1, group2, normal = self.split_cluster(cluster)
            choice = np.random.choice([0, 1])
            if choice == 0:
                rotate = group1
                still = group2
            else:
                rotate = group2
                still = group1

            angle = np.random.uniform(0, 2 * np.pi)
            matrix = rotation_matrix(normal, angle)

            positions = []

            for atom in still:
                positions.append(atom.position)

            for atom in rotate:
                positions.append(np.dot(matrix, atom.position))

            if self.configuration_validity(np.array(positions)):
                for atom in rotate:
                    atom.position = np.dot(matrix, atom.position)
                break

        return cluster

    def etching_subtraction(self, cluster: Atoms) -> None:
        """
        Deletes a random atom from the cluster, optimizes the cluster, and
        then adds a new atom to maintain the same number of atoms.
        :param cluster: The atomic cluster to modify
        """
        atom_index = np.random.randint(len(cluster))

        del cluster[atom_index]  # type: ignore
        opt = BFGS(cluster, logfile="log.txt")
        opt.run(fmax=0.02)  # type: ignore

        val = False
        i = 0

        while not val:
            i += 1
            new_atom = Atom(
                self.global_optimizer.atom_type,
                position=np.random.uniform(
                    -self.global_optimizer.box_length,
                    self.global_optimizer.box_length,
                    size=3,
                ),
            )  # type: ignore
            initial_energy = cluster.get_potential_energy()  # type: ignore
            cluster.append(new_atom)  # type: ignore
            new_energy = cluster.get_potential_energy()  # type: ignore

            if self.metropolis_criterion(initial_energy, new_energy):
                val = True
                break
            cluster.pop(-1)  # type: ignore

            if i > 100:
                print("Unable to find a valid etching move.")

    def etching_addition(self, cluster: Atoms) -> None:
        """
        Adds a new atom to the cluster, optimizes the cluster, and then deletes the highest energy atom.
        :param cluster: The atomic cluster to modify
        """
        val = False
        i = 0

        while not val:
            i += 1
            new_atom = Atom(
                self.global_optimizer.atom_type,
                position=np.random.uniform(
                    -self.global_optimizer.box_length,
                    self.global_optimizer.box_length,
                    size=3,
                ),
            )  # type: ignore
            initial_energy = cluster.get_potential_energy()  # type: ignore
            cluster.append(new_atom)  # type: ignore
            new_energy = cluster.get_potential_energy()  # type: ignore

            if self.metropolis_criterion(initial_energy, new_energy):
                val = True
                break

            cluster.pop(-1)  # type: ignore

            if i > 100:
                print("Unable to find a valid etching move.")

        opt = BFGS(cluster, logfile="log.txt")
        opt.run(fmax=0.02)  # type: ignore

        atom_index = np.argmax(cluster.get_potential_energies())  # type: ignore
        del cluster[atom_index]  # type: ignore

    def split_cluster(
        self,
        cluster: Atoms,
        p1: np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]] = np.random.rand(3)
        - 0.5,
        p2: np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]] = np.random.rand(3)
        - 0.5,
        p3: np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]] = np.random.rand(3)
        - 0.5,
    ) -> Tuple[
        List[Atom], List[Atom], np.ndarray[Tuple[Literal[3]], np.dtype[np.float64]]
    ]:
        """
        Takes a geometric plane defined by three points (if points are not specified,
        randomly generated points are taken) and splits the cluster atoms into two groups based on that plane.
        :param cluster: Cluster object to be split
        :param p1: 3D coordinates of the first point
        :param p2: 3D coordinates of the second point
        :param p3: 3D coordinates of the third point
        :return: List of the two atom groups as well as the normal of the geometric plane.
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
        Aligns a cluster such that the center of mass is the origin and
        the highest variance lies in the x-axis, while the smallest one in the z-axis.
        :param cluster: Cluster to be aligned.
        :return: Cluster object with the new atom coordinates.
        """
        center_of_mass = cluster.get_center_of_mass()  # type: ignore
        cluster_centered = cluster.get_positions() - center_of_mass  # type: ignore
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

    def configuration_validity(
        self, positions: np.ndarray[Tuple[Any, Literal[3]], np.dtype[np.float64]]
    ) -> bool:
        """
        TODO: Write this.
        """
        distances = pdist(positions)
        return bool(float(np.min(distances)) >= 0.1)
