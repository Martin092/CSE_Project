"""Utility methods module"""

from typing import Any, List, Literal, Tuple, Counter
import time
import sys

import numpy as np
from sklearn.decomposition import PCA  # type: ignore
from scipy.spatial.distance import pdist  # type: ignore
from ase import Atoms, Atom
from ase.io import Trajectory
from ase.units import fs
from ase.md.langevin import Langevin
from ase.optimize.minimahopping import PassedMinimum
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


class Utility:
    """
    Class with all the methods to disturb a cluster
    """

    def __init__(
        self,
        global_optimizer: Any,
        atoms: str,
        temp: float = 0.8,
        step: float = 1,
    ) -> None:
        """
        Utility Class Constructor
        :param global_optimizer: Global optimizer object.
        :param atoms: Number of atoms and atomic type in cluster.
        :param temp: Temperature.
        :param step: Step length.
        """
        self.global_optimizer = global_optimizer
        self.atoms = atoms
        self.temp = temp
        self.step = step
        self.big_jumps: list[int] = []
        self.covalent_radius: float = 1.0
        self.num_atoms = len(Atoms(atoms))  # type: ignore
        self.box_length: float = (
            2
            * self.covalent_radius
            * (0.5 + ((3.0 * self.num_atoms) / (4 * np.pi * np.sqrt(2))) ** (1 / 3))
        )
        self.atom_types: Counter[str, int] = Counter(self.generate_cluster().get_chemical_symbols())  # type: ignore

    def generate_cluster(
        self,
        positions: (
            np.ndarray[Tuple[Any, Literal[3]], np.dtype[np.float64]] | None
        ) = None,
        seed: int | None = None,
    ) -> Atoms:
        """
        Generate Atoms object with given or randomly generated coordinates.
        :param positions: Atomic coordinates, if None or Default, randomly generated.
        :param seed: seed for the random number generator
        :return: Atoms object with cell and calculator specified.
        """
        rng: np.random.Generator
        if seed:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        if positions is None:
            positions = (rng.random((self.num_atoms, 3)) - 0.5) * self.box_length * 1.5
            while not self.configuration_validity(positions):
                positions = (
                    (rng.random((self.num_atoms, 3)) - 0.5) * self.box_length * 1.5
                )
        clus = Atoms(
            self.atoms,
            positions=positions,
            cell=np.array(
                [
                    [self.box_length, 0, 0],
                    [0, self.box_length, 0],
                    [0, 0, self.box_length],
                ]
            ),
            calculator=self.global_optimizer.calculator(),
        )  # type: ignore
        return clus

    def random_step(
        self,
        cluster: Atoms,
        max_rejects: int = 5,
        sensitivity: float = 0.01,
        max_local_steps: int = 0,
    ) -> None:
        """
        Moves the highest energy atom in a random direction.
        :param cluster: The cluster to perform the random step for.
        :param max_rejects: Maximum number of steps that can be rejected before a move at temperature infinity is made
        :param sensitivity: how quickly does the step size in order to keep the metropolis at 0.5
        :param max_local_steps: Maximum number of steps for the local optimizer.
        :return: Result is written directly to cluster, nothing is returned.
        """
        energies = cluster.get_potential_energies()  # type: ignore
        index = np.argmax(energies)
        energy_before = cluster.get_potential_energy()  # type: ignore

        rejected = 0
        while True:
            step = (np.random.rand(3) - 0.5) * 2 * self.step

            cluster.positions[index] += step

            opt = self.global_optimizer.local_optimizer(
                cluster, logfile=self.global_optimizer.logfile
            )
            if max_local_steps == 0:
                opt.run()
            else:
                opt.run(steps=max_local_steps)
            energy_after = cluster.get_potential_energy()  # type: ignore

            accept: float
            if rejected > max_rejects:
                self.big_jumps.append(self.global_optimizer.current_iteration)
                break

            # Metropolis criterion gives an acceptance probability based on temperature for each move
            accept = self.metropolis_criterion(energy_before, energy_after)
            self.step = self.step * (1 - sensitivity * (0.5 - accept))

            if np.random.uniform() > accept:
                rejected += 1
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

            rotation = self.rotation_matrix(vector, angle)

            rotated_position = np.dot(rotation, cluster.positions[index])
            cluster.positions[index] = rotated_position

            new_energy = cluster.get_potential_energy()  # type: ignore

            accept = self.metropolis_criterion(initial_energy, new_energy)
            if np.random.uniform() < accept:
                break
            cluster.positions = initial_positions
        else:  # pragma: no cover
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

    def twist(self, cluster: Atoms) -> Atoms:
        """
        Performs twist mutation by splitting the cluster in two by generating a random geometric plane,
        and rotates one of the two parts around the normal of the generated geometric plane.
        :param cluster: Cluster to be twist mutated.
        :return: Cluster configuration after twist mutation.
        """
        group1, group2, normal = self.split_cluster(cluster)
        choice = np.random.choice([0, 1])
        if choice == 0:
            rotate = group1
            still = group2
        else:
            rotate = group2
            still = group1

        angle = np.random.uniform(0, 2 * np.pi)
        matrix = self.rotation_matrix(normal, angle)

        positions = []

        for atom in still:
            positions.append(atom.position)

        for atom in rotate:
            positions.append(np.dot(matrix, atom.position))

        if self.configuration_validity(np.array(positions)):
            for atom in rotate:
                atom.position = np.dot(matrix, atom.position)

        return cluster

    def etching_subtraction(self, cluster: Atoms, max_local_steps: int = 20000) -> None:
        """
        Deletes the highest energy atom from the cluster, optimizes the cluster, and
        then adds a new atom to maintain the same number of atoms.
        :param max_local_steps: Maximum number of steps for the local optimizer.
        :param cluster: The atomic cluster to modify
        """
        atom_index = np.argmax(cluster.get_potential_energies())  # type: ignore
        atomic_type = cluster[atom_index].symbol
        del cluster[atom_index]  # type: ignore

        opt = self.global_optimizer.local_optimizer(
            cluster, logfile=self.global_optimizer.logfile
        )
        opt.run(steps=max_local_steps)

        self.append_atom(cluster, atomic_type)

    def etching_addition(self, cluster: Atoms, max_local_steps: int = 20000) -> None:
        """
        Adds a new atom to the cluster, optimizes the cluster, and then deletes the highest energy atom.
        :param max_local_steps: Maximum number of steps for the local optimizer.
        :param cluster: The atomic cluster to modify
        """
        atomic_type = np.random.choice(list(self.atom_types.keys()))
        self.append_atom(cluster, atomic_type)

        opt = self.global_optimizer.local_optimizer(
            cluster, logfile=self.global_optimizer.logfile
        )
        opt.run(steps=max_local_steps)

        energies = cluster.get_potential_energies()  # type: ignore
        index = -1
        energy = -float("inf")
        for idx, eng in enumerate(energies):
            if eng < energy and cluster[idx].get_chemical_symbol() == atomic_type:
                energy = eng
                index = idx
        del cluster[index]  # type: ignore

    def append_atom(self, cluster: Atoms, atomic_type: str) -> None:
        """
        Appends an atom at a random position in the cluster.
        :param cluster: Cluster to which an atom to be appended.
        :param atomic_type: Atomic type of appended atom.
        :return: None, since cluster object is dynamically updated.
        """
        position = (np.random.rand(3) - 0.5) * self.box_length * 1.5
        while not self.configuration_validity(
            np.append(cluster.positions, [position], axis=0)
        ):
            position = (np.random.rand(3) - 0.5) * self.box_length * 1.5
        new_atom = Atom(
            atomic_type,
            position=position,
        )  # type: ignore
        cluster.append(new_atom)  # type: ignore

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
        cluster.set_center_of_mass((0, 0, 0))  # type: ignore
        pca = PCA(n_components=3)
        pca.fit(cluster.positions)
        principal_axes = pca.components_
        cluster.positions = np.dot(cluster.positions, principal_axes.T)
        return cluster

    def vector(self, length: float) -> np.ndarray:
        """
        Generates a 3D vector in random direction, given its length.
        :param length: Length of the vector.
        :return: 3D vector in random direction, of specified length.
        """
        azimuthal_angle = np.random.uniform(0, 2 * np.pi)
        polar_angle = np.random.uniform(0, np.pi)

        x = length * np.sin(polar_angle) * np.cos(azimuthal_angle)
        y = length * np.sin(polar_angle) * np.sin(azimuthal_angle)
        z = length * np.cos(polar_angle)

        return np.array([x, y, z])

    def random_displacement(
        self, cluster: Atoms, prob: float, length: float = 0.1
    ) -> None:
        """
        Performs random displacement of specified length on atoms in cluster by given probability.
        :param cluster: Cluster object to be mutated by random displacement.
        :param prob: Probability of random displacement per atom.
        :param length: Length of the displacement.
        :return: None, since cluster object is dynamically updated.
        """
        for i in range(self.num_atoms):
            if np.random.rand() < prob:
                if self.global_optimizer.debug:
                    print("Random displacement", flush=True)
                positions = cluster.positions.copy()
                vector = self.vector(length)
                positions[i] += vector
                while not self.configuration_validity(positions):
                    positions = cluster.positions.copy()
                    vector = self.vector(length)
                    positions[i] += vector
                cluster[i].position += vector

    def compare_clusters(
        self, cluster1: Atoms, cluster2: Atoms, atol: float
    ) -> np.bool:
        """
        Checks whether two clusters are equal based on their potential energy.
        This method may be changed in the future to use more sophisticated methods,
        such as overlap matrix fingerprint thresholding.
        :param cluster1: First cluster
        :param cluster2: Second cluster
        :param atol: Tolerance in potential energy difference.
        :return: boolean
        """
        return np.isclose(
            cluster1.get_potential_energy(),  # type: ignore
            cluster2.get_potential_energy(),  # type: ignore
            atol=atol,
            rtol=0,
        )

    def configuration_validity(
        self, positions: np.ndarray[Tuple[Any, Literal[3]], np.dtype[np.float64]]
    ) -> bool:
        """
        Checks if a potential configuration doesn't invalidate the physical laws.
        :param positions: Numpy array of the potential atomic configuration.
        :return: Boolean indicating stability of configuration.
        """
        if positions.shape[0] == 0:
            return True
        distances = pdist(positions)
        return bool(float(np.min(distances)) >= 0.15)

    def rotation_matrix(
        self, normal: np.ndarray[Any, np.dtype[np.float64]], angle: float
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """
        Rotation matrix around a normal with an angle.
        :param normal: Normal of the rotation direction vector.
        :param angle: Angle of rotation.
        :return: Rotation matrix.
        """
        normal = normal / np.linalg.norm(normal)
        x, y, z = normal
        cos = np.cos(angle)
        sin = np.sin(angle)
        one_minus_cos = 1 - cos

        return np.array(
            [
                [
                    cos + x * x * one_minus_cos,
                    x * y * one_minus_cos - z * sin,
                    x * z * one_minus_cos + y * sin,
                ],
                [
                    y * x * one_minus_cos + z * sin,
                    cos + y * y * one_minus_cos,
                    y * z * one_minus_cos - x * sin,
                ],
                [
                    z * x * one_minus_cos - y * sin,
                    z * y * one_minus_cos + x * sin,
                    cos + z * z * one_minus_cos,
                ],
            ]
        )

    def write_trajectory(self, filename: str) -> None:  # pragma: no cover
        """
        Writes all clusters in the history to a trajectory file
        :param filename: Path of the trajectory file, with .traj extension
        :return: None, writes to file
        """
        with Trajectory(filename, mode="w") as traj:  # type: ignore
            for cluster in self.global_optimizer.configs:
                cluster.center()
                traj.write(cluster)  # pylint: disable=E1101
