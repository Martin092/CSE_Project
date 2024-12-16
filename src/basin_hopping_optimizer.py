"""Basin Hopping Optimizer module"""

import sys
from typing import Any

from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.io import write
from ase.optimize import FIRE
import numpy as np
from mpi4py import MPI  # pylint: disable=E0611
from src.global_optimizer import GlobalOptimizer


class BasinHoppingOptimizer(GlobalOptimizer):
    """
    A class for optimizing atomic clusters using the basin hopping algorithm

    Attributes:
        local_optimizer     the optimizer used for local optimization of the clusters
        atoms               number of atoms that we want to optimize
        atom_type           type of the atoms
        calculator          the calculator used for calculating the intermolecular potentials
        num_clusters        the number of clusters we want to work on (sequentially)
        alpha               the minimum energy difference between the lowest and highest
        energy pair for angular movements to take place. Keep in mind that this parameter
        changes overtime and will oscillate around a certain value. Optimally it should
        start around that value.
        sensitivity         how quickly will alpha change. Larger values result in bigger oscillations
    """

    def __init__(
        self,
        local_optimizer: Any = FIRE,
        calculator: Any = LennardJones,
        alpha: float = 2,
        sensitivity: float = 0.3,
        comm: MPI.Intracomm | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(
            local_optimizer=local_optimizer,
            calculator=calculator,
            comm=comm,
            debug=debug,
        )
        self.last_energy = float("inf")
        self.alpha = alpha
        self.angular_moves = 0
        self.alphas = np.array([self.alpha])
        self.sensitivity = sensitivity

    def iteration(self) -> None:
        """
        Performs single iteration of the Basin Hopping Optimizer.
        :return: None.
        """
        if self.comm and self.debug:
            print(
                f"Iteration {self.current_iteration} in {self.comm.Get_rank()}"
            )  # pragma: no cover
        elif self.debug:
            print(f"Iteration {self.current_iteration}")

        if self.current_iteration == 0:
            self.last_energy = self.current_cluster.get_potential_energy()  # type: ignore

        self.last_energy = self.current_cluster.get_potential_energy()  # type: ignore

        energies = self.current_cluster.get_potential_energies()  # type: ignore
        min_en = min(energies)
        max_energy = max(energies)

        if max_energy - min_en < self.alpha:
            self.utility.random_step(self.current_cluster)
        else:
            self.angular_moves += 1
            self.utility.angular_movement(self.current_cluster)

            if self.current_iteration != 0:
                fraction = self.angular_moves / self.current_iteration
                self.alpha = self.alpha * (1 - self.sensitivity * (0.5 - fraction))

        self.alphas = np.append(self.alphas, self.alpha)
        opt = self.local_optimizer(self.current_cluster, logfile=self.logfile)
        opt.run()
        self.configs.append(self.current_cluster.copy())  # type: ignore

        curr_energy = self.current_cluster.get_potential_energy()  # type: ignore
        self.potentials.append(curr_energy)
        if curr_energy < self.best_potential:
            self.best_config = self.current_cluster.copy()  # type: ignore
            self.best_potential = curr_energy

    def is_converged(self) -> bool:
        """
        Checks if convergence criteria is satisfied.
        :return: True if convergence criteria is met, otherwise False.
        """
        if self.current_iteration < self.conv_iterations:
            return False

        decreased = False
        biggest = self.current_cluster.get_potential_energy()
        for i in reversed(range(self.current_iteration - self.conv_iterations, self.current_iteration - 1)):
            self.configs[i].calc = self.calculator()
            energy = self.configs[i].get_potential_energy()

            decreased |= energy > biggest

        # if not decreased:
        #     print(f"Converged at {self.current_iteration}")

        return not decreased

    def seed(self, starting_from: int) -> Atoms:  # pragma: no cover
        """
        Finds the best cluster by starting from the given number atoms,
        globally optimizing and then either adding or removing atoms
        until you get to the desired number defined in self.num_atoms
        :param starting_from: The number of atoms you want to start from
        :return: A cluster of size self.num_atoms that is hopefully closer to the global minima
        """
        assert (
            starting_from != self.utility.num_atoms
        ), "You cant seed from the same cluster size as the one you are optimizing for"
        assert self.utility.num_atoms in {
            starting_from + 1,
            starting_from - 1,
        }, "Seeding only works with one more or one less atoms"

        min_energy = float("inf")
        best_cluster: Atoms
        for i in range(10):
            alg = BasinHoppingOptimizer(
                local_optimizer=self.local_optimizer, calculator=self.calculator
            )
            alg.run(
                num_atoms=self.utility.num_atoms,
                atom_type=self.utility.atom_type,
                max_iterations=300,
            )

            energy_curr = alg.current_cluster.get_potential_energy()  # type: ignore
            if energy_curr < min_energy:
                min_energy = energy_curr
                best_cluster = alg.current_cluster

        write("clusters/seeded_LJ_before.xyz", best_cluster)
        print(f"seeding before {best_cluster.get_potential_energy()}")  # type: ignore

        # Add or remove atom from the found cluster
        positions = None
        if starting_from > self.utility.num_atoms:
            # if we started with more atoms just remove the highest energy one
            energies = best_cluster.get_potential_energies()  # type: ignore
            index = np.argmax(energies)
            positions = np.delete(best_cluster.positions, index, axis=0)
        else:
            # if we started with fewer atoms add at the distance
            # furthest from the center of mass plus some random number between 0 and 1
            best_cluster.set_center_of_mass(0)  # type: ignore
            distances = np.zeros(starting_from)
            for i in range(starting_from):
                distances[i] = np.linalg.norm(best_cluster.positions[i])

            max_dist = np.max(distances)

            new_pos = max_dist + np.random.rand(1, 3)
            positions = best_cluster.positions.copy()
            if positions is None:
                sys.exit("Something went wrong")
            positions = np.vstack((positions, new_pos))

        new_cluster = Atoms(self.utility.atom_type + str(self.utility.num_atoms), positions=positions)  # type: ignore
        new_cluster.calc = self.calculator()

        write("clusters/seeded_LJ_finished.xyz", new_cluster)
        print(f"seeded finished {new_cluster.get_potential_energy()}")  # type: ignore

        return new_cluster
