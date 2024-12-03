"""TODO: Write this."""

import sys
import time
from typing import Any

import matplotlib.pyplot as plt
from ase import Atoms
from ase.optimize import BFGS
from ase.calculators.lj import LennardJones
from ase.io import write
import numpy as np
from mpi4py import MPI

from src.global_optimizer import GlobalOptimizer
from src.oxford_database import get_cluster_energy


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
        local_optimizer: Any,
        atoms: int,
        atom_type: str,
        calculator: Any = LennardJones,
        num_clusters: int = 1,
        alpha: float = 2,
        sensitivity: float = 0.3,
        comm: MPI.Intracomm | None = None,
    ) -> None:
        super().__init__(
            num_clusters=num_clusters,
            local_optimizer=local_optimizer,
            atoms=atoms,
            atom_type=atom_type,
            calculator=calculator,
            comm=comm,
        )
        self.last_energy = float("inf")
        self.alpha = alpha
        self.angular_moves = 0
        self.alphas = np.array([self.alpha])
        self.sensitivity = sensitivity

    def iteration(self) -> None:
        # TODO: make sure the algorithm doesnt stay for too long on the same energy levels
        """
        TODO: Write this.
        :return:
        """
        if self.comm:
            print(f"Iteration {self.current_iteration} in {self.comm.Get_rank()}")
        else:
            print(f"Iteration {self.current_iteration}")
        if self.current_iteration == 0:
            self.last_energy = self.cluster_list[0].get_potential_energy()

        for index, clus in enumerate(self.cluster_list):
            self.last_energy = self.cluster_list[index].get_potential_energy()

            energies = self.cluster_list[index].get_potential_energies()
            min_en = min(energies)
            max_energy = max(energies)

            # self.disturber.random_step(cluster)
            if max_energy - min_en < self.alpha:
                self.utility.random_step(clus)
            else:
                self.angular_moves += 1
                self.utility.angular_movement(clus)

            if self.current_iteration != 0:
                fraction = self.angular_moves / self.current_iteration
                self.alpha = self.alpha * (1 - self.sensitivity * (0.5 - fraction))

            self.alphas = np.append(self.alphas, self.alpha)
            self.optimizers[index].run(fmax=0.2)
            self.history[index].append(clus.copy())

    def is_converged(self) -> bool:
        """
        TODO: Write this.
        :return:
        """
        if self.current_iteration < 10:
            return False

        ret = True
        cur = self.cluster_list[0].get_potential_energy()
        for i in range(self.current_iteration - 8, self.current_iteration - 1):
            self.history[0][i].calc = self.calculator()
            energy_hist = self.history[0][i].get_potential_energy()
            ret &= bool(abs(cur - energy_hist) <= 1e-14)
        # return ret
        return False

    def seed(self, starting_from: int) -> Atoms:
        """
        Finds the best cluster by starting from the given number atoms,
        globally optimizing and then either adding or removing atoms
        until you get to the desired number defined in self.atoms
        :param starting_from: The number of atoms you want to start from
        :return: A cluster of size self.atoms that is hopefully closer to the global minima
        """
        assert (
            starting_from != self.atoms
        ), "You cant seed from the same cluster size as the one you are optimizing for"
        assert self.atoms in {
            starting_from + 1,
            starting_from - 1,
        }, "Seeding only works with one more or one less atoms"

        min_energy = float("inf")
        best_cluster: Atoms
        for i in range(10):
            alg = BasinHoppingOptimizer(
                local_optimizer=self.local_optimizer,
                atoms=starting_from,
                atom_type=self.atom_type,
            )
            alg.run(1000)

            energy_curr = alg.cluster_list[0].get_potential_energy()
            if energy_curr < min_energy:
                min_energy = energy_curr
                best_cluster = alg.cluster_list[0]

        write("clusters/seeded_LJ_before.xyz", best_cluster)
        print(f"seeding before {best_cluster.get_potential_energy()}")  # type: ignore

        # Add or remove atom from the found cluster
        positions = None
        if starting_from > self.atoms:
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

        new_cluster = Atoms(self.atom_type + str(self.atoms), positions=positions)  # type: ignore
        new_cluster.calc = self.calculator()

        write("clusters/seeded_LJ_finished.xyz", new_cluster)
        print(f"seeded finished {new_cluster.get_potential_energy()}")  # type: ignore

        return new_cluster

    def plot_energies(self) -> None:
        """
        Plots the energy values over the course of the entire run
        """
        energies = np.array([])
        for clus in self.history[0]:
            clus.calc = self.calculator()
            energies = np.append(energies, clus.get_potential_energy())

        plt.plot(energies)
        plt.title(f"Energy levels discovered for LJ{self.atoms}")
        plt.xlabel("Iteration")
        plt.ylabel("Energy")
        plt.show()


if __name__ == "__main__":
    bh = BasinHoppingOptimizer(local_optimizer=BFGS, atoms=13, atom_type="Fe")
    print(bh.box_length)

    start = time.time()
    bh.run(100)
    print(f"Algorithm finished for {time.time() - start}")

    energy, cluster = bh.best_energy(0)
    print(f"Result: {energy}")
    print(f"Actual: {get_cluster_energy(bh.atoms, bh.atom_type)}")

    print(bh.current_iteration)
    print(bh.angular_moves)

    print(f"alpha: {bh.alpha}")
    print(f"step: {bh.utility.step}")

    plt.plot(bh.alphas)
    plt.title("Alpha values per iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Alpha value")
    plt.show()

    bh.plot_energies()

    write("clusters/LJmin.xyz", cluster)
