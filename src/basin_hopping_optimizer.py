"""TODO: Write this."""

from typing import Any
from ase import Atoms
from ase.optimize import BFGS
from ase.calculators.lj import LennardJones
from ase.io import write
import numpy as np
from src.global_optimizer import GlobalOptimizer


class BasinHoppingOptimizer(GlobalOptimizer):
    """
    TODO: Write this.
    """

    def __init__(
        self,
        local_optimizer: Any,
        atoms: int,
        atom_type: str,
        calculator: Any = LennardJones,
        num_clusters: int = 1,
    ) -> None:
        super().__init__(
            num_clusters=num_clusters,
            local_optimizer=local_optimizer,
            atoms=atoms,
            atom_type=atom_type,
            calculator=calculator,
        )
        self.last_energy = float("inf")

    def iteration(self) -> None:
        """
        TODO: Write this.
        :return:
        """
        if self.current_iteration == 0:
            self.last_energy = self.cluster_list[0].get_potential_energy()

        for index, clus in enumerate(self.cluster_list):
            self.last_energy = self.cluster_list[index].get_potential_energy()

            energies = self.cluster_list[index].get_potential_energies()
            min_en = min(energies)
            max_energy = max(energies)

            # self.disturber.random_step(cluster)
            if abs(min_en - max_energy) < 1.5:
                self.disturber.random_step(clus)
            else:
                self.disturber.angular_movement(clus)

            self.optimizers[index].run(fmax=0.2)
            self.history[index].append(clus)

    def is_converged(self) -> bool:
        """
        TODO: Write this.
        :return:
        """
        if self.current_iteration < 2:
            return False

        # TODO this takes in only one cluster into account, use all of them
        current = self.cluster_list[0].get_potential_energy()
        return bool(abs(current - self.last_energy) < 2e-6)

    # def setup(self) -> None:
    #     """
    #     TODO: Write this.
    #     :return:
    #     """

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
        for i in range(5):
            alg = BasinHoppingOptimizer(
                local_optimizer=self.local_optimizer,
                atoms=starting_from,
                atom_type=self.atom_type,
            )
            alg.run(1000)

            energy = alg.cluster_list[0].get_potential_energy()
            if energy < min_energy:
                min_energy = energy
                best_cluster = alg.cluster_list[0]

        write("../clusters/seeded_LJ_before.xyz", best_cluster)
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
                exit("Something went wrong")
            positions = np.vstack((positions, new_pos))

        new_cluster = Atoms(self.atom_type + str(self.atoms), positions=positions) # type: ignore
        new_cluster.calc = self.calculator()

        write("../clusters/seeded_LJ_finished.xyz", new_cluster)
        print(f"seeded finished {new_cluster.get_potential_energy()}") # type: ignore

        return new_cluster


bh = BasinHoppingOptimizer(local_optimizer=BFGS, atoms=14, atom_type="Fe")
print(bh.box_length)

bh.run(1000, seed=bh.seed(bh.atoms - 1))

min_energy2 = float("inf")
BEST_CLUSTER2: Atoms
for cluster in bh.history[0]:
    cluster.calc = bh.calculator()
    curr_energy = cluster.get_potential_energy()
    if curr_energy < min_energy2:
        min_energy2 = curr_energy
        BEST_CLUSTER2 = cluster

print(min_energy2)

write("../clusters/LJ_min.xyz", BEST_CLUSTER2)
