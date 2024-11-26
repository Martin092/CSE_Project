"""TODO: Write this."""
from typing import Any
from ase.optimize import BFGS
from ase.calculators.lj import LennardJones
from ase.io import write
from global_optimizer import GlobalOptimizer


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
        super().__init__(  # type: ignore
            num_clusters=num_clusters,
            local_optimizer=local_optimizer,
            atoms=atoms,
            atom_type=atom_type,
            calculator=calculator,
        )
        self.last_energy = self.cluster_list[0].get_potential_energy()

    def iteration(self):  # type: ignore
        for index, clus in enumerate(self.cluster_list):
            self.last_energy = self.cluster_list[index].get_potential_energy()

            energies = self.cluster_list[index].get_potential_energies()
            min_en = min(energies)
            max_energy = max(energies)

            # self.disturber.random_step(cluster)
            if abs(min_en - max_energy) < 1.5:
                self.disturber.random_step(clus)  # type: ignore
            else:
                self.disturber.angular_movement(clus)

            self.optimizers[index].run(fmax=0.2)
            self.history[index].append(cluster)

    def is_converged(self):  # type: ignore
        if self.current_iteration < 2:
            return False

        # TODO this takes in only one cluster into account, use all of them
        current = self.cluster_list[0].get_potential_energy()
        return abs(current - self.last_energy) < 2e-6

    def setup(self):  # type: ignore
        pass


bh = BasinHoppingOptimizer(local_optimizer=BFGS, atoms=13, atom_type="Fe")  # type: ignore
print(bh.box_length)
write("clusters/basin_before.xyz", bh.cluster_list[0])
bh.run(1000)

min_energy = float("inf")
BEST = None
for cluster in bh.history[0]:
    cluster.calc = bh.calculator()
    curr_energy = cluster.get_potential_energy()
    if curr_energy < min_energy:
        min_energy = curr_energy
        BEST = cluster

print(min_energy)

write("clusters/basin_optimized.xyz", BEST)  # type: ignore
