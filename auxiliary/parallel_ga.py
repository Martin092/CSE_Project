"""GA parallel execution module"""

import sys
import os

from ase.io import write
from mpi4py import MPI  # pylint: disable=E0611
import numpy as np
from matplotlib import pyplot as plt

sys.path.append("./")

from src.genetic_algorithm import GeneticAlgorithm  # pylint: disable=C0413
from auxiliary.cambridge_database import get_cluster_energy  # pylint: disable=C0413


def parallel_ga(
    ga: GeneticAlgorithm,
    num_atoms: int,
    atom_type: str,
    max_iterations: int,
    conv_iterations: int = 0,
    seed: int | None = None,
) -> None:
    """
    Execute GA in parallel using mpiexec.
    :param ga: Genetic Algorithm instance
    :param num_atoms: Number of atoms in cluster to optimize for.
    :param atom_type: Atomic type of cluster.
    :param max_iterations: Number of maximum iterations to perform.
    :param conv_iterations: Number of iterations to be considered in the convergence criteria.
    :param seed: Seeding for reproducibility.
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    ga.comm = comm
    ga.setup(num_atoms, atom_type)

    if rank == 0:
        ga.run(num_atoms, atom_type, max_iterations, conv_iterations, seed)

        for i in range(1, comm.Get_size()):
            comm.Send(np.zeros((num_atoms, 3), dtype=float), tag=0, dest=i)

        comm.free()

        if not os.path.exists("./data/optimizer"):
            os.mkdir("./data")
            os.mkdir("./data/optimizer")

        best_cluster = ga.best_config
        best_cluster.center()  # type: ignore
        write(f"./data/optimizer/LJ{num_atoms}.xyz", best_cluster)

        best = get_cluster_energy(num_atoms, "./")

        ga.utility.write_trajectory(f"./data/optimizer/LJ{num_atoms}.traj")

        plt.plot(ga.potentials)
        plt.title(f"Execution on LJ{num_atoms}")
        plt.xlabel("Iteration")
        plt.ylabel("Potential Energy")
        plt.savefig(f"./data/optimizer/LJ{num_atoms}.png")
        plt.close()

        print(
            f"LJ {num_atoms}: {ga.current_iteration} iterations for "
            f"{int(np.floor_divide(ga.execution_time, 60))} min {int(ga.execution_time) % 60} sec"
        )

        if abs(ga.best_potential - best) < 0.0001:
            print("Found global minimum from database.")
        elif ga.best_potential < best:
            print(
                f"Found new global minimum. Found {ga.best_potential}, but database minimum is {best}."
            )
        else:
            print(
                f"Didn't find global minimum. Found {ga.best_potential}, but global minimum is {best}."
            )

        MPI.Finalize()

    else:
        while True:
            if ga.debug:
                print(f"Rank {ga.comm.Get_rank()}", flush=True)
            pos = np.empty((num_atoms, 3), dtype=np.float64)
            if ga.debug:
                print(f"Rank {ga.comm.Get_rank()} receiving.", flush=True)
            status = MPI.Status()
            ga.comm.Recv([pos, MPI.DOUBLE], source=0, tag=MPI.ANY_TAG, status=status)
            if status.tag == 0:
                break
            clus = ga.utility.generate_cluster(pos)
            opt = ga.local_optimizer(clus, logfile=ga.logfile)
            opt.run(steps=20000)
            if ga.debug:
                print(f"Rank {ga.comm.Get_rank()} sending.", flush=True)
            ga.comm.Send([clus.positions, MPI.DOUBLE], dest=0, tag=2)
