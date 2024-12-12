"""GA parallel execution module"""

import sys

from mpi4py import MPI  # pylint: disable=E0611
import numpy as np

sys.path.append("./")

from src.genetic_algorithm import GeneticAlgorithm  # pylint: disable=C0413
from auxiliary.benchmark import Benchmark  # pylint: disable=C0413


def parallel_ga(
    num_atoms: int,
    num_iterations: int,
    conv_iters: int,
    num_clusters: int = 8,
    preserve: bool = True,
) -> None:
    """
    Execute GA in parallel using mpiexec.
    :param num_atoms: Number of atoms for which to optimize.
    :param num_iterations: Max number of iterations to execute.
    :param conv_iters: Number of iterations to be considered in the convergence criteria.
    :param num_clusters: Number of clusters per generation/iteration.
    :param preserve: Whether to preserve selected parents in future generation or not.
    :return: None
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    print(f"Hello from {rank}", flush=True)

    ga = GeneticAlgorithm(num_clusters=num_clusters, preserve=preserve, comm=comm)
    ga.setup(num_atoms, "C")

    if rank == 0:
        b = Benchmark(ga)
        b.benchmark_run([num_atoms], num_iterations, conv_iters)

        for i in range(1, comm.Get_size()):
            comm.Send(np.zeros((num_atoms, 3), dtype=float), tag=0, dest=i)

        comm.free()

    else:
        while True:
            print(f"Rank {ga.comm.Get_rank()}", flush=True)  # type: ignore
            pos = np.empty((num_atoms, 3), dtype=np.float64)
            print(f"Rank {ga.comm.Get_rank()} receiving.", flush=True)  # type: ignore
            status = MPI.Status()
            ga.comm.Recv([pos, MPI.DOUBLE], source=0, tag=MPI.ANY_TAG, status=status)  # type: ignore
            if status.tag == 0:
                break
            clus = ga.utility.generate_cluster(pos)
            opt = ga.local_optimizer(clus)
            opt.run(steps=20000)
            print(f"Rank {ga.comm.Get_rank()} sending.", flush=True)  # type: ignore
            ga.comm.Send([clus.positions, MPI.DOUBLE], dest=0, tag=2)  # type: ignore
