"""GA playground"""

import sys

from ase import Atoms
from ase.io import read
from ase.visualize import view
from mpi4py import MPI  # pylint: disable=E0611
import numpy as np

sys.path.append("./")

from src.genetic_algorithm import GeneticAlgorithm  # pylint: disable=C0413
from auxiliary.benchmark import Benchmark  # pylint: disable=C0413

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

lj = [30]

print(f"Hello from {rank}", flush=True)

ga = GeneticAlgorithm(num_clusters=8, atoms=30, preserve=True, comm=comm)

if rank == 0:

    b = Benchmark(ga)
    b.benchmark_run(lj, 100)

    for i in range(1, comm.Get_size()):
        comm.Send(np.zeros((ga.atoms, 3), dtype=float), tag=0, dest=i)

    comm.free()

    for i in lj:
        final_atoms = read(f"../data/optimizer/LJ{i}.xyz")
        view(final_atoms)  # type: ignore
        database = read(f"../data/oxford_minima/LJ{i}.xyz")
        view(database)  # type: ignore

else:
    while True:
        print(f"Rank {ga.comm.Get_rank()}", flush=True)  # type: ignore
        pos = np.empty((ga.atoms, 3), dtype=np.float64)
        print(f"Rank {ga.comm.Get_rank()} receiving.", flush=True)  # type: ignore
        status = MPI.Status()
        ga.comm.Recv([pos, MPI.DOUBLE], source=0, tag=MPI.ANY_TAG, status=status)  # type: ignore
        if status.tag == 0:
            break
        clus = Atoms(  # type: ignore
            ga.atom_type + str(ga.atoms),
            positions=pos,
            cell=np.array(
                [[ga.box_length, 0, 0], [0, ga.box_length, 0], [0, 0, ga.box_length]]
            ),
        )
        clus.calc = ga.calculator()
        opt = ga.local_optimizer(clus)
        opt.run(steps=20000)
        print(f"Rank {ga.comm.Get_rank()} sending.", flush=True)  # type: ignore
        ga.comm.Send([clus.positions, MPI.DOUBLE], dest=0, tag=2)  # type: ignore
