"""
Parallel implementation of the basin hopping algorithm
"""

import sys
import time
from ase import Atoms
from ase.optimize import BFGS
from ase.io import write
import numpy as np
from mpi4py import MPI  # pylint: disable=E0611

sys.path.append("./")

from src.basin_hopping_optimizer import BasinHoppingOptimizer  # pylint: disable=C0413
from auxiliary.oxford_database import get_cluster_energy  # pylint: disable=C0413


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print(rank)
bh = BasinHoppingOptimizer(local_optimizer=BFGS, atoms=16, atom_type="Fe", comm=comm)

start = time.time()
bh.run(200)

runtime = time.time() - start

energy, cluster = bh.best_energy_cluster()

print(f"Energy is {energy} in process {rank} found in {runtime}")

if rank == 0:
    best_cluster = cluster
    best_energy = energy
    for i in range(size - 1):
        data = np.empty((bh.atoms, 3))
        print("Receiving...")
        comm.Recv([data, MPI.DOUBLE], tag=1)

        if data.shape != (bh.atoms, 3):
            continue

        new_cluster = Atoms(bh.atom_type + str(bh.atoms), positions=data)  # type: ignore
        new_cluster.calc = bh.calculator()
        new_energy = new_cluster.get_potential_energy()  # type: ignore

        if new_energy < best_energy:
            best_cluster = new_cluster
            best_energy = new_energy

    print(f"Best energy is {best_energy}")
    print(f"Actual best is    {get_cluster_energy(bh.atoms, bh.atom_type)}")

    write("clusters/LJ_min.xyz", cluster)
else:
    data = cluster.positions
    comm.Send([data, MPI.DOUBLE], dest=0, tag=1)
