"""
Parallel implementation of the basin hopping algorithm
On linux can be ran with
>> PYTHONPATH=$PYTHONPATH:/home/martin/Projects/PycharmProjects/CSE_Project mpiexec -n 10 python auxiliary/parallel_bh.py
"""

import sys
import time
from ase import Atoms
from ase.optimize import FIRE
from ase.io import write
import numpy as np
from mpi4py import MPI  # pylint: disable=E0611

sys.path.append("./")

from src.basin_hopping_optimizer import BasinHoppingOptimizer  # pylint: disable=C0413
from auxiliary.cambridge_database import get_cluster_energy  # pylint: disable=C0413


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print(rank, flush=True)
bh = BasinHoppingOptimizer(local_optimizer=FIRE, comm=comm)

start = time.time()
bh.run(max_iterations=10, num_atoms=13, atom_type="Fe")

runtime = time.time() - start
print(f"Energy is {bh.best_potential} in process {rank} found in {runtime}")

if rank == 0:
    best_cluster = bh.best_config
    best_energy = bh.best_potential
    for i in range(size - 1):
        data = np.empty((bh.utility.num_atoms, 3))
        print("Receiving...")
        comm.Recv([data, MPI.DOUBLE], tag=1)

        if data.shape != (bh.utility.num_atoms, 3):
            continue

        new_cluster = Atoms(bh.utility.atom_type + str(bh.utility.num_atoms), positions=data)  # type: ignore
        new_cluster.calc = bh.calculator()
        new_energy = new_cluster.get_potential_energy()  # type: ignore

        if new_energy < best_energy:
            best_cluster = new_cluster
            best_energy = new_energy

    print(f"Best energy is {best_energy}")
    print(f"Actual best is    {get_cluster_energy(bh.num_atoms, bh.atom_type)}")

    write("src/clusters/LJ_min.xyz", best_cluster)
else:
    data = bh.current_cluster.positions
    comm.Send([data, MPI.DOUBLE], dest=0, tag=1)
