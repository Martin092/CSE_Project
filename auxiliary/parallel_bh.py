"""
Parallel implementation of the basin hopping algorithm
On linux can be ran with
>> PYTHONPATH=$PYTHONPATH:/home/martin/Projects/PycharmProjects/CSE_Project
mpiexec -n 10 python auxiliary/parallel_bh.py
"""

import sys
import os
import time
from ase import Atoms
from ase.optimize import FIRE
from ase.io import write
import numpy as np
from mpi4py import MPI  # pylint: disable=E0611

sys.path.append("./")

from src.basin_hopping_optimizer import BasinHoppingOptimizer  # pylint: disable=C0413
from auxiliary.cambridge_database import get_cluster_energy  # pylint: disable=C0413


lj = 13  # pylint: disable=C0103

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print(rank, flush=True)
bh = BasinHoppingOptimizer(local_optimizer=FIRE, comm=comm, debug=False)

start = time.time()
bh.run(max_iterations=1000, num_atoms=lj, atom_type="C", conv_iterations=150)

runtime = time.time() - start
print(
    f"Energy is {bh.best_potential} in process {rank} found in {runtime} for {bh.current_iteration} iterations"
)

if rank == 0:
    best_cluster = bh.best_config
    best_energy = bh.best_potential
    RANK_BEST = 0
    for i in range(size - 1):
        data = np.ones((bh.utility.num_atoms, 3))
        print("Receiving...")
        status = MPI.Status()
        comm.Recv([data, MPI.DOUBLE], tag=MPI.ANY_TAG, status=status)
        rank_curr = status.Get_source()

        new_cluster = Atoms(bh.utility.atom_type + str(bh.utility.num_atoms), positions=data)  # type: ignore
        new_cluster.calc = bh.calculator()
        new_energy = new_cluster.get_potential_energy()  # type: ignore

        if new_energy < best_energy:
            best_cluster = new_cluster
            best_energy = new_energy
            RANK_BEST = rank_curr

    print(f"Best energy is {best_energy} from {RANK_BEST}")
    best = get_cluster_energy(bh.num_atoms, "./")
    print(f"Actual best is {best}")
    if not os.path.exists("./data/optimizer"):
        os.mkdir("./data/optimizer")
    write(f"data/optimizer/LJ{lj}.xyz", best_cluster)

else:
    data = bh.current_cluster.positions
    comm.Send([data, MPI.DOUBLE], dest=0, tag=1)
