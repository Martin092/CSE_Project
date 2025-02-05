"""GA empirical testing playground"""

import sys
from collections import OrderedDict
from ase.io import read
from ase.io.trajectory import TrajectoryReader
from ase.visualize import view

from auxiliary.parallel_ga import parallel_ga
from src.basin_hopping_optimizer import BasinHoppingOptimizer

sys.path.append("./")

from src.genetic_algorithm import GeneticAlgorithm  # pylint: disable=C0413
from auxiliary.benchmark import Benchmark  # pylint: disable=C0413

mutation = OrderedDict(
    [
        ("twist", 0.3),
        ("random displacement", 0.1),
        ("angular", 0.3),
        ("random step", 0.3),
        ("etching", 0.1),
    ]
)

ga = GeneticAlgorithm(mutation=mutation, num_clusters=4)
lj = [13, 19, 26]

# Serial Execution
b = Benchmark(ga)
b.benchmark_run(lj, 100, 10)

# Parallel Execution
for i in lj:
    bh = BasinHoppingOptimizer()
    parallel_ga(ga, "C" + str(i), 100, 10, max_local_steps=10000, bh_optimizer=bh)

# Visualize Results (run serially)
for i in lj:
    final_atoms = read(f"../data/optimizer/LJ{i}.xyz")
    view(final_atoms)  # type: ignore
    database = read(f"../data/database/LJ{i}.xyz")
    view(database)  # type: ignore
    traj = TrajectoryReader(f"../data/optimizer/LJ{i}.traj")  # type: ignore
    view(traj)  # type: ignore
