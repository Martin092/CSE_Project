"""GA empirical testing playground"""

import sys
from collections import OrderedDict
from ase.io import read
from ase.io.trajectory import TrajectoryReader
from ase.visualize import view

sys.path.append("./")

from src.genetic_algorithm import GeneticAlgorithm  # pylint: disable=C0413
from auxiliary.benchmark import Benchmark  # pylint: disable=C0413
from auxiliary.parallel_ga import parallel_ga  # pylint: disable=C0413

mutation = OrderedDict(
    [
        ("twist", 0.3),
        ("random displacement", 0.1),
        ("angular", 0.3),
        ("random step", 0.3),
        ("etching", 0.1),
    ]
)

ga = GeneticAlgorithm(mutation=mutation, num_clusters=32, debug=True)

# Serial Execution
lj = [13]
b = Benchmark(ga)
b.benchmark_run(lj, 100, 10)

# Parallel Execution
parallel_ga(ga, 13, "C", 100, 20, max_local_steps=10000)

# Visualize Results
final_atoms = read(f"../data/optimizer/LJ{38}.xyz")
view(final_atoms)  # type: ignore
database = read(f"../data/database/LJ{38}.xyz")
view(database)  # type: ignore
traj = TrajectoryReader(f"../data/optimizer/LJ{38}.traj")  # type: ignore
view(traj)  # type: ignore
