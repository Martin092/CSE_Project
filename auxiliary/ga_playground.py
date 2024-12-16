"""GA empirical testing playground"""

import sys
from ase.io import read
from ase.io.trajectory import TrajectoryReader
from ase.visualize import view

sys.path.append("./")

from src.genetic_algorithm import GeneticAlgorithm  # pylint: disable=C0413
from auxiliary.benchmark import Benchmark  # pylint: disable=C0413
from auxiliary.parallel_ga import parallel_ga  # pylint: disable=C0413

# Serial Execution
lj = [13]
ga = GeneticAlgorithm(num_clusters=8, preserve=True, debug=True)
b = Benchmark(ga)
b.benchmark_run(lj, 100)

# Parallel Execution
parallel_ga(13, 100, 10, num_clusters=8, debug=True)

# Visualize Results
final_atoms = read(f"../data/optimizer/LJ{13}.xyz")
view(final_atoms)  # type: ignore
database = read(f"../data/database/LJ{13}.xyz")
view(database)  # type: ignore
traj = TrajectoryReader(f"../data/optimizer/LJ{13}.traj")
view(traj)
