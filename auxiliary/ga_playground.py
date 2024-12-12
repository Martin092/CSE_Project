"""GA empirical testing playground"""

import sys
from ase.io import read
from ase.visualize import view

sys.path.append("./")

from src.genetic_algorithm import GeneticAlgorithm  # pylint: disable=C0413
from auxiliary.benchmark import Benchmark  # pylint: disable=C0413

# from auxiliary.parallel_ga import parallel_ga  # pylint: disable=C0413

# Serial Execution
lj = [38]
ga = GeneticAlgorithm(num_clusters=8, preserve=True)
b = Benchmark(ga)
b.benchmark_run(lj, 100)

# Parallel Execution
# parallel_ga(38, 100, num_clusters=8)

# Visualize Results
for i in lj:
    final_atoms = read(f"../data/optimizer/LJ{i}.xyz")
    view(final_atoms)  # type: ignore
    database = read(f"../data/oxford_minima/LJ{i}.xyz")
    view(database)  # type: ignore
