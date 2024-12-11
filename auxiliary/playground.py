"""GA playground"""

from src.genetic_algorithm import GeneticAlgorithm
from src.basin_hopping_optimizer import BasinHoppingOptimizer
from auxiliary.benchmark import Benchmark


ga = GeneticAlgorithm(num_clusters=8, preserve=True)
bh = BasinHoppingOptimizer(atoms=13, atom_type="C")
lj = [39]
b = Benchmark(bh)
b.benchmark_run(lj, 500)

# for i in lj:
#     final_atoms = read(f"../data/optimizer/LJ{i}.xyz")
#     view(final_atoms)  # type: ignore
#     database = read(f"../data/oxford_minima/LJ{i}.xyz")
#     view(database)  # type: ignore
