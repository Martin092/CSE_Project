"""GA playground"""

from ase.io import read
from ase.visualize import view

from src.genetic_algorithm import GeneticAlgorithm
from auxilary.benchmark import Benchmark


ga = GeneticAlgorithm(num_clusters=8, preserve=True)
lj = [13]
b = Benchmark(ga)
b.benchmark_run(lj, 100)

for i in lj:
    final_atoms = read(f"../data/optimizer/LJ{i}.xyz")
    view(final_atoms)  # type: ignore
    database = read(f"../data/oxford_minima/LJ{i}.xyz")
    view(database)
