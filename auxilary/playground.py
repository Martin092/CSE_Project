"""GA playground"""

import os
import sys

from ase.io import read
from ase.visualize import view

from src.genetic_algorithm import GeneticAlgorithm


if not os.path.exists("../data/GA_clusters/"):
    os.makedirs("../data/GA_clusters/")

ga = GeneticAlgorithm(num_clusters=4)
lj = [13]
ga.benchmark_run(lj, 100, 13)

for i in lj:
    final_atoms = read(f"../data/GA_clusters/LJ{i}.xyz")
    view(final_atoms)  # type: ignore
    database = read(f"../data/oxford_minima/LJ{i}.xyz")
    view(database)
