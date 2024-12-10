"""GA playground"""

import os
import sys

from ase.io import read
from ase.visualize import view

from src.genetic_algorithm import GeneticAlgorithm

sys.path.append("./")

if not os.path.exists("clusters/"):
    os.makedirs("clusters/")

ga = GeneticAlgorithm(num_clusters=8)
ga.benchmark_run([55], 100, 13)

final_atoms = read("clusters/minima_optimized.xyz")
view(final_atoms)  # type: ignore
