"""GA playground"""

import os
import sys

from ase.io import read
from ase.io.trajectory import TrajectoryReader
from ase.visualize import view

from src.genetic_algorithm import GeneticAlgorithm

sys.path.append("./")

if not os.path.exists("clusters/"):
    os.makedirs("clusters/")

ga = GeneticAlgorithm(num_clusters=8)
ga.benchmark_run([38], 100)

final_atoms = read("clusters/minima_optimized.xyz")
view(final_atoms)  # type: ignore

traj = TrajectoryReader("clusters/minima_progress.traj")  # type: ignore
view(traj)  # type: ignore