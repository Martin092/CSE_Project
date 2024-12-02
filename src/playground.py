"""GA playground"""

import os,sys

sys.path.append('./')

from ase.io import read
from ase.io.trajectory import TrajectoryReader
from ase.visualize import view

from src.genetic_algorithm import GeneticAlgorithm

if not os.path.exists("clusters/"):
    os.makedirs("clusters/")

ga = GeneticAlgorithm(num_clusters=4)
ga.benchmark_run([38], 100)

final_atoms = read("clusters/minima_optimized.xyz")
view(final_atoms)  # type: ignore

traj = TrajectoryReader("clusters/minima_progress.traj")  # type: ignore
view(traj)  # type: ignore
