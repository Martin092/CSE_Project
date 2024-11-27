"""GA playground"""

import os

from ase.io import read
from ase.io.trajectory import TrajectoryReader
from ase.visualize import view

from src.genetic_algorithm import GeneticAlgorithm

ga = GeneticAlgorithm()
ga.benchmark_run([5, 13], 20)

if not os.path.exists("clusters/minima_optimized.xyz"):
    os.makedirs("clusters/minima_optimized.xyz")

if not os.path.exists("clusters/minima_progress.traj"):
    os.makedirs("clusters/minima_progress.traj")

final_atoms = read("clusters/minima_optimized.xyz")
view(final_atoms)  # type: ignore

traj = TrajectoryReader("clusters/minima_progress.traj")  # type: ignore
view(traj)  # type: ignore
