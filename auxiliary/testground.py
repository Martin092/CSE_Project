"""Benchmark Local Optimizers Playground"""

import time
from ase.optimize import FIRE
from ase.calculators.lj import LennardJones
from src.fgolo import FGOLO
from src.utility import Utility
from src.basin_hopping_optimizer import BasinHoppingOptimizer


def benchmark_optimizers() -> None:
    """
    Benchmark FGOLO and FIRE Local Optimizers against each other.
    """
    # Set up utility for cluster generation
    basin_hopping = BasinHoppingOptimizer()
    utility = Utility(basin_hopping, "H10")

    # Generate initial atomic configuration
    atoms = utility.generate_cluster()

    # Clone atoms for fair comparison
    atoms_fgolo = atoms.copy()  # type: ignore
    atoms_fgolo.calc = LennardJones()  # type: ignore
    atoms_fire = atoms.copy()  # type: ignore
    atoms_fire.calc = LennardJones()  # type: ignore

    # Initialize and run FGOLO optimizer
    fgolo_optimizer = FGOLO(atoms_fgolo)
    print("\nRunning FGOLO optimizer...")
    start_time_fgolo = time.time()
    fgolo_optimizer.run(fmax=0.01)  # type: ignore
    end_time_fgolo = time.time()

    # Record FGOLO results
    fgolo_final_energy = atoms_fgolo.get_potential_energy()
    fgolo_time = end_time_fgolo - start_time_fgolo

    # Initialize and run FIRE optimizer
    fire_optimizer = FIRE(atoms_fire, logfile=None)
    print("\nRunning FIRE optimizer...")
    start_time_fire = time.time()
    fire_optimizer.run(fmax=0.01)
    end_time_fire = time.time()

    # Record FIRE results
    fire_final_energy = atoms_fire.get_potential_energy()
    fire_time = end_time_fire - start_time_fire

    # Print benchmark results
    print("\n=== Benchmark Results ===")
    print(f"FGOLO: Time = {fgolo_time:.4f} s, Final Energy = {fgolo_final_energy:.6f}")
    print(f"FIRE : Time = {fire_time:.4f} s, Final Energy = {fire_final_energy:.6f}")
