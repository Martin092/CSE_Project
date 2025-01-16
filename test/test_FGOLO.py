from src.FGOLO import FGOLO

from ase.optimize import FIRE
from ase import Atoms
from ase.calculators.lj import LennardJones
from src.utility import Utility
from src.basin_hopping_optimizer import BasinHoppingOptimizer
import time

def benchmark_optimizers():
    # Set up utility for cluster generation
    basin_hopping = BasinHoppingOptimizer()
    utility = Utility(basin_hopping, 'H100')

    
    # Generate initial atomic configuration
    atoms = utility.generate_cluster()
    atoms.calc = LennardJones()  # Assign Lennard-Jones potential calculator

    # Clone atoms for fair comparison
    atoms_fgolo = atoms.copy()
    atoms_fgolo.calc = LennardJones()
    atoms_fire = atoms.copy()
    atoms_fire.calc = LennardJones()

    # Initialize and run FGOLO optimizer
    fgolo_optimizer = FGOLO(atoms_fgolo)
    print("\nRunning FGOLO optimizer...")
    start_time_fgolo = time.time()
    fgolo_optimizer.run(fmax=0.01)
    end_time_fgolo = time.time()

    # Record FGOLO results
    fgolo_final_positions = atoms_fgolo.get_positions()
    fgolo_final_energy = atoms_fgolo.get_potential_energy()
    fgolo_time = end_time_fgolo - start_time_fgolo

    # Initialize and run FIRE optimizer
    fire_optimizer = FIRE(atoms_fire, logfile=None)
    print("\nRunning FIRE optimizer...")
    start_time_fire = time.time()
    fire_optimizer.run(fmax=0.01)
    end_time_fire = time.time()

    # Record FIRE results
    fire_final_positions = atoms_fire.get_positions()
    fire_final_energy = atoms_fire.get_potential_energy()
    fire_time = end_time_fire - start_time_fire

    # Print benchmark results
    print("\n=== Benchmark Results ===")
    print(f"FGOLO: Time = {fgolo_time:.4f} s, Final Energy = {fgolo_final_energy:.6f}")
    print(f"FIRE : Time = {fire_time:.4f} s, Final Energy = {fire_final_energy:.6f}")

    # Compare final positions
    # print("\nFGOLO Final Positions:")
    # print(fgolo_final_positions)
    # print("\nFIRE Final Positions:")
    # print(fire_final_positions)


def test_fgolo_optimizer():
    a = BasinHoppingOptimizer()
    utility = Utility(a, 'H4')


    atoms1 = utility.generate_cluster()
    atoms2 = atoms1

    #atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.3]])

    atoms1.calc = LennardJones()

    optimizer = FGOLO(atoms1)

    # Run the optimization
    optimizer.run(fmax=0.01)

    # Check results
    print("Final positions:")
    print(atoms.get_positions())
    print("Final energy:")
    print(atoms.get_potential_energy())


if __name__ == "__main__":
    benchmark_optimizers()