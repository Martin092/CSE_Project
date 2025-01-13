from src.FGOLO import FGOLO
from ase import Atoms
from ase.calculators.lj import LennardJones

def test_fgolo_optimizer():
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.8]])

    atoms.calc = LennardJones()

    optimizer = FGOLO(atoms)

    # Run the optimization
    optimizer.run(fmax=0.01)

    # Check results
    print("Final positions:")
    print(atoms.get_positions())
    print("Final energy:")
    print(atoms.get_potential_energy())

if __name__ == "__main__":
    test_fgolo_optimizer()