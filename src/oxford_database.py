"""
TODO: Write this
"""

import sys
import requests  # type: ignore
from ase.io import read
from ase.calculators.lj import LennardJones


def create_xyz_file(atoms: int, atom_type: str) -> str:
    """
    Makes a request to the oxford database and creates a .xyz file from it
    """
    response = requests.get(
        f"http://doye.chem.ox.ac.uk/jon/structures/LJ/points/{atoms}", timeout=2
    )
    if response.status_code != 200:
        print(f"ERROR: Web request failed with {response}", file=sys.stderr)

    values = response.text
    result = str(atoms) + "\n\n"
    for line in iter(values.splitlines()):
        result += atom_type + line + "\n"

    name = f"../oxford_minimas/LJ{atoms}.xyz"
    with open(name, "w", encoding="UTF-8") as file:
        file.write(result)
    return name


def get_cluster_energy(atoms: int, atom_type: str) -> float:
    """
    Creates a file from the database and prints its energy
    """
    filename = create_xyz_file(atoms, atom_type)
    cluster = read(filename)

    cluster.calc = LennardJones()  # type: ignore
    return cluster.get_potential_energy()  # type: ignore


print(get_cluster_energy(8, "C"))
