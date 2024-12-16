"""
Global minima Database module
"""

import os
import sys
import requests
from ase.io import read
from ase.calculators.lj import LennardJones


def create_xyz_file(atoms: int, atom_type: str) -> str:
    """
    Makes a request to the Cambridge database and creates a .xyz file from it.
    :param atoms: Number of atoms in cluster.
    :param atom_type: Atomic type of cluster.
    :return: The file path.
    """
    name = f"../data/database/LJ{atoms}.xyz"
    if not os.path.exists(name):
        response = requests.get(
            f"http://doye.chem.ox.ac.uk/jon/structures/LJ/points/{atoms}", timeout=2
        )
        print("GET request sent to the database")
        if response.status_code != 200:
            print(f"ERROR: Web request failed with {response}", file=sys.stderr)

        values = response.text
        result = str(atoms) + "\n\n"
        for line in iter(values.splitlines()):
            result += atom_type + line + "\n"

        if not os.path.exists("../data/database"):
            os.mkdir("../data/database")

        with open(name, "w", encoding="UTF-8") as file:
            file.write(result)
    return name


def get_cluster_energy(atoms: int, atom_type: str) -> float:
    """
    Creates a file from the database and prints its energy
    :param atoms: Number of atoms in cluster.
    :param atom_type: Atomic type of cluster.
    :return: Database global minima potential energy.
    """
    filename = create_xyz_file(atoms, atom_type)
    cluster = read(filename)

    cluster.calc = LennardJones()  # type: ignore
    return cluster.get_potential_energy()  # type: ignore
