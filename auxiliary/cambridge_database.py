"""
Global minima Database module
"""

import os
import sys

import numpy as np
import requests
from ase.io import read
from ase.calculators.lj import LennardJones


def create_xyz_file(atoms: int, root: str = "../") -> str:
    """
    Makes a request to the Cambridge database and creates a .xyz file from it.
    :param atoms: Number of atoms in cluster.
    :param root: Directory root folder.
    :return: The file path.
    """
    name = root + f"data/database/LJ{atoms}.xyz"
    if not os.path.exists(name):
        try:
            response = requests.get(
                f"http://doye.chem.ox.ac.uk/jon/structures/LJ/points/{atoms}",
                timeout=10,
            )
        except requests.exceptions.ConnectionError:
            print(
                "ERROR: Web request failed, please check your internet connection. Setting minimum to infinity"
            )
            return ""
        print("GET request sent to the database")
        if response.status_code != 200:
            print(f"ERROR: Web request failed with {response}", file=sys.stderr)

        values = response.text
        result = str(atoms) + "\n\n"
        for line in iter(values.splitlines()):
            result += "C" + line + "\n"

        if not os.path.exists(root + "data/database"):
            if not os.path.exists(root + "data"):
                os.mkdir(root + "data")
            os.mkdir(root + "data/database")

        with open(name, "w", encoding="UTF-8") as file:
            file.write(result)
    return name


def get_cluster_energy(atoms: int, root: str = "../") -> float:
    """
    Creates a file from the database and prints its energy
    :param atoms: Number of atoms in cluster.
    :param root: Directory root folder.
    :return: Database global minima potential energy.
    """
    filename = create_xyz_file(atoms, root)
    if filename == "":
        return np.inf
    cluster = read(filename)

    cluster.calc = LennardJones()  # type: ignore
    return cluster.get_potential_energy()  # type: ignore
