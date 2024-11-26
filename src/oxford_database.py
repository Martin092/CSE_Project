import requests
import sys
from ase import Atoms
from ase.io import read
from ase.calculators.lj import LennardJones

def create_xyz_file(atoms, atom_type):
    response = requests.get(f'http://doye.chem.ox.ac.uk/jon/structures/LJ/points/{atoms}')
    if response.status_code != 200:
        print(f"ERROR: Web request failed with {response}", file=sys.stderr)

    values = response.text
    result = str(atoms) + "\n\n"
    for line in iter(values.splitlines()):
        result += atom_type + line + "\n"

    name = f"oxford_minimas/LJ{atoms}.xyz"
    with open(name, 'w') as file:
        file.write(result)
    return name

def get_cluster_energy(atoms, atom_type):
    filename = create_xyz_file(atoms, atom_type)
    cluster = read(filename)

    cluster.calc = LennardJones()
    print(cluster.get_potential_energy())


get_cluster_energy(38, "C")