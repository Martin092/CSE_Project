import os, sys
sys.path.append('./')

from src.genetic_algorithm import GeneticAlgorithm
from ase.visualize import view


def test_ga():
    ga = GeneticAlgorithm(num_clusters=4, preserve=True)
    ga.run(13, 'C', 50)
    atoms = ga.cluster_list[-1]
    view(atoms)

test_ga()


#python test/test_GA.py
#set PYDEVD_DISABLE_FILE_VALIDATION=1  