from src.genetic_algorithm import GeneticAlgorithm


def test_GA():
    ga = GeneticAlgorithm(num_clusters=4, atoms=13, preserve=True)
    ga.run(50)
    ga.best_energy()
    ga.best_cluster()
    ga.potentials_history()
