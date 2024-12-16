from src.genetic_algorithm import GeneticAlgorithm


def test_ga():
    ga = GeneticAlgorithm(num_clusters=4, preserve=True, debug=True)
    ga.run(13, 'C', 50, conv_iterations=10)
