from collections import OrderedDict
from src.genetic_algorithm import GeneticAlgorithm


def test_ga():
    mutation = OrderedDict(
        [
            ("twist", 0.3),
            ("random displacement", 0.1),
            ("angular", 0.3),
            ("random step", 0.3),
            ("etching", 0.1),
        ]
    )
    ga = GeneticAlgorithm(mutation=mutation, num_clusters=4, preserve=True, debug=True)
    ga.run(13, 'C', 50, conv_iterations=10)
