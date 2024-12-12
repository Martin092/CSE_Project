from src.basin_hopping_optimizer import BasinHoppingOptimizer


def test_ga():
    ga = BasinHoppingOptimizer()
    ga.run(13, 'C', 50)
