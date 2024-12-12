from src.minima_hopping_optimizer import MinimaHoppingOptimizer


def test_mh():
    mh = MinimaHoppingOptimizer()
    mh.run(13, 'C', 50)
