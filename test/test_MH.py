from src.minima_hopping_optimizer import MinimaHoppingOptimizer


def test_mh():
    mh = MinimaHoppingOptimizer(debug=True)
    mh.run('C13', 50)
