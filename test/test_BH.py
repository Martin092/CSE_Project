from src.basin_hopping_optimizer import BasinHoppingOptimizer


def test_bh():
    bh = BasinHoppingOptimizer(debug=True)
    bh.run(13, 'C', 50)
