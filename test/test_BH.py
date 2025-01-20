from src.basin_hopping_optimizer import BasinHoppingOptimizer
from src.fgolo import FGOLO

def test_bh():
    bh = BasinHoppingOptimizer(local_optimizer=FGOLO)
    bh.run('C13', 50)
