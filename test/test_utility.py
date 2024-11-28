import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones
from src.genetic_algorithm import GeneticAlgorithm

np.random.seed(0)


@pytest.fixture()
def cluster_2():
    positions = []
    for i in range(2):
        positions.append(np.random.rand(3))
    cluster = Atoms('Fe' + str(2), positions=positions)
    cluster.calc = LennardJones()
    return cluster


@pytest.fixture()
def cluster_30():
    positions = []
    for i in range(30):
        positions.append(np.random.rand(3))
    cluster = Atoms('Fe' + str(30), positions=positions)
    cluster.calc = LennardJones()
    return cluster


@pytest.fixture()
def utility():
    glob = GeneticAlgorithm()
    return glob.utility


def test_cluster_split(cluster_30, utility):
    group1, group2, _ = utility.split_cluster(cluster_30)
    assert len(group1) + len(group2) == 30
    assert len(set(group1).intersection(set(group2))) == 0


def test_cluster_alignment(cluster_30, utility):
    p1 = cluster_30.positions[0]
    p2 = cluster_30.positions[1]
    distance = np.sqrt(np.sum((p1-p2)**2))
    cluster = utility.align_cluster(cluster_30)
    assert np.sqrt(np.sum((p1-p2)**2)) - distance < 10**-15
    assert np.sum(np.mean(cluster.positions)) < 10**-15


def test_md(cluster_2, utility):
    prev = cluster_2.copy()
    prev.calc = LennardJones()
    utility.md(cluster_2, 100, 2, seed=0)
    assert not utility.compare_clusters(prev, cluster_2)


def test_compare_clusters(cluster_30, cluster_2, utility):
    assert utility.compare_clusters(cluster_30, cluster_30)
    assert not utility.compare_clusters(cluster_30, cluster_2)


def test_twist(cluster_30, utility):
    prev = np.unique(cluster_30.positions)
    utility.twist(cluster_30)
    cur = np.unique(cluster_30.positions)
    inter = len(np.intersect1d(prev, cur))
    assert inter > 0
    assert inter < 90
