import numpy as np
import pytest
from ase import Atoms
from Disturber import Disturber
from GeneticAlgorithm import GeneticAlgorithm

@pytest.fixture()
def cluster_30():
    positions = []
    for i in range(30):
        positions.append(np.random.rand(3))
    cluster = Atoms('Fe' + str(30), positions=positions)
    return cluster


@pytest.fixture()
def cluster_30_2():
    positions1 = []
    for i in range(30):
        positions1.append(np.random.rand(3))
    cluster1 = Atoms('Fe' + str(30), positions=positions1)
    positions2 = []
    for i in range(30):
        positions2.append(np.random.rand(3))
    cluster2 = Atoms('Fe' + str(30), positions=positions2)
    return cluster1, cluster2


@pytest.fixture()
def disturber():
    glob = GeneticAlgorithm()
    dist = Disturber(glob)
    return dist


def test_cluster_split_30(cluster_30, disturber):
    group1, group2, _ = disturber.split_cluster(cluster_30)
    assert len(group1) + len(group2) == 30
    assert len(set(group1).intersection(set(group2))) == 0


def test_cluster_alignments_30(cluster_30, disturber):
    p1 = cluster_30.positions[0]
    p2 = cluster_30.positions[1]
    distance = np.sqrt(np.sum((p1-p2)**2))
    cluster = disturber.align_cluster(cluster_30)
    assert np.sqrt(np.sum((p1-p2)**2)) - distance < 10**-15
    assert np.sum(np.mean(cluster.positions)) < 10**-15


def test_crossover(cluster_30_2):
    ga = GeneticAlgorithm()
    a, b = ga.crossover(cluster_30_2[0], cluster_30_2[1])
    assert len(a) == len(b)


def test_twist(cluster_30, disturber):
    prev = np.unique(cluster_30.positions)
    disturber.twist(cluster_30)
    cur = np.unique(cluster_30.positions)
    inter = len(np.intersect1d(prev, cur))
    assert inter > 0
    assert inter < 90