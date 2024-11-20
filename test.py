import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones
from Disturber import Disturber
from GeneticAlgorithm import GeneticAlgorithm

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
def cluster_50():
    positions = []
    for i in range(50):
        positions.append(np.random.rand(3))
    cluster = Atoms('Fe' + str(50), positions=positions)
    cluster.calc = LennardJones()
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
    cluster1.calc = LennardJones()
    cluster2.calc = LennardJones()
    return cluster1, cluster2


def test_cluster_split_30(cluster_30):
    group1, group2, _ = Disturber.split_cluster(cluster_30)
    assert len(group1) + len(group2) == 30
    assert len(set(group1).intersection(set(group2))) == 0


def test_cluster_split_50(cluster_50):
    group1, group2, _ = Disturber.split_cluster(cluster_50)
    assert len(group1) + len(group2) == 50
    assert len(set(group1).intersection(set(group2))) == 0

def test_cluster_alignments_30(cluster_30):
    p1 = cluster_30.positions[0]
    p2 = cluster_30.positions[1]
    distance = np.sqrt(np.sum((p1-p2)**2))
    cluster = Disturber.align_cluster(cluster_30)
    assert np.sqrt(np.sum((p1-p2)**2)) - distance < 10**-15
    assert np.sum(np.mean(cluster.positions)) < 10**-15

def test_cluster_alignments_50(cluster_50):
    p1 = cluster_50.positions[0]
    p2 = cluster_50.positions[1]
    distance = np.sqrt(np.sum((p1-p2)**2))
    cluster = Disturber.align_cluster(cluster_50)
    assert np.sqrt(np.sum((p1-p2)**2)) - distance < 10**-15
    assert np.sum(np.mean(cluster.positions)) < 10**-15

def test_crossover(cluster_30_2):
    a, b = GeneticAlgorithm.crossover(cluster_30_2[0], cluster_30_2[1])
    assert len(a) == len(b)

def test_md(cluster_2):
    Disturber.md(cluster_2, 100, 2, seed=0)
    resulting_positions = np.array([[5665.16673931,420178.45149765,-62162.09537009],[-5664.07304262,-420177.31265348,62163.34402758]])
    assert np.isclose(cluster_2.positions, resulting_positions).all()