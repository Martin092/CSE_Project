import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from ase.calculators.lj import LennardJones
from ase.io import write
from ase.io import read
from rotation_matrices import rotation_y, rotation_x, rotation_z

LJ_minima = {
    7: -16.505384,
    13: -44.326801,
    19: -72.659782,
    20: -77.177043,
    55: -279.248470,
}


def random_cluster(atoms):
    radius = 2
    postitions = (np.random.rand(atoms, 3) - 0.5) * 2 * radius
    return postitions


atoms = 13
iterations = 100


def one_atom_move_directed(cluster, stepSize):
    energies = cluster.get_potential_energies()
    highest_index = np.argmax(energies)

    for i in range(100):
        mean_cluster = (
            np.mean(cluster.positions, axis=0)
            - cluster.positions[highest_index] / atoms
        )
        step = (np.random.rand(3) - 0.5) * 2 * stepSize + mean_cluster
        step = (step / np.linalg.norm(step)) * stepSize
        cluster.positions[highest_index] += step

    return cluster


def one_atom_move(cluster, stepSize):
    energies = cluster.get_potential_energies()
    highest_index = np.argmax(energies)

    for i in range(15):
        step = (np.random.rand(3) - 0.5) * 2 * stepSize
        cluster.positions[highest_index] += step

    return cluster


def one_atom_move_bounded(cluster, stepSize, radius=2):
    energies = cluster.get_potential_energies()
    highest_index = np.argmax(energies)

    for i in range(150):
        step = (np.random.rand(3) - 0.5) * 2 * stepSize
        cluster.positions[highest_index] += step
        if (cluster.positions[highest_index] > radius).any():
            cluster.positions[highest_index] -= step

    return cluster


def rotation(cluster, stepSize, radius=2):
    energies = cluster.get_potential_energies()
    highest_index = np.argmax(energies)
    mean_cluster = (
        np.mean(cluster.positions, axis=0) - cluster.positions[highest_index] / atoms
    )

    positions = cluster.positions - mean_cluster
    angle = (np.random.rand(3) - 0.5) * 2 * np.pi / 5

    atom = positions[highest_index]
    atom = np.matmul(rotation_x(angle[0]), atom)
    atom = np.matmul(rotation_y(angle[1]), atom)
    atom = np.matmul(rotation_z(angle[2]), atom)

    positions[highest_index] = atom

    cluster.positions = positions
    return cluster


def global_optimization(atoms, iterations, disturbance, stepSize=0.1, positions=None):
    minima = float("inf")
    best_pos_before = 0
    best_pos_after = 0
    local_minima_found = 0

    if positions is None:
        positions = random_cluster(atoms)

    cluster = Atoms("Fe" + str(atoms), positions=positions)
    cluster.calc = LennardJones()
    for i in range(iterations):
        step = (np.random.rand(3) - 0.5) * 2 * stepSize
        cluster.positions -= step

        energies = cluster.get_potential_energies()
        highest_index = np.argmax(energies)
        lowest_energy = np.min(energies)

        alpha = 1.4
        if abs(lowest_energy) < alpha * abs(energies[highest_index]):
            cluster = rotation(cluster, stepSize)
        else:
            cluster = one_atom_move_directed(cluster, 0.01)

        before = cluster.positions
        opt = BFGS(cluster, logfile="log.txt")
        opt.run(fmax=0.02)

        after = cluster.get_potential_energy()
        if after < minima:
            minima = after
            best_pos_before = positions
            best_pos_after = cluster.get_positions()
            local_minima_found += 1
        else:
            cluster.set_positions(before)

    print(f"{local_minima_found} local minima found")
    write(
        "minima/basin_before.xyz", Atoms("Fe" + str(atoms), positions=best_pos_before)
    )

    final_cluster = Atoms("Fe" + str(atoms), positions=best_pos_after)
    final_cluster.calc = LennardJones()

    energies = cluster.get_potential_energies()
    highest_index = np.argmax(energies)
    lowest_energy = np.min(energies)

    print(f"Lowest:{lowest_energy}; highest:{energies[highest_index]}")

    print(final_cluster.get_potential_energy())
    write("minima/basin_optimized.xyz", final_cluster)
    return final_cluster


for i in range(10):
    global_optimization(13, 100, rotation)
