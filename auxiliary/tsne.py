import sys
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.visualize import view
from ase.calculators.lj import LennardJones
from ase.calculators.emt import EMT
from sklearn.manifold import TSNE

sys.path.append("./")
from collections import OrderedDict
from src.genetic_algorithm import GeneticAlgorithm
from ase.visualize import view
from src.basin_hopping_optimizer import BasinHoppingOptimizer
from ase.optimize import FIRE

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from numpy import log
from scipy.interpolate import griddata
import pickle

mutation = OrderedDict(
    [
        ("twist", 0.3),
        ("random displacement", 0.2),
        ("angular", 0.3),
        ("random step", 0.2),
        ("etching", 0.1),
    ]
)

# ===================================================================================

limit = -1
division = 1
iterations = 10 // division
iterations2 = 50# // division
iterations3 = 10 // division
max_energy = 50
randomness = 3
num_atoms = 13
clusterS = f'C{num_atoms}'
filename = f'clusters_and_energiesC{num_atoms}(2).pkl'
interpolation_type = 'cubic'
#interpolation_type = 'linear'
#interpolation_type = 'nearest'
layers = 10
min_out_of_limits_range = 0#0.2
out_of_limits_range = -0.5#0.2
calc = LennardJones
#calc = EMT

# ===================================================================================

def compute_clusters():
    clusters = []
    energies = []
    for i in range(iterations):
        print("Iteration: ", i)
        ga = GeneticAlgorithm(mutation=mutation, num_clusters=2, preserve=True)
        prob = np.random.rand()
        if prob < 0.3:
            ga.run(clusterS, 1, conv_iterations=1)
        elif prob < 0.6 and prob >= 0.3:
            ga.run(clusterS, 5, conv_iterations=5)
        elif prob >= 0.6 and prob < 0.9:
            ga.run(clusterS, 10, conv_iterations=7)
        else:
            ga.run(clusterS, 50, conv_iterations=10)
        clusters.append(ga.best_config)
        energies.append(ga.energies[-1])

    for i in range(iterations3):
        print("Iteration: ", i)
        bh = BasinHoppingOptimizer(local_optimizer=FIRE)
        bh.run(clusterS, 50)
        for cluster in bh.configs:
            cluster.calc = calc()
            clusters.append(cluster)
            energies.append(cluster.get_potential_energy())

    for i in range(iterations2):
        print("Iteration: ", i)
        cluster = Atoms(clusterS, positions=((np.random.rand(num_atoms, 3) * 2) - 1) * np.random.uniform(1, randomness))
        cluster.calc = calc()
        clusters.append(cluster)
        energies.append(cluster.get_potential_energy())

    filtered_clusters = []
    filtered_energies = []

    for cluster, energy in zip(clusters, energies):
        if energy <= max_energy:
            filtered_clusters.append(cluster)
            filtered_energies.append(energy)

    clusters = filtered_clusters
    energies = filtered_energies
    positions = [cluster.positions for cluster in clusters]

    min_energy_index = np.argmin(energies)
    min_energy_cluster = clusters[min_energy_index]

    num_copies = 20  # Number of copies
    for _ in range(num_copies):
        clusters.append(min_energy_cluster)
        energies.append(min(energies))

    print("Energies: ", energies)
    print("Min Energy: ", min(energies))
    print("Max Energy: ", max(energies))
    # print("Positions: ", positions)
    return clusters, energies, min_energy_index

# Check if the clusters and energies are already saved

try:
    with open(filename, 'rb') as f:
        clusters, energies, min_energy_index = pickle.load(f)
except EOFError:
    clusters, energies, min_energy_index = compute_clusters()
    with open(filename, 'wb') as f:
        pickle.dump((clusters, energies, min_energy_index), f)
    print("Computed and saved clusters and energies due to EOFError.")
except FileNotFoundError:
    clusters, energies, min_energy_index = compute_clusters()
    # Save the clusters and energies to a file
    with open(filename, 'wb') as f:
        pickle.dump((clusters, energies, min_energy_index), f)
    print("Computed and saved clusters and energies.")


def compute_features(atoms):
    atoms.calc = calc()
    flattened_positions = atoms.positions.flatten()  

    center_of_mass = atoms.get_center_of_mass() 
    distances_from_com = np.linalg.norm((atoms.positions - center_of_mass), axis=1)

    distances = np.linalg.norm(atoms.positions[:, np.newaxis, :] - atoms.positions[np.newaxis, :, :], axis=2)
    pairwise_distances = distances[np.triu_indices(len(atoms), k=1)]

    angles = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            for k in range(j + 1, len(atoms)):
                vec1 = atoms.positions[j] - atoms.positions[i]
                vec2 = atoms.positions[k] - atoms.positions[i]
                cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
                angles.append(angle)
    angles = np.array(angles)

    features = np.concatenate((
        flattened_positions, 
        distances_from_com * 1, 
        pairwise_distances * 1, 
        angles * 1, 
    ))

    return features

features_list = []

for atoms in clusters:
    features = compute_features(atoms)  
    features_list.append(features)

features_array = np.array(features_list)
tsne = TSNE(n_components=2)
reduced_features = tsne.fit_transform(features_array)


x = reduced_features[:, 0]
y = reduced_features[:, 1]

xi = np.linspace(np.min(x), np.max(x), 100)  # 100 points along the x-axis
yi = np.linspace(np.min(y), np.max(y), 100)  # 100 points along the y-axis
xi, yi = np.meshgrid(xi, yi) 

zi = griddata((x, y), energies, (xi, yi), method=interpolation_type)
zi_out_1 = np.copy(zi)
zi_out_1[zi >= min(energies) * (1 + min_out_of_limits_range)] = np.nan
zi[zi < min(energies) * (1 + min_out_of_limits_range)] = np.nan

zi_out_2 = np.copy(zi)
if max(energies) > 0:
    zi_out_2[zi <= max(energies) * (1 + out_of_limits_range)] = np.nan
    zi[zi > max(energies) * (1 + out_of_limits_range)] = np.nan
else:
    zi_out_2[zi <= max(energies) * (1 - out_of_limits_range)] = np.nan
    zi[zi > max(energies) * (1 - out_of_limits_range)] = np.nan

def plot():
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(xi, yi, zi, levels=layers, cmap='viridis')
    contour2 = plt.contourf(xi, yi, zi_out_1, levels=layers, cmap='gray')
    contour3 = plt.contourf(xi, yi, zi_out_2, levels=layers, cmap='gray')
    
    plt.legend()
    plt.colorbar(contour, label='Energy (eV)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('2D t-SNE with Energy Contour Plot')
    if limit is None:
        x_margin = (xi.max() - xi.min()) * 0.1
        y_margin = (yi.max() - yi.min()) * 0.1
        plt.xlim(xi.min() - x_margin, xi.max() + x_margin)
        plt.ylim(yi.min() - y_margin, yi.max() + y_margin)
    elif limit > 0:
        plt.xlim(-limit, limit)
        plt.ylim(-limit, limit)
    else:
        pass


    '''
    max_energy = max(energies)
    threshold_energy_max = max_energy  # 10% above the maximum energy
    contour_lines_max = plt.contour(xi, yi, zi, levels=[threshold_energy_max], colors='red', linestyles='dashdot')
    plt.clabel(contour_lines_max, fmt='%2.1f', colors='green', fontsize=12)

    min_energy = min(energies)
    threshold_energy_min = min_energy * 0.9  # 10% below the minimum energy
    contour_lines_min = plt.contour(xi, yi, zi, levels=[threshold_energy_min], colors='orange', linestyles='dotted')
    plt.clabel(contour_lines_min, fmt='%2.1f', colors='blue', fontsize=12)

    # Scatter only the 10% lower energy values
    min_indices = np.argsort(energies)[:5]
    plt.scatter(x[min_indices], y[min_indices], color='black', label='5 Minimum Energies')
    '''
    min_indice = np.argmin(energies)
    plt.scatter(x[min_indice], y[min_indice], color='red', s=100, edgecolor='black', label='Minimum Energy')

    contour = plt.contourf(xi, yi, zi, levels=np.linspace(np.nanmin(zi), np.nanmax(zi), layers * 2), cmap='viridis')
    plt.savefig(f'{clusterS}.png')

plot()