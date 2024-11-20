import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.optimize import QuasiNewton, BFGS
from ase.optimize.minimahopping import ComparePositions
from ase.units import fs
from ase.md.langevin import Langevin
from ase.io import Trajectory
from ase.io import write
import numpy as np

# I don't like ASE's minima hopping code, it makes 200 files and is weird to read
# so I'm making this based on their code, along with the paper it's based on

def random_cluster(atoms, box_length):
    positions = (np.random.rand(atoms, 3) - 0.5) * box_length * 2
    return positions

#Minima Hopping hyperparams
optimizer = BFGS
beta1 = beta2 = 1.05
beta3 = 1/beta1

alpha1 = 1.02
alpha2 = 1/alpha1

minima = []
minima_threshold = 0.5
temperature = 1000
Ediff = 0.5

def run_MD(atoms, number_of_steps, write_traj=False):
    #This uses Langevin dynamics since that uses the NVT thermostat
    #What are Langevin dynamics? Ask the chemists
    dyn = Langevin(
        atoms,
        timestep=5.0 * fs,
        temperature_K=temperature,  # temperature in K
        friction=0.5 / fs, #not sure how to fuck around with this yet
    )

    if write_traj:
        traj = Trajectory('minima/md.traj', 'w', atoms)
        dyn.attach(traj.write, interval=5)
    dyn.run(number_of_steps)

def save_to_traj(atoms, filename = 'minima/progress.traj'):
    with Trajectory(filename, mode='a') as traj:
        traj.write(atoms)


def youre_staying_in_the_box(atoms, box_length):
    atoms.positions = np.clip(atoms.get_positions(), -box_length, box_length)


#def unique_minimum_position(atoms, minima_threshold=minima_threshold):
#    unique = True
#    dmax_closest = 99999.
#    compare = ComparePositions(translate=True)
#    for minimum in minima:
#        dmax = compare(minimum, atoms)
#        if dmax < minima_threshold:
#            unique = False
#        if dmax < dmax_closest:
#            dmax_closest = dmax
#    return unique, dmax_closest

def optimize(atoms, optimizer=optimizer, fmax=0.02): #I should really check what fmax does
    atoms.set_momenta(np.zeros(atoms.get_momenta().shape))
    with optimizer(atoms, logfile='log.txt') as opt:
        opt.run(fmax=fmax)

def check_results(found_minima, prev_optimum, prev_energy):
    global temperature
    global Ediff
    next_minima = found_minima
    if prev_optimum is not None: #Case 1: We did not find a new minimum
        compare = ComparePositions(translate=False)
        dmax = compare(found_minima, prev_optimum)
        if dmax < minima_threshold:
            temperature *= beta1
        return next_minima, temperature

    #unique, dmax_closest = unique_minimum_position(found_minima)
    #if not unique: #Case 2, new minimum is actually not new, it's one we found before
    #    temperature *= beta2
    #    if prev_optimum is not None:
    #        next_minima = prev_optimum
    #        return next_minima, temperature

    #Case 3, we found a unique minimum
    temperature *= beta3
    # acceptance/rejection step
    if prev_energy is None or found_minima.get_potential_energy() < prev_energy + Ediff:
        Ediff *= alpha1
        minima.append(found_minima)
        #save_to_traj(found_minima)
    else:
        found_minima.positions = prev_optimum.positions
        Ediff *= alpha2

    return next_minima, temperature

if __name__ == "__main__":
    number_of_atoms = 13
    iterations = 100
    box_length = 3

    positions = random_cluster(number_of_atoms, box_length)
    atoms = Atoms('Fe' + str(number_of_atoms), positions=positions)
    atoms.calc = LennardJones()

    write('minima/start.xyz', Atoms('Fe' + str(number_of_atoms), positions=atoms.get_positions()))

    prev_optimum = None
    prev_energy = None

    min_energy = 0
    best_positions = None

    for i in range(iterations):
        run_MD(atoms, 100)
        optimize(atoms)
        check_results(atoms, prev_optimum, prev_energy)

        if(len(atoms.get_positions()) > number_of_atoms):
            print("WHAT THE FUCK HOW HOW HOW")
            quit()

        prev_optimum = atoms.copy()
        prev_energy = atoms.get_potential_energy()

        if i % 10 == 0:
            print("Iteration:", i)

        if atoms.get_potential_energy() < min_energy:
            min_energy = atoms.get_potential_energy()
            best_positions = atoms.get_positions()
            print(min_energy)
            save_to_traj(atoms)

        youre_staying_in_the_box(atoms, box_length)

    write('minima/end.xyz', Atoms('Fe' + str(number_of_atoms), positions=best_positions))


#5 atoms:
#Box length: 3
#iterations: 100
#number of MD steps: 100
#Most times this reaches the optimal configuration, (out of like around 10)
#I've seen it "explode" once (atoms go way too far apart)

#It seems as though ignoring case 2 for checking results may actually result in better outcomes???
#For 5 atoms, it would land on a local minimum more often after I fixed that (around -6)

#Ok the exploding problem is becoming bigger
#I need to find ways to bound the temperature, or the box

#Dividing by temperature every so often doesn't seem to help, I'm gonna try bounding the box
#Bounding the box seems to yield better results?

# The trajectory files keeps showing more atoms than there should be, but I can't seem to catch that in my code???
