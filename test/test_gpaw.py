from ase.io import read, write
from gpaw import GPAW, PW, FermiDirac
from gpaw.tddft import *


def test_gpw(traj_file):
    si = traj_file
    si.set_pbc([True, True, True]) # Periodic boundary conditions

    #------------------------------
    atoms = si.copy()

    atoms.center(vacuum=6.0)

    calc = GPAW(mode='fd', nbands=20, h=0.3, symmetry={'point_group': False})
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    calc.write('test/gpaw/be_gs.gpw', 'all')
    
    time_step = 8.0                  # 1 attoseconds = 0.041341 autime
    iterations = 100                # 2500 x 8 as => 20 fs
    kick_strength = [0.0,0.0,1e-3]   # Kick to z-direction

    # Read ground state
    td_calc = TDDFT('test/gpaw/be_gs.gpw')

    # Save the time-dependent dipole moment to 'be_dm.dat'
    DipoleMomentWriter(td_calc, 'test/gpaw/be_dm.dat')

    # Use 'be_td.gpw' as restart file
    RestartFileWriter(td_calc, 'test/gpawbe_td.gpw')

    # Kick with a delta pulse to z-direction
    td_calc.absorption_kick(kick_strength=kick_strength)

    # Propagate
    td_calc.propagate(time_step, iterations)

    # Save end result to 'be_td.gpw'
    td_calc.write('test/gpaw/be_td.gpw', mode='all')

    # Calculate photoabsorption spectrum and write it to 'be_spectrum_z.dat'
    photoabsorption_spectrum('test/gpaw/be_dm.dat', 'test/gpaw/be_spectrum_z.dat')
    
    time_step = 8.0                  # 1 attoseconds = 0.041341 autime
    iterations = 100                # 2500 x 8 as => 20 fs
    # Read restart file with result of previous propagation
    td_calc = TDDFT('test/gpaw/be_td.gpw')

    # Append the time-dependent dipole moment to the already existing 'be_dm.dat'
    DipoleMomentWriter(td_calc, 'test/gpaw/be_dm.dat')

    # Use 'be_td2.gpw' as restart file
    RestartFileWriter(td_calc, 'test/gpaw/be_td2.gpw')

    # Propagate more
    td_calc.propagate(time_step, iterations)

    # Save end result to 'be_td2.gpw'
    td_calc.write('test/gpaw/be_td2.gpw', mode='all')

    photoabsorption_spectrum('test/gpaw/be_dm.dat', 'test/gpaw/be_spectrum_z2.dat')
    return 'test/gpaw/be_spectrum_z2.dat'
    
'''
    import matplotlib.pyplot as plt

    energies = calc.get_energies()  # Energies in eV
    values = calc.get_dos()  # DOS values

    plt.plot(energies, values)
    plt.xlabel('Energy (eV)')
    plt.ylabel('DOS (states/eV)')
    plt.title('Density of States')
    plt.show()

    
'''
