from ase.io import read, write
from gpaw import GPAW, PW, FermiDirac
from gpaw.tddft import *


def gpw(traj_file):
    si = traj_file
    si.set_pbc([True, True, True])  # Periodic boundary conditions

    # ------------------------------
    atoms = si.copy()

    atoms.center(vacuum=6.0)

    calc = GPAW(mode="fd", nbands=50, h=0.3, symmetry={"point_group": False})
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    calc.write("./gpaw/be_gs.gpw", "all")

    time_step = 8.0  # 1 attoseconds = 0.041341 autime
    iterations = 100  # 2500 x 8 as => 20 fs
    kick_strength = [0.0, 0.0, 1e-3]  # Kick to z-direction

    # Read ground state
    td_calc = TDDFT("./gpaw/be_gs.gpw")

    # Save the time-dependent dipole moment to 'be_dm.dat'
    DipoleMomentWriter(td_calc, "./gpaw/be_dm.dat")

    # Use 'be_td.gpw' as restart file
    RestartFileWriter(td_calc, "./gpawbe_td.gpw")

    # Kick with a delta pulse to z-direction
    td_calc.absorption_kick(kick_strength=kick_strength)

    # Propagate
    td_calc.propagate(time_step, iterations)

    # Save end result to 'be_td.gpw'
    td_calc.write("./gpaw/be_td.gpw", mode="all")

    # Calculate photoabsorption spectrum and write it to 'be_spectrum_z.dat'
    photoabsorption_spectrum("./gpaw/be_dm.dat", "./gpaw/be_spectrum_z.dat")

    time_step = 8.0  # 1 attoseconds = 0.041341 autime
    iterations = 100  # 2500 x 8 as => 20 fs
    # Read restart file with result of previous propagation
    td_calc = TDDFT("./gpaw/be_td.gpw")

    # Append the time-dependent dipole moment to the already existing 'be_dm.dat'
    DipoleMomentWriter(td_calc, "./gpaw/be_dm.dat")

    # Use 'be_td2.gpw' as restart file
    RestartFileWriter(td_calc, "./gpaw/be_td2.gpw")

    # Propagate more
    td_calc.propagate(time_step, iterations)

    # Save end result to 'be_td2.gpw'
    td_calc.write("./gpaw/be_td2.gpw", mode="all")

    photoabsorption_spectrum("./gpaw/be_dm.dat", "./gpaw/be_spectrum_z2.dat")
    return "./gpaw/be_spectrum_z2.dat"
