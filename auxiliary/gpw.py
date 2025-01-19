"""GPAW calculator module"""

from ase import Atoms
from gpaw import GPAW  # type: ignore  # pylint: disable=E0401,E0611
from gpaw.tddft import *  # type: ignore  # pylint: disable=E0401,E0611,W0401


def gpw(atoms: Atoms) -> None:
    """
    GPAW calculator method
    """
    atoms.center(vacuum=6.0)  # type: ignore
    atoms.set_pbc([True, True, True])  # type: ignore

    calc = GPAW(mode="fd", nbands=50, h=0.3, symmetry={"point_group": False})
    atoms.calc = calc
    atoms.get_potential_energy()  # type: ignore
    calc.write("./gpaw/be_gs.gpw", "all")

    time_step = 8.0  # 1 attoseconds = 0.041341 autime
    iterations = 100  # 2500 x 8 as => 20 fs
    kick_strength = [0.0, 0.0, 1e-3]  # Kick to z-direction

    # Read ground state
    td_calc = TDDFT("./gpaw/be_gs.gpw")  # type: ignore  # pylint: disable=E0602

    # Save the time-dependent dipole moment to 'be_dm.dat'
    DipoleMomentWriter(td_calc, "./gpaw/be_dm.dat")  # type: ignore  # pylint: disable=E0602

    # Use 'be_td.gpw' as restart file
    RestartFileWriter(td_calc, "./gpawbe_td.gpw")  # type: ignore  # pylint: disable=E0602

    # Kick with a delta pulse to z-direction
    td_calc.absorption_kick(kick_strength=kick_strength)

    # Propagate
    td_calc.propagate(time_step, iterations)

    # Save end result to 'be_td.gpw'
    td_calc.write("./gpaw/be_td.gpw", mode="all")

    # Calculate photoabsorption spectrum and write it to 'be_spectrum_z.dat'
    photoabsorption_spectrum("./gpaw/be_dm.dat", "./gpaw/be_spectrum_z.dat")  # type: ignore  # pylint: disable=E0602

    time_step = 8.0  # 1 attoseconds = 0.041341 autime
    iterations = 100  # 2500 x 8 as => 20 fs
    # Read restart file with result of previous propagation
    td_calc = TDDFT("./gpaw/be_td.gpw")  # type: ignore  # pylint: disable=E0602

    # Append the time-dependent dipole moment to the already existing 'be_dm.dat'
    DipoleMomentWriter(td_calc, "./gpaw/be_dm.dat")  # type: ignore  # pylint: disable=E0602

    # Use 'be_td2.gpw' as restart file
    RestartFileWriter(td_calc, "./gpaw/be_td2.gpw")  # type: ignore  # pylint: disable=E0602

    # Propagate more
    td_calc.propagate(time_step, iterations)

    # Save end result to 'be_td2.gpw'
    td_calc.write("./gpaw/be_td2.gpw", mode="all")

    photoabsorption_spectrum("./gpaw/be_dm.dat", "./gpaw/be_spectrum_z2.dat")  # type: ignore  # pylint: disable=E0602
