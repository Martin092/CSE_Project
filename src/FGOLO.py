import numpy as np
from typing import IO, Any, Callable, Dict, List, Optional, Union

from ase import Atoms
from ase.optimize.optimize import Optimizer
from ase.optimize.optimize import OptimizableAtoms

class FGOLO(Optimizer):

    def __init__(
        self,
        atoms: Atoms,
        restart: Optional[str] = None,
        logfile: Union[IO, str] = '-',
        trajectory: Optional[str] = None,
        df: float = 0.001,
        dE: float = 0.01,
        **kwargs,
    ):
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, **kwargs)
        self.offsetAtoms1 = OptimizableAtoms(atoms)
        self.offsetAtoms2 = OptimizableAtoms(atoms)

        self.atoms = atoms
        self.df = df
        self.dE = dE

        self.lastEnergy = None

    def converged(self, forces=None):
        if self.lastEnergy is None:
            return False
        
        print("elloe!")
        return abs(self.lastEnergy - self.optimizable.get_potential_energy()) < self.dE

    def step(self):
        optimizable = self.optimizable


        negativeGradient = optimizable.get_forces()
        # test clusters modified by moving along the (negative) gradient
        positions = optimizable.get_positions()

        energy0 = optimizable.get_potential_energy()

        optimizable.set_positions(positions + self.df * negativeGradient)
        energy1 = optimizable.get_potential_energy()

        optimizable.set_positions(positions + 2 * self.df * negativeGradient)
        energy2 = optimizable.get_potential_energy()
        
        # based on the energy at these 3 points along the same 3*N dimensional line, fit a y=ax^2+bx+c polynomial, and estimate the minimum on this fitted curve
        print("energy0: ", energy0)
        print("energy1: ", energy1)
        print("energy2: ", energy2)

        optimalMoveDistance = -(energy2 - 4 * energy1 + 3 * energy0) / (2 * energy2 - 4 * energy1 + 2 * energy0)
        print("optimalMoveDistance: ", optimalMoveDistance)

        optimizable.set_positions(positions + (self.df * optimalMoveDistance) * negativeGradient)

        print("best Energy found: ", optimizable.get_potential_energy())

        #self.dump((self.v, self.dt))
