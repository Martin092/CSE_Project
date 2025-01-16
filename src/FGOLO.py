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
        dE: float = 0.001,
        **kwargs,
    ):
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, **kwargs)
        self.offsetAtoms1 = OptimizableAtoms(atoms)
        self.offsetAtoms2 = OptimizableAtoms(atoms)

        self.atoms = atoms
        self.df = df
        self.dE = dE

        self.lastEnergy = None

    #def log(self, forces=None):
    #    return

    def converged(self, forces=None):
        if self.lastEnergy is None:
            return False
        
        #print("dE ", abs(self.lastEnergy - self.optimizable.get_potential_energy()))
        return abs(self.lastEnergy - self.optimizable.get_potential_energy()) < self.dE
    
    def step(self):
        
        # retrieve the gradient, and if it is already minimal, the cluster is either already at the local minimum, or split into groups that are too far apart making them effectively also local equilibria
        optimizable = self.optimizable
        energy0 = optimizable.get_potential_energy()

        negativeGradient = optimizable.get_forces()
        gradLength = np.sqrt((negativeGradient ** 2).sum(axis=1).sum())

        if(gradLength < 0.000001):
            self.lastEnergy = energy0
            return
        
        # the gradient size is clipped within a reasonable range, to ensure that the sampling distance remains reasonable
        negativeGradient = negativeGradient * ((min(100, gradLength)) / gradLength)

        # evaluate the energies of clusters moved along the gradient
        positions = optimizable.get_positions()

        optimizable.set_positions(positions + self.df * negativeGradient)
        energy1 = optimizable.get_potential_energy()

        optimizable.set_positions(positions + 2 * self.df * negativeGradient)
        energy2 = optimizable.get_potential_energy()
        
        # based on the energy at these 3 points along the same 3*N dimensional line, fit a y=ax^2+bx+c polynomial, and estimate the minimum on this fitted curve
        val1 = (2 * energy2 - 4 * energy1 + 2 * energy0)
        if(abs(val1) < 0.00000001):
            optimizable.set_positions(positions)
            self.lastEnergy = energy0
            return

        optimalMoveDistance = (energy2 - 4 * energy1 + 3 * energy0) / val1
        #print("move Dist: ", optimalMoveDistance)

        if (optimalMoveDistance < 0):
            optimalMoveDistance = 20
        optimalMoveDistance = min(optimalMoveDistance, 50)

        optimizable.set_positions(positions + (self.df * optimalMoveDistance) * negativeGradient)
        self.lastEnergy = energy0