import numpy as np
from typing import IO, Any, Callable, Dict, List, Optional, Union

from ase import Atoms
from ase.optimize.optimize import Optimizer
from ase.optimize.optimize import OptimizableAtoms

# class implementing the (continuous) local optimization method developed for the Fuzzy Global Optimization (FGO) algorithm. 
# This LO is designed for situations where the given cluster is already somewhat close to a stable configuration; i.e. after most cluster disturbances have been applied. 
# Though it does work for general clusters, its convergence rate is significantly lower than more established local optimizers. 
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
        self.atoms = atoms
        self.df = df
        self.dE = dE

        self.last_energy = None

    def converged(self, forces=None):
        if self.last_energy is None:
            return False
        
        # the local optimization is likely close to a minimum if the energy decrease from the last iteration is small
        return abs(self.last_energy - self.optimizable.get_potential_energy()) < self.dE
    
    def step(self):
        
        # retrieve the gradient, and if it is already minimal, the cluster is either already at the local minimum, or split into groups that are too far apart making them effectively also local equilibria
        optimizable = self.optimizable
        energy0 = optimizable.get_potential_energy()

        negative_gradient = optimizable.get_forces()
        grad_length = np.sqrt((negative_gradient ** 2).sum(axis=1).sum())

        if(grad_length < 0.000001):
            self.last_energy = energy0
            return
        
        # the gradient size is clipped within a reasonable range, to ensure that the sampling distance remains reasonable
        negative_gradient = negative_gradient * ((min(100, grad_length)) / grad_length)

        # evaluate the energies of clusters moved along the gradient
        positions = optimizable.get_positions()

        optimizable.set_positions(positions + self.df * negative_gradient)
        energy1 = optimizable.get_potential_energy()

        optimizable.set_positions(positions + 2 * self.df * negative_gradient)
        energy2 = optimizable.get_potential_energy()
        
        # based on the energy at these 3 points along the same 3*N dimensional line, fit a y=ax^2+bx+c polynomial, and estimate the minimum on this fitted curve
        val1 = (2 * energy2 - 4 * energy1 + 2 * energy0)
        if(abs(val1) < 0.00000001):
            optimizable.set_positions(positions)
            self.last_energy = energy0
            return

        optimal_move_units = (energy2 - 4 * energy1 + 3 * energy0) / val1

        # clamp the estimated minimum of the polynomial appropriately
        # if the minimum is in the opposite direction of the negative gradient, the energy surface is better approximated as a negative parabola, meaning (for LJ) the atoms are too far apart
        # instead, nudge the atoms a steady small amount in the direction of the negative gradient
        if (optimal_move_units < 0):
            optimal_move_units = 20
        # the basic polynomial used for fitting is only accurate for a limited range, thus clamp the distance to a reasonable range
        optimal_move_units = min(optimal_move_units, 50)

        # update cluster atom locations
        optimizable.set_positions(positions + (self.df * optimal_move_units) * negative_gradient)
        self.last_energy = energy0