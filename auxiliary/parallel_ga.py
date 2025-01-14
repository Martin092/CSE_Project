"""GA parallel execution module"""

import sys
import os

from ase.io import write
from mpi4py import MPI  # pylint: disable=E0611
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator


sys.path.append("./")

from src.genetic_algorithm import GeneticAlgorithm  # pylint: disable=C0413
from auxiliary.cambridge_database import get_cluster_energy  # pylint: disable=C0413
from src.basin_hopping_optimizer import BasinHoppingOptimizer

def parallel_ga(
    ga: GeneticAlgorithm,
    atoms: str,
    max_iterations: int,
    conv_iterations: int = 0,
    seed: int | None = None,
    max_local_steps: int = 20000,
    bh_optimizer: BasinHoppingOptimizer | None = None,
) -> None:
    """
    Execute GA in parallel using mpiexec.
    :param ga: Genetic Algorithm instance
    :param atoms: Number of atoms and atomic type in cluster.
    :param max_iterations: Number of maximum iterations to perform.
    :param conv_iterations: Number of iterations to be considered in the convergence criteria.
    :param seed: Seeding for reproducibility.
    :param max_local_steps: Maximum number of steps for the local optimizer.
    :param bh_optimizer: A basin hopping optimizer to be used instead of a local optimizer
    """

    comm = MPI.COMM_WORLD  # Generate global communicator
    rank = comm.Get_rank()  # Get the processes rank

    ga.comm = comm  # set up the GA's global communicator
    ga.setup(atoms)  # Ensure GA is properly set up

    num_atoms = ga.utility.num_atoms

    if rank == 0:  # If master process
        ga.run(atoms, max_iterations, conv_iterations, seed)  # Execute GA

        for i in range(
            1, comm.Get_size()
        ):  # After execution, send kill message to worker processes
            comm.Send(np.zeros((num_atoms, 3), dtype=float), tag=0, dest=i)  # tag=0

        comm.free()  # disband the global communicator

        if not os.path.exists("./data/optimizer"):  # Check if directory exists
            if not os.path.exists("./data"):  # Check if parent directory exists
                os.mkdir("./data")  # Create parent directory
            os.mkdir("./data/optimizer")  # Create results directory

        best_cluster = ga.best_config  # Get best cluster and center it
        best_cluster.center()  # type: ignore
        write(f"./data/optimizer/LJ{num_atoms}.xyz", best_cluster)  # Write to file

        best = get_cluster_energy(
            num_atoms, "./"
        )  # Get the best potential from database

        ga.configs = ga.configs[
            1:
        ]  # Remove best initial configuration since all are random

        ga.utility.write_trajectory(
            f"./data/optimizer/LJ{num_atoms}.traj"
        )  # Write trajectory

        plt.plot(
            ga.potentials[1:]
        )  # Remove best initial potential since all are random
        plt.gca().xaxis.set_major_locator(
            MaxNLocator(integer=True)
        )  # Make x-axis ticks integers
        plt.gca().yaxis.set_major_formatter(
            ticker.ScalarFormatter(useOffset=False)
        )  # Make y-axis ticks scalar
        ticks = list(plt.gca().get_xticks())[1:-1]  # Get x-axis ticks
        if ticks[-1] != len(ga.potentials) - 2:
            ticks.append(len(ga.potentials) - 2)  # Explicitly add last index
        plt.gca().set_xticks(ticks)  # Adjust x-axis ticks
        plt.title(f"Execution on LJ{num_atoms}")  # Give title to the plot
        plt.xlabel("Iteration")  # Label the x-axis
        plt.ylabel("Potential Energy")  # Label the y-axis
        plt.tight_layout()  # Fix the plot's location on the figure
        plt.savefig(f"./data/optimizer/LJ{num_atoms}.png")  # Save plot to file
        plt.close()  # Close plot object

        print(  # Print execution summary
            f"LJ {num_atoms}: {ga.current_iteration-1} iterations for "
            f"{int(np.floor_divide(ga.execution_time, 60))} min {int(ga.execution_time) % 60} sec",
            flush=True,
        )

        if (
            abs(ga.best_potential - best) < 0.000001 * num_atoms * num_atoms
        ):  # Check if within uncertainty boundaries
            print("Found global minimum from database.")
        elif ga.best_potential < best:  # Else if sufficiently better
            print(
                f"Found new global minimum. Found {ga.best_potential}, but database minimum is {best}."
            )
        else:  # Finally, it is suboptimal
            print(
                f"Didn't find global minimum. Found {ga.best_potential}, but global minimum is {best}."
            )

        MPI.Finalize()  # End parallel execution

    else:  # If worker process
        ranks = comm.Get_size() - 1  # Compute number of worker processes
        size = int(
            np.ceil(ga.num_clusters / ranks)
        )  # Compute number of clusters per worker process
        while True:  # Execute until master process kills worker process
            pos = np.empty(
                (size, num_atoms, 3), dtype=np.float64
            )  # Clusters to receive
            if ga.debug:
                print(f"Rank {ga.comm.Get_rank()} receiving.", flush=True)
            status = MPI.Status()  # Status object
            ga.comm.Recv(
                [pos, MPI.DOUBLE], source=0, tag=MPI.ANY_TAG, status=status
            )  # Receive clusters
            if status.tag == 0:
                break  # If status tag is 0, kill worker process
            clusters = []  # Set up clusters list for each worker process
            for (
                i
            ) in (
                pos
            ):  # For each received cluster, generate Atoms object from atomic configuration
                clus = ga.utility.generate_cluster(i)  # type: ignore
                clusters.append(clus)  # Add Atoms object to list of clusters
            for clus in clusters:  # For each received cluster
                ga.mutation(clus, max_local_steps)  # Perform mutation
                clus = ga.utility.generate_cluster(pos)
                if bh_optimizer:
                    bh_optimizer.run(
                        atoms, 40, 20, initial_configuration=clus.positions
                    )
                    clus = bh_optimizer.best_config
                else:
                    ga.local_optimizer(clus).run(steps=20000)  # Perform local optimization
            if ga.debug:
                print(f"Rank {ga.comm.Get_rank()} sending.", flush=True)
            send = []  # Generate list of clusters to send
            for clus in clusters:  # For each mutated and optimized cluster
                send.append(clus.positions)  # Add atomic configuration to send list
            ga.comm.Send(
                [np.array(send), MPI.DOUBLE], dest=0, tag=2
            )  # Send processed clusters
