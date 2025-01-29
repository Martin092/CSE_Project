# Parallel Library for Global Geometry Optimization

## What is this?
This is a library with implementations of Basin Hopping, Minima Hopping, and a Genetic Algorithm, which are Global Geometry Optimization methods used to find the global minimum of the PES for an atomic cluster. There is also an implementation of a local optimization method developed for Fuzzy Global Optimization.

There are parallel distributed memory implementations (MPI) of Basin Hopping and Genetic Algorithms.

Basin Hopping can be run with operator sequencing, where you can specify which operators to use as disturbances. This can be done both statically or dynamically, see the docstrings for more information.

The Global Geometry Optimization methods are all subclasses of our `GlobalOptimizer` class. Which each use the `Utility` class for methods such as disturbances.

There also exists a GUI to run these algorithms on arbitrary atomic clusters. And a benchmark file to log statistics on runs of the algorithms.

This implementation is based on the [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/).

## How can I use this?

### Installation
- You can install all the necessary dependencies by running
`pip install -r requirements.txt`
in the root of the project.


- If you want to use the parallel implementations of algorithms, you may need to set up MPI for your machine. The documentation for [mpi4py](https://pypi.org/project/mpi4py/) explains how to do this for different operating systems under the "Install" section.


- If you would like to use the `gpaw` calculators, this will need to be installed separately from [here](https://gpaw.readthedocs.io/install.html) (We **strongly** urge the developers of `gpaw` to update their documentation). If this is unable to be installed, you can comment out any imports of `gpw.py` and the code should work the same, excluding any features reliant on `gpw.py`.

### Usage
- You can run the GUI by running `python3 aux/gui.py` from the root of the project.
- You can run implementations of the Global Geometry Optimization algorithms by creating an instance of a subclass of `GlobalOptimizer` passing that  into an instance of `Benchmark`, then using `benchmark_run()`. See `ga_playground.py` for an example of this.

### DelftBlue
- If you have access to [DelftBlue](https://doc.dhpc.tudelft.nl/delftblue/) then you can also run these implementations there. (You could of course do this with any supercomputer, but we provide scripts specifically for DelftBlue). All scripts in the `scripts` folder (except for `run-checks.sh`) can be used to setup delftblue, transfer files to and from it, and run jobs.

## How can I extend this implementation?
- If you want to add your own Global Optimizer, you can make a subclass of `GlobalOptimizer` and implement the necessary methods.
- If you want to add your own disturbance operators, these can be added in the `Utility` class. Please ensure that any disturbances implemented result in a valid atomic cluster.
- We have a pipeline that does linting, type checking, and style checking. These can be run locally by running `scripts/run_checks.sh`. Please ensure any extensions to the implementation pass the pipeline. 

## Remarks
- Running certain parts of the implementation may require an internet connection, as we make request to the [Cambridge Cluster Database](http://doye.chem.ox.ac.uk/jon/structures/LJ.html) to confirm whether the global minimum has been found.