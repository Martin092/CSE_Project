echo "This has to be run from the root of the project. It will fail otherwise."

module load 2024r1
module load mpi/2021.11
module load python/3.10.12
module load openmpi/4.1.6
module load py-pip/23.1.2

pip install -r requirements.txt