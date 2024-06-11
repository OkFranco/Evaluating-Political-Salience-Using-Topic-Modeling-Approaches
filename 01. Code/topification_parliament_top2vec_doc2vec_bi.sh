#!/bin/bash

#SBATCH --job-name=topification_parliament_top2vec_doc2vec_bi
#SBATCH --time=90:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # Assuming Python is not parallelized
#SBATCH --cpus-per-task=16   # Assuming 16 CPUs are available on each node
#SBATCH --partition=hpc

# Used to guarantee that the environment does not have any other loaded module
module purge

# Load required software modules
module load gcc63/openmpi/4.0.3
module load python/3.10.8  # Adjust the version based on your system


source venv_twitter/bin/activate

# Run Python script
echo "=== Running Python script ==="
echo "Topification_parliament_top2vec_doc2vec_bi.py"
python3.10 Topification_parliament_top2vec_doc2vec_bi.py

echo "Finished with job "

