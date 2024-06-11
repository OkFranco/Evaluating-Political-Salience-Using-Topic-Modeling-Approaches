#!/bin/bash

#SBATCH --job-name=parallelized_urls_final_data_treatement
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
##echo "Urls_news_filtering_I.py"
##python3.10  Urls_news_filtering_I.py

##echo "Urls_news_infer_date.py"
##python3.10  Urls_news_infer_date.py

##echo "Urls_news_cleaning.py"
##python3.10  Urls_news_cleaning.py
total_urls=9
batch_size=1
num_batches=$((total_urls / batch_size))
# Loop to run multiple instances of the Python script in parallel
for ((i=0; i<num_batches; i++))
do
	start=$((i * batch_size))
	end=$((start + batch_size))
	
	echo "Processing batch $((i+1)) from $start to $end"
	python3.10  Urls_news_filtering_II_paralize.py $start $end &  # Run in background
done
# Wait for all background processes to finish
wait




##echo "filtering II"
##python3.10   # Run in background


echo "Concluded"

