#!/bin/bash
#SBATCH --job-name=mulMatrix
#SBATCH --output=result.csv
#SBATCH --ntasks=1
#
#SBATCH --cpus-per-task=20
#
#SBATCH --time=20:00
#SBATCH --mem-per-cpu=100

num_for_repeat=5

echo "NumThreads,Dim,Runtime" 
for omp_threads in 1 2 4 6 8 10 12 14 16 18 20
do

export OMP_NUM_THREADS=$omp_threads

for dim in 1000 1500 2000 2500
do
	for ((i=0; i < $num_for_repeat; i++))
	do
	./main $dim 
	done
done

done