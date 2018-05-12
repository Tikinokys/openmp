#!/bin/bash

touch result2.csv

num_for_repeat=5

echo "NumThreads,Dim,Runtime" > result2.csv
for omp_threads in 1 2 3 4
do
	export OMP_NUM_THREADS=$omp_threads

	for dim in 1000 1500 2000 2500
	do
		for ((i=0; i < $num_for_repeat; i++))
		do
		./main $dim >> result2.csv
		done
	done
done