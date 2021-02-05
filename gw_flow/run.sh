#!/bin/bash

for nprocs in 1 2 3 4
do
    printf "\n$ mpirun -n %d python jacobi.py\n" $nprocs
    mpirun -n $nprocs python jacobi.py
done
