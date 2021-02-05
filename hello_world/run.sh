#!/bin/bash

# mpirun or mpiexec is the call to run mpi programs
# -n : number of processes to spawn
# In this case, python is the executable to be called and hello_world.py is the first argument to python
# This is similar to calling a C or Fortran MPI program executable. 

mpirun -n 4 python hello_world.py
