
for nprocs in 1 2 3 4
do
    printf "\n$ mpirun -n %d python solve.py\n" $nprocs
    mpiexec -n $nprocs python solve.py
done
