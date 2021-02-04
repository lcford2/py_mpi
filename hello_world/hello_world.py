from mpi4py import MPI

# setup the communicator
comm = MPI.COMM_WORLD

# get the rank of this process and the number of processors.
rank = comm.Get_rank()
nprocs = comm.Get_size()

print(f"Hello, World from processor {rank} of {nprocs}")


