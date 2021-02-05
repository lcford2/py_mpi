from mpi4py import MPI
from funcs import func1
import random

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
nprocs = comm.Get_size()

# sets the range of random numbers
# will pull a random floating point number between
# -BIG_BOUND and BIG_BOUND
BIG_BOUND = 100000

# Number of function evaluations
N = int(10e6)

# Each process will evaluate the function approximately N/nprocs times
# The division is floored so the result is an integer
evals = N//nprocs

# set max_value to something very negative
max_value = -float("inf")
# set variable to store the x value that produces the maximum function value
max_x = 0

# we want each process to generate different random numbers, 
# so we can seed the random number generator with the process number
# added to the current time, so we get different answers every time we run. 
random.seed(MPI.Wtime() + rank)

# We also want to time the function, but it can get wacky if you try to use
# the time module. We use MPI.Wtime() instead because it is safe for this app.
time1 = MPI.Wtime()

# perform the calculation
for i in range(evals):
    x = random.uniform(-BIG_BOUND, BIG_BOUND)
    value = func1(x)
    if value > max_value:
        max_x = x
        max_value = value

# To get both the processor rank and the max value
# we need to create a tuple with those values.
pair = (max_value, rank)

# We want the `pair` that has the largest value at index 0.
# This command says go to all of the processors, look at the variable 
# pair, perform MPI.MAXLOC (which finds the `pair` with the global max value), 
# and, if `rank` is equal to `root`, return that pair. 
# If `rank` != `root`, `global_max` will be None.
global_max = comm.reduce(pair, op=MPI.MAXLOC, root=0)

# func1 theoretical max's
func1_max = -3.4375
func1_x = 0.375

# need to do different things if we are on the root process
if rank == 0:
    # tuple unpacking
    global_max, max_loc = global_max
    # if the maximum value was not found on the root process
    if max_loc != 0:
        # non blocking recieve from max_lox
        req = comm.irecv(source=max_loc, tag=11)
        # force the recieve to finish
        global_x = req.wait()
    else:
        # if the max value was found on the root process, 
        # and we tried to recieve, we would enter into a 
        # infinite loop
        global_x = max_x
    # end the time, in general you should include the communication
    # in your timing because it is part of the overhead and it is not
    # a fair comparison between a single process and multiple processes
    time2 = MPI.Wtime()
    # print out the results
    print(f"Theoretical maximum : max f(x) = f({func1_x:.3f}) = {func1_max:.4f}")
    print(f"Calculate maximum   : max f(x) = f({global_x:.3f}) = {global_max:.4f}")
    print(f"Time taken : {time2-time1:.3f} seconds")
    print(f"Processes  : {nprocs}") 
else:
    # this is the reason non-blocking communication is used
    # If blocking communication is used (comm.send vs comm.isend)
    # then these processes will be waiting on the `dest` to recieve the information
    # With non-blocking, these can theoretically move on to do more work. 
    # Tag is just a way of making sure that you are recieving the same data that is being sent. 
    comm.isend(max_x, dest=0, tag=11)



