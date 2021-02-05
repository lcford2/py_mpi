from mpi4py import MPI
import numpy as np

# setup communicator and get # procs
comm = MPI.COMM_WORLD
nprocs = comm.Get_size()

# We are solving the Inhomogeneous 1D groundwater flow problem
# 0 = d/dx[k(x)dh/dx]
# from x = 0 to x = L, i.e. we are solving along a 1 dim grid
# To simplify the parallelization, we can use the Cartesian Conv. Function
# from MPI. For this example, we could figure this out ourselves pretty easily
# but when scaling up, this is really convenient
# get dimensions
dims = MPI.Compute_dims(nprocs, 1)
# setup grid
grid = comm.Create_cart(dims)
# now we want the gridded rank
rank = grid.Get_rank()

# the jacobi algorithm is needs the previous and next values
# grid.Shift(direction (dim), displacement)
# So shift along the zeroth axis, 1 unit in both directions
# Shift return -1 for values less than 0 or greater than nprocs
left, right = grid.Shift(0,1)

# grab the status to be used in sendrecv calls later
status = MPI.Status()

# setup problem parameters
h0 = 1.0 # GW head at x = 0
hL = 0.0 # GW head at x = L
x0 = 0 # initial position
xL = 10 # final position

# parameters for hydraulic conductivity of medium [k(x)]
# k(x) = ax^2 + bx + c
a = 0.007
b = -0.07
c = 0.2

# number of iterations
nit = 10000

# Make sure the number of trials is evenly divisible
# by the number of processors used
N = 100
n = N // nprocs
N = n * nprocs

# determine delta x based on N and total distance
delx = (xL - x0) / (N + (2 * nprocs) - 1)

# mpi4py runs much faster if we can use primitive data types
# when sharing information. Because python objects are complex, 
# it will serialize them using the pickle library then send them to
# the correct processor. This is obviously very slow. Numpy arrays
# store data as primitive types under the hood and can be treated as 
# buffers, thus speeding up communications. However, looping over numpy
# arrays is much slower than looping over python lists. Therefore,
# we will be using lists within each process but passing as np arrays.
# You can replace the lists with numpy arrays and you will notice a dropoff in performance
# the + 2 is because we know the two end points, we want to evaluate 
# everything in between 
h = list(np.zeros(n+2))
hnew = list(np.zeros(n+2))

# precalculate k for this processor so it is just a look up value
k = list(np.zeros(n+2))
for i in range(n+2):
    x = (rank * n + i) * delx
    k[i] = (a*x*x) + (b*x) + c

# message passing parameters
count = 1
tag1 = 1
tag2 = 2

# so MPI knows not to try to send anything to those
if left == -1: left = MPI.PROC_NULL
if right == -1: right == MPI.PROC_NULL

# setup the boundary conditions
if rank == 0: h[0] = h0
if rank == nprocs-1: h[n+1] = hL

# using Sendrecv calls and assigning data types to what I am sending. 
# in mpi4py capital communication calls (Sendrecv vs sendrcv) expect buffer objects
# in Sendrecv, 
time1 = MPI.Wtime()
for j in range(nit):
    # simultaneously send the left point so it can be the upper boundary condition
    # for the processor to the left, and recieve our upper boundary condition 
    # from the processor to the right
    h1 = np.array(h[1])
    hn1 = np.array(h[n+1])
    h_0 = np.array(h0)
    hn = np.array(h[n])
    grid.Sendrecv([h1, MPI.DOUBLE], left, tag1,
                  [hn1, MPI.DOUBLE], right, tag1, status)
    # simultaneously send the right point so it can be the lower boundary condition 
    # for the processor to the right, and recieve our lower boundary condition from 
    # the processor to the left
    grid.Sendrecv([hn, MPI.DOUBLE], right, tag2,
                  [h_0, MPI.DOUBLE], left, tag2, status)

    # the above communication pattern would take at least 4 command if using traditional Send and Recv
    h[1] = h1
    h[n+1] = hn1
    h[0] = h_0
    h[n] = hn

    # Computational portion
    for i in range(1, n+1):
        hnew[i] = ((k[i+1] + k[i]) * h[i+1] + (k[i] + k[i-1]) * h[i-1]) \
                    / (k[i+1] + 2 * k[i] + k[i-1])

    # store the values
    for i in range(1, n+1):
        h[i] = hnew[i]

    # ensure boundary conditions still hold true
    if rank == 0: h[0] = h0
    if rank == nprocs - 1: h[n+1] = hL

time2 = MPI.Wtime()

# calculate floating point operations per second.
# in our loop we have 9 ops per iteration and overall we have nit * N iterations
total_time = time2 - time1
mflops = 9 * nit * N * 1e-6 / total_time

# constants for analytical solution
c0 = -0.0055
c1 = 0.5

# get error
errsq = np.zeros(1)

# calculate the error for this processor
an_coef = 2 / np.math.sqrt((4 * a * c) - (b * b))
for i in range(1, n + 1):
    h_act = c0 * (an_coef * np.math.atan((2 * a * x + b) * an_coef / 2)) + c1
    errsq[0] += (h_act - h[i])**2

sumsqerr = None
h_all = None
if rank == 0:
    sumsqerr = np.array([0.0])
    h_all = np.zeros((n + 2) * nprocs)

harr = np.array(h)

# these communication calls need to happen on all processes
# process that are not `root` will send information and `root`
# will recieve it. 
# get sum sq error
comm.Reduce([errsq, MPI.DOUBLE], [sumsqerr, MPI.DOUBLE], op=MPI.SUM, root=0)
# get final solution for h
comm.Gather([harr, MPI.DOUBLE], [h_all, MPI.DOUBLE], root=0)

if rank == 0:
    error = np.sqrt(sumsqerr[0]) / N
    print(f"Total time        = {total_time:0.8f}")
    print(f"Mflops            = {mflops:.1f}")
    print(f"Error             = {error:.16f}")
    print(f"N                 = {N}")
    with open("head.out", "w") as f:
        for i, H in enumerate(h_all):
            x = i * delx
            f.write("{},{}\n".format(x, H))

MPI.Finalize()
