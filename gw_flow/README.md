## Parallel Finite Differences

In this example, we calculate a numerical solution to the one-dimensional inhomogenous groundwater flow equation (1) using the Jacobi Method (2). In the below equations, *h* refers to the water head in the aquifer, *k* is the hydraulic conductivity with the aquifer which varys with *x*, the distance along a single axis in the aquifer. This is a simplified representation because in reality, we need to consider all three dimensions.

<img src="https://latex.codecogs.com/svg.latex?\frac{d}{dx}\left(k(x)\frac{dh}{dx}\right) = 0 \text{ (1)}" title="GW Flow Equation"/>

<br>
<img src="https://latex.codecogs.com/svg.latex?h_i^{(j+1)} = \frac{(k_{i+1} + k_i)h_{i+1}^{(j)}+(k_i + k_{i-1})h_{i_1}^{(j)}}{k_{i+1}+2k_i+k_{i-1}} \text{ (2)}" title="Jacobi Method for GW Flow Equation"/>

The problem is decomposed by giving each processes a chunk of the distance the problem is solved for (i.e. if the total distance os 10 meters, and you use 5 processes, each process will solve ~2 meters of the problem). Each process sends the values of at the end of its solution distance to its neighboring processes to act as boundary conditions for their problem. 

If you are running linux, you can simply run the `run.sh` file to see this example in action (assuming your environment is correct, see main repository [README.md](https://github.com/lcford2/py_mpi/blob/main/README.md)). Otherwise, you can open `run.sh` to see how I am running the code and run it on your own. 

This example is pulled from a homework assignment for CE 791 High Performance Computing at NC State University. 
