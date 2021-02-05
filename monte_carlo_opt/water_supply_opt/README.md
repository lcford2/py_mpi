## Monte Carlo Optimization

In this example, we perform a random search for the maximum of a function. 
The `funcs.py` file contains the function (shown below) we are trying to maximize.

<img src="./eqns/func.svg" title="f(x)=-4x^2 + 3x - 4">

Though this could have been included in `solve.py`, it is important to demonstrate an external function being called with `mpi4py`. The idea is that users can setup one file with all of the parallelization and other files that actually contain the nuts and bolts of what they want to accomplish. 

If you are running linux, you can simply run the `run.sh` file to see this example in action (assuming your environment is correct, see main repository [README.md](https://github.com/lcford2/py_mpi/blob/main/README.md)). Otherwise, you can open `run.sh` to see how I am running the code and run it on your own. 

Flipping some of the maximization logic in `solve.py`, or simply negating the values returned from the function, can change this to a minimization problem. 

Adding statements in the for loop to check that the random value selected and the corresponding value from the objective function are within constraints can change this from an unconstrained optimization to a constrained optimization. 

Adding a gradient calculation and some basic descent logic can make the algorithm slightly smarter. 
