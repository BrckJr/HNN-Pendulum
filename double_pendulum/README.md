# HNN-Double-Pendulum
The goal of this repository is to learn the Hamiltonian system of a double pendulum from data. 
To do so, I implemented the following:
1. The <code>double_pendulum.py</code> file which includes all the physics from the double pendulum
2. Three numerical solvers (<code>explicit_euler.py</code> and <code>symplectic_euler.py</code>) to solve the known PDE directly and
to analyze the characteristics of these solvers (energy conservation, stability over time, ...)
3. A standard feed-forward neural network (<code>FFNN.py</code> and <code>FFNN_utils.py</code>) which directly learns the gradients from data samples. The learned gradients
are then used with the numerical solvers from the last step to solve the "learned PDE".
4. A Hamiltonian neural network (<code>HNN.py</code> and <code>HNN_utils.py</code>) which learns how to predict the Hamiltonian function to ensure the energy conservation
of the system compared to the FFNN. The learned Hamiltonian function is then used together with the numerical solvers
from the second step to solve the system.

The constants of the double pendulum system are stored in <code>constants.py</code> and the simulation can be started from <code>main.py</code>. 
Additional plotting functions for analysis purpose are contained in <code>utils.py</code>.