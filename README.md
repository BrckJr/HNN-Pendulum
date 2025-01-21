# Hamiltonian Neural Networks for Single and Double Pendulum

This repository contains code for the simulation of a single and double pendulum with numerical solvers and learning of the Hamiltonian system with a Hamiltonian Neural Network. 

The goal is to learn neural networks which can be used to predict the trajectory of a single and double pendulum. 
Like in this case where the prediction of the numerical integration of the true, underlying Hamiltonian corresponds 
nicely to the learned Hamiltonian:

![Alt text](plots/traj_sympl_euler.png)

![Alt text](plots/traj_pred_HNN.png)

![Alt text](plots/learned_vs_true_hamiltonian.png)
