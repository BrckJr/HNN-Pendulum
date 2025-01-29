import matplotlib.pyplot as plt
import numpy as np
import torch
import constants
from double_pendulum import hamiltonian

M1, M2 = constants.M1, constants.M2  # Masses of the pendulums
L1, L2 = constants.L1, constants.L2  # Lengths of the pendulums

def plot_positions_in_cartesian(t: np.array, y: np.array, title: str):
    """
    Plot the positions of two pendulums in Cartesian coordinates over time.

    Parameters:
        t: Array of time values.
        y: Array of state values, where each row is [q1, q2, p1, p2] at a given time step.
        title: Title of the plot.
    """

    # Extract the angle values (q1, q2) from the state vector y
    q1 = y[:, 0]  # Angle of pendulum 1
    q2 = y[:, 1]  # Angle of pendulum 2

    # Calculate the Cartesian coordinates for both pendulums
    x1 = L1 * np.sin(q1)
    y1 = - L1 * np.cos(q1)

    x2 = x1 + L2 * np.sin(q2)
    y2 = y1 - L2 * np.cos(q2)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x1, y1, label="Pendulum 1 (x1, y1)", color='blue', zorder=1)
    plt.plot(x2, y2, label="Pendulum 2 (x2, y2)", color='lightgreen', zorder=1)

    plt.scatter(x1[0], y1[0], marker='o', color='red', s=100, label="Starting position pendulum 1", zorder=2)
    plt.scatter(x1[-1], y1[-1], marker='x', color='red', s=100, label="End position pendulum 1", zorder=2)
    plt.scatter(x2[0], y2[0], marker='o', color='darkorange', s=100, label="Starting position pendulum 2", zorder=2)
    plt.scatter(x2[-1], y2[-1], marker='x', color='darkorange', s=100, label="End position pendulum 2", zorder=2)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Position of the Double Pendulum Cartesian Coordinates from {title}')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def plot_hamiltonian_deviation_over_time(t: np.array, y: torch.tensor, title: str):
    """
    Plot the relative deviation in the value of the Hamiltonian function compared to t=0 over time.

    Parameters:
        t (np.array): Array of time values.
        y (tensor.torch): Tensor of state values, where each row is [p1, p2, q1, q2] at a given time step.
        title: Title of the plot.
    """

    h = hamiltonian(y) # Calculate the value of the Hamiltonian based on the system states y
    h0 = h[0] # Capture initial state for relative deviation

    deviation = np.abs(h0 - h) / np.abs(h0) # Compute the relative deviation

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(t, deviation)
    plt.xlabel('t')
    plt.ylabel('Rel. Deviation')
    plt.title(f'Relative Deviation of the Hamiltonian Function over Time for {title} solution')
    plt.show()

def plot_losses(loss_history: list, used_model: str):
    """ Plot the loss curve over all epochs

    Parameters:
        loss_history: List of loss values over all epochs.
        used_model: Name of the model which was used for training.
    """
    # Plot the loss history
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(loss_history)), loss_history, label="Training Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss History for {used_model}")
    plt.legend()
    plt.grid(True)
    plt.show()


