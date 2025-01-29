from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import constants
from single_pendulum import hamiltonian

M = constants.M  # Mass of the pendulum
L = constants.L  # Length of the pendulum

def plot_positions_in_cartesian(t: np.array, y: np.array, title: str):
    """
    Plot the position of the single pendulum in Cartesian coordinates over time.

    Parameters:
        t: Array of time values.
        y: Array of state values, where each row is [q, p] at a given time step.
        title: Title of the plot.
    """

    # Extract the angle values (q) from the state vector y
    q = y[:, 0]  # Angle of the pendulum

    # Calculate the Cartesian coordinates for the pendulum
    x = L * np.sin(q)
    y_coord = -L * np.cos(q)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_coord, label="Pendulum (x, y)", color='blue', zorder=1)

    plt.scatter(x[0], y_coord[0], marker='o', color='red', s=100, label="Starting position", zorder=2)
    plt.scatter(x[-1], y_coord[-1], marker='x', color='red', s=100, label="End position", zorder=2)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Position of the Single Pendulum Cartesian Coordinates from {title}')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def plot_hamiltonian_deviation_over_time(t: np.array, y: torch.tensor, title: str):
    """
    Plot the relative deviation in the value of the Hamiltonian function compared to t=0 over time.

    Parameters:
        t (np.array): Array of time values.
        y (tensor.torch): Tensor of state values, where each row is [p, q] at a given time step.
        title: Title of the plot.
    """

    h = hamiltonian(y)  # Calculate the value of the Hamiltonian based on the system states y
    h0 = h[0]  # Capture initial state for relative deviation

    deviation = np.abs(h0 - h) / np.abs(h0)  # Compute the relative deviation

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


def compare_hamiltonian_single_pendulum(model: nn.Module) -> None:
    """
    Compare true and learned Hamiltonian functions for a single pendulum system.

    Args:
        model (nn.Module): Learned Hamiltonian.
    """
    p_values = np.linspace(-np.pi, np.pi, 100)
    q_values = np.linspace(-np.pi, np.pi, 100)
    P, Q = np.meshgrid(p_values, q_values)

    P_flat = torch.tensor(P.flatten(), dtype=torch.float32)
    Q_flat = torch.tensor(Q.flatten(), dtype=torch.float32)

    system_states = torch.stack([Q_flat, P_flat], dim=1)
    H_values = hamiltonian(system_states).reshape(100, 100).numpy()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Create interactive plot
    surf = ax.plot_surface(Q, P, H_values, cmap='viridis')

    ax.set_xlabel('q')
    ax.set_ylabel('p')
    ax.set_zlabel('H')
    ax.set_title('Hamiltonian Surface Plot')

    plt.show()
