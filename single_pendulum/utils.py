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

    def plot_hamiltonian(ax, x, y, H, title, xlabel, ylabel, vmin, vmax):
        """Plot a single Hamiltonian as a contour."""
        contour = ax.contourf(x, y, H, levels=200, cmap="viridis", vmin=vmin, vmax=vmax)
        plt.colorbar(contour, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    test_size = 100
    q_test = torch.linspace(-np.pi, np.pi, test_size)
    p_test = torch.linspace(-2, 2, test_size)

    # Generate a grid for (q, p) and flatten for Hamiltonian evaluation
    Q, P = torch.meshgrid(q_test, p_test, indexing='ij')

    # Flatten and stack into system states for the single pendulum
    Q_flat, P_flat = Q.flatten(), P.flatten()

    system_state_test = torch.stack([Q_flat, P_flat], dim=1)

    # Compute true and learned Hamiltonians and reshape for plotting
    H_true = hamiltonian(system_state_test).detach().numpy().reshape(test_size, test_size)
    H_learned = model(system_state_test).detach().numpy().reshape(test_size, test_size)

    # Determine global min and max for consistent scaling
    vmin, vmax = min(H_true.min(), H_learned.min()) * 1.1, max(H_true.max(), H_learned.max()) * 1.1

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    plot_hamiltonian(axes[0], Q.numpy(), P.numpy(), H_true, 'True Hamiltonian (q, p)', 'q', 'p', vmin, vmax)
    plot_hamiltonian(axes[1], Q.numpy(), P.numpy(), H_learned, 'Learned Hamiltonian (q, p)', 'q', 'p', vmin, vmax)

    plt.tight_layout()
    plt.show()
