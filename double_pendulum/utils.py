from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
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

def compare_hamiltonian_double_pendulum(model: nn.Module) -> None:
    """
    Compare true and learned Hamiltonian functions for a double pendulum system.

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
    q1_test = torch.linspace(-np.pi, np.pi, test_size)
    q2_test = torch.linspace(-np.pi, np.pi, test_size)
    p1_test = torch.linspace(-2, 2, test_size)
    p2_test = torch.linspace(-2, 2, test_size)

    # Generate a grid for (q1, p1) and flatten for Hamiltonian evaluation
    Q1, P1 = torch.meshgrid(q1_test, p1_test, indexing='ij')
    Q2, P2 = torch.meshgrid(q2_test, p2_test, indexing='ij')

    # Flatten and stack into system states for the double pendulum
    Q1_flat, P1_flat = Q1.flatten(), P1.flatten()
    Q2_flat, P2_flat = Q2.flatten(), P2.flatten()

    system_state_test = torch.stack([Q1_flat, Q2_flat, P1_flat, P2_flat], dim=1)

    # Compute true and learned Hamiltonians and reshape for plotting
    H_true = hamiltonian(system_state_test).detach().numpy().reshape(test_size, test_size)
    H_learned = model(system_state_test).detach().numpy().reshape(test_size, test_size)

    # Determine global min and max for consistent scaling
    vmin, vmax = min(H_true.min(), H_learned.min()) * 1.1, max(H_true.max(), H_learned.max()) * 1.1

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    plot_hamiltonian(axes[0, 0], Q1.numpy(), P1.numpy(), H_true, 'True Hamiltonian (q1, p1)', 'q1', 'p1', vmin, vmax)
    plot_hamiltonian(axes[0, 1], Q1.numpy(), P1.numpy(), H_learned, 'Learned Hamiltonian (q1, p1)', 'q1', 'p1', vmin, vmax)
    plot_hamiltonian(axes[1, 0], Q2.numpy(), P2.numpy(), H_true, 'True Hamiltonian (q2, p2)', 'q2', 'p2', vmin, vmax)
    plot_hamiltonian(axes[1, 1], Q2.numpy(), P2.numpy(), H_learned, 'Learned Hamiltonian (q2, p2)', 'q2', 'p2', vmin, vmax)

    plt.tight_layout()
    plt.show()


