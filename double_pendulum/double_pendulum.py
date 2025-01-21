import torch
import numpy as np
import constants

M1, M2 = constants.M1, constants.M2  # Masses of the pendulums
L1, L2 = constants.L1, constants.L2  # Lengths of the pendulums
G = constants.G                      # Gravity constant

# Hamiltonian function rewritten for PyTorch
def hamiltonian(system_states: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hamiltonian (total energy) of the double pendulum for a tensor of states.

    Parameters:
        system_states: torch.Tensor of shape (N, 4)
            - Each row contains [q1, q2, p1, p2], representing the states of the system.

    Returns:
        H: torch.Tensor of shape (N,)
            - Hamiltonian (total energy) for each state.
    """
    # Extract states
    system_states = system_states.detach()
    if system_states.ndim == 1:
        q1, q2, p1, p2 = system_states[0], system_states[1], system_states[2], system_states[3]
    else:
        q1, q2, p1, p2 = system_states[:, 0], system_states[:, 1], system_states[:, 2], system_states[:, 3]

    # Compute kinetic energy
    numerator = (M2 * L2**2 * p1**2) + ((M1 + M2) * L1**2 * p2**2) - (2 * M2 * L1 * L2 * p1 * p2 * torch.cos(q1 - q2))
    denominator = 2 * M2 * L1**2 * L2**2 * (M1 + (M2 * torch.sin(q1 - q2)**2))
    H_kin = numerator / denominator

    # Compute potential energy
    H_pot = - (M1 + M2) * G * L1 * torch.cos(q1) - M2 * G * L2 * torch.cos(q2)

    # Hamiltonian as a summation of kinetic and potential energy
    H = H_kin + H_pot

    return H

# Vector field function rewritten for PyTorch
def vector_field(system_states: torch.Tensor) -> torch.Tensor:
    """
    Compute the time derivatives (vector field) for the double pendulum for a tensor of states.

    Parameters:
        system_states: torch.Tensor of shape (N, 4) or (4)
            - Each row contains [q1, q2, p1, p2], representing the states of the system.

    Returns:
        derivatives: torch.Tensor of shape (N, 4)
            - Each row contains [dq1/dt, dq2/dt, dp1/dt, dp2/dt].
    """

    # Extract states
    system_states = system_states.detach()
    if system_states.ndim == 1:
        q1, q2, p1, p2 = system_states[0], system_states[1], system_states[2], system_states[3]
    else:
        q1, q2, p1, p2 = system_states[:, 0], system_states[:, 1], system_states[:, 2], system_states[:, 3]

    # Precomputation of terms in dp1_dt and dp2_dt
    h1_numerator = (p1 * p2 * torch.sin(q1 - q2))
    h1_denominator = (L1 * L2 * (M1 + (M2 * torch.sin(q1 - q2)**2)))
    h1 = h1_numerator / h1_denominator

    h2_numerator = ((M2 * L2**2 * p1**2) + ((M1 + M2) * L1**2 * p2**2) - (2 * M2 * L1 * L2 * p1 * p2 * torch.cos(q1 - q2))) * torch.sin(2 * (q1 - q2))
    h2_denominator = 2 * L1**2 * L2**2 * (M1 + (M2 * torch.sin(q1 - q2)**2))**2
    h2 = h2_numerator / h2_denominator

    # Computation of Hamilton’s equations of motions
    dq1_dt = (L2 * p1 - L1 * p2 * torch.cos(q1 - q2)) / (L1**2 * L2 * (M1 + (M2 * torch.sin(q1 - q2)**2)))
    dq2_dt = (-M2 * L2 * p1 * torch.cos(q1 - q2) + ((M1 + M2) * L1 * p2)) / (M2 * L1 * L2**2 * (M1 + (M2 * torch.sin(q1 - q2)**2)))
    dp1_dt = (-(M1 + M2) * G * L1 * torch.sin(q1)) - h1 + h2
    dp2_dt = (-M2 * G * L2 * torch.sin(q2)) + h1 - -h2

    # Combine derivatives into a single tensor
    derivatives = torch.stack([dq1_dt, dq2_dt, dp1_dt, dp2_dt], dim=-1)

    return derivatives

# Monte Carlo sampling rewritten for PyTorch
def monte_carlo_sampling(q_range=(-torch.pi, torch.pi), p_range=(-1, 1), num_samples=1000) -> dict:
    """
    Generate training data for a Neural Network using Monte Carlo sampling.

    This function returns randomly sampled points in the state space with
        - angles q1 and q2 between q_range and p_range which is by default -pi to pi.
        - momentum p1 and p2 which is by default between -1 and 1
    together with the respective derivatives at these points which are calculated from the Hamiltonian equations.

    Parameters:
        q_range: Tuple of floats (min, max) defining the range of angles (q1, q2) to sample from
        p_range: Tuple of floats (min, max) defining the range of momenta (p1, p2) to sample from
        num_samples: Number of samples to generate

    Returns:
        data: A dictionary with keys 'states' and 'derivatives'
            - 'states': torch.Tensor of shape (num_samples, 4) containing [q1, q2, p1, p2]
            - 'derivatives': torch.Tensor of shape (num_samples, 4) containing [dq1/dt, dq2/dt, dp1/dt, dp2/dt]
    """
    # Randomly sample states
    q1_samples = torch.empty(num_samples).uniform_(*q_range)
    q2_samples = torch.empty(num_samples).uniform_(*q_range)
    p1_samples = torch.empty(num_samples).uniform_(*p_range)
    p2_samples = torch.empty(num_samples).uniform_(*p_range)

    # Combine sampled states into a tensor
    states = torch.stack([q1_samples, q2_samples, p1_samples, p2_samples], dim=1)

    # Compute derivatives for all sampled states using the vectorized `vector_field`
    derivatives = vector_field(states)

    # Return the data as a dictionary
    return {'states': states, 'derivatives': derivatives}
