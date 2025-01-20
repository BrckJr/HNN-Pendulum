import torch
import constants

# Constants for the single pendulum
M = constants.M  # Mass of the pendulum
L = constants.L  # Length of the pendulum
G = constants.G  # Gravitational constant

# Hamiltonian function for the single pendulum
def hamiltonian(system_states: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hamiltonian (total energy) of the single pendulum for a tensor of states.

    Parameters:
        system_states: torch.Tensor of shape (N, 2)
            - Each row contains [q, p], representing the states of the system.

    Returns:
        H: torch.Tensor of shape (N,)
            - Hamiltonian (total energy) for each state.
    """
    # Extract states
    system_states = system_states.detach()
    if system_states.ndim == 1:
        q, p = system_states[0], system_states[1]
    else:
        q, p = system_states[:, 0], system_states[:, 1]

    # Kinetic energy (T = p^2 / 2M)
    H_kin = p**2 / (2 * M)

    # Potential energy (V = M * G * L * (1 - cos(q)))
    H_pot = M * G * L * (1 - torch.cos(q))

    # Total Hamiltonian (H = T + V)
    H = H_kin + H_pot

    return H

# Vector field function for the single pendulum
def vector_field(system_states: torch.Tensor) -> torch.Tensor:
    """
    Compute the time derivatives (vector field) for the single pendulum for a tensor of states.

    Parameters:
        system_states: torch.Tensor of shape (N, 2) or (2)
            - Each row contains [q, p], representing the states of the system.

    Returns:
        derivatives: torch.Tensor of shape (N, 2)
            - Each row contains [dq/dt, dp/dt].
    """
    # Extract states
    system_states = system_states.detach()
    if system_states.ndim == 1:
        q, p = system_states[0], system_states[1]
    else:
        q, p = system_states[:, 0], system_states[:, 1]

    # Compute time derivatives from Hamilton's equations
    dq_dt = p / M  # dq/dt = p / M
    dp_dt = -M * G * L * torch.sin(q)  # dp/dt = -M * G * L * sin(q)

    # Combine derivatives into a single tensor
    derivatives = torch.stack([dq_dt, dp_dt], dim=-1)

    return derivatives

# Monte Carlo sampling for the single pendulum
def monte_carlo_sampling(q_range=(-torch.pi, torch.pi), p_range=(-1, 1), num_samples=1000) -> dict:
    """
    Generate training data for a Neural Network using Monte Carlo sampling.

    This function returns randomly sampled points in the state space with
        - angle q between q_range and momentum p between p_range.

    Parameters:
        q_range: Tuple of floats (min, max) defining the range of angles q to sample from
        p_range: Tuple of floats (min, max) defining the range of momenta p to sample from
        num_samples: Number of samples to generate

    Returns:
        data: A dictionary with keys 'states' and 'derivatives'
            - 'states': torch.Tensor of shape (num_samples, 2) containing [q, p]
            - 'derivatives': torch.Tensor of shape (num_samples, 2) containing [dq/dt, dp/dt]
    """
    # Randomly sample states
    q_samples = torch.empty(num_samples).uniform_(*q_range)
    p_samples = torch.empty(num_samples).uniform_(*p_range)

    # Combine sampled states into a tensor
    states = torch.stack([q_samples, p_samples], dim=1)

    # Compute derivatives for all sampled states using the vectorized `vector_field`
    derivatives = vector_field(states)

    # Return the data as a dictionary
    return {'states': states, 'derivatives': derivatives}
