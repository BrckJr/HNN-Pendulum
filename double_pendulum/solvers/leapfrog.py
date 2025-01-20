import torch
import torch.nn as nn

def get_vector_field(model: nn.Module, y: torch.Tensor) -> torch.Tensor:
    """
    Calculate the derivatives for the points in the phase state based on the trained HNN.

    Args:
        model (nn.Module): Trained Hamiltonian Neural Network (HNN).
        y (torch.Tensor): Current state, a tensor of shape (4) containing [q1, q2, p1, p2].

    Returns:
        torch.Tensor: Tensor of shape (4) containing [dq1/dt, dq2/dt, dp1/dt, dp2/dt].
    """
    y = y.detach().clone().requires_grad_(True)
    H = model(y)

    grad_H = torch.autograd.grad(H, y, grad_outputs=torch.ones_like(H), create_graph=True)[0]

    # Extract predicted time derivatives using Hamilton's equations
    q1_dot_pred = grad_H[0]   # ∂H/∂p1
    q2_dot_pred = grad_H[1]   # ∂H/∂p2
    p1_dot_pred = -grad_H[2]  # -∂H/∂q1
    p2_dot_pred = -grad_H[3]  # -∂H/∂q2

    return torch.stack([q1_dot_pred, q2_dot_pred, p1_dot_pred, p2_dot_pred])


def step(func, y: torch.Tensor, h: float) -> torch.Tensor:
    """
    Perform a single step of the Leapfrog method.

    Args:
        func (Union[nn.Module, Callable]): Function or model that computes the time derivatives (vector field).
        y (torch.Tensor): Current state, a tensor of shape (4) containing [q1, q2, p1, p2].
        h (float): Step size.

    Returns:
        torch.Tensor: Updated state, a tensor of shape (4).
    """
    # Split state into positions and momenta
    q = y[:2]  # Positions [q1, q2]
    p = y[2:]  # Momenta [p1, p2]

    # Compute half-step momentum update
    if isinstance(func, nn.Module):
        derivatives = get_vector_field(func, y)  # Get derivatives for current state from the trained model
    else:
        derivatives = func(y)  # Get derivatives for current state from known vector field

    p_half = p + 0.5 * h * derivatives[2:]

    # Update positions using half-step momentum
    q_next = q + h * func(torch.cat((q, p_half)))[:2]

    # Compute full-step momentum update
    if isinstance(func, nn.Module):
        derivatives_next = get_vector_field(func, torch.cat((q_next, p_half)))  # Compute full-step momentum update with the trained model
    else:
        derivatives_next = func(torch.cat((q_next, p_half)))  # Compute full-step momentum update with known vector field

    p_next = p_half + 0.5 * h * derivatives_next[2:]

    # Combine updated positions and momenta
    y_next = torch.cat((q_next, p_next))

    return y_next


def solve(func, y0: torch.Tensor, t_span: tuple, h: float = 0.01) -> tuple:
    """
    Solve the system of ODEs using the Leapfrog method.

    Args:
        func (Union[nn.Module, Callable]): Function or model that computes the time derivatives (vector field).
        y0 (torch.Tensor): Initial state, a tensor of shape (4) containing [q1, q2, p1, p2].
        t_span (tuple): A tuple (t_start, t_end) defining the time interval.
        h (float, optional): Step size. Defaults to 0.01.

    Returns:
        tuple: A tuple containing:
            - t_values (torch.Tensor): Tensor of time points.
            - y_values (torch.Tensor): Tensor of state values at the corresponding time points.
    """
    t_start, t_end = t_span
    t = t_start
    y = y0.clone().detach()

    t_values = [t]
    y_values = [y.clone().detach()]

    while t < t_end:
        # Ensure we don't step past the end time
        if t + h > t_end:
            h = t_end - t

        # Perform a single Leapfrog step
        y = step(func, y, h)
        t += h

        # Store the results
        t_values.append(t)
        y_values.append(y.clone().detach())

    return torch.tensor(t_values, dtype=torch.float32), torch.stack(y_values)
