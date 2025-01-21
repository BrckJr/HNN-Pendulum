import torch
import torch.nn as nn

def get_vector_field(
        model: nn.Module,
        y: torch.Tensor,
        device: torch.device
) -> torch.Tensor:
    """
    Calculate the derivatives for the points in the phase state based on the trained HNN.

    Args:
        model (nn.Module): Trained Hamiltonian Neural Network (HNN).
        y (torch.Tensor): Current state, a tensor of shape (4) containing [q1, q2, p1, p2].
        device (torch.device): The device on which the computation will be performed.

    Returns:
        torch.Tensor: Tensor of shape (4) containing [dq1/dt, dq2/dt, dp1/dt, dp2/dt].
    """
    y = y.to(device).detach().clone().requires_grad_(True)
    model = model.to(device)

    model.eval()
    H = model(y)

    grad_H = torch.autograd.grad(H, y, grad_outputs=torch.ones_like(H), create_graph=True)[0]

    # Extract predicted time derivatives using Hamilton's equations
    q1_dot_pred = grad_H[2]
    q2_dot_pred = grad_H[3]
    p1_dot_pred = -grad_H[0]
    p2_dot_pred = -grad_H[1]

    return torch.stack([q1_dot_pred, q2_dot_pred, p1_dot_pred, p2_dot_pred])


def step(
        func,
        func_type: str,
        y: torch.Tensor,
        h: float,
        device: torch.device
) -> torch.Tensor:
    """
    Perform a single step of the Leapfrog method.

    Args:
        func (Union[nn.Module, Callable]): Function or model that computes the time derivatives (vector field).
        func_type (str): Either "HNN", "FFNN" or "_" for vector_field
        y (torch.Tensor): Current state, a tensor of shape (4) containing [q1, q2, p1, p2].
        h (float): Step size.
        device (torch.device): The device on which the computation will be performed.

    Returns:
        torch.Tensor: Updated state, a tensor of shape (4).
    """
    y.to(device)

    if func_type == 'HNN':
        derivatives = get_vector_field(func, y, device)  # Get derivatives for current state from the trained model
    elif func_type == 'FFNN':
        with torch.no_grad():
            derivatives = func(y) # Get derivatives for current state from the trained model
    else:
        derivatives = func(y)  # Get derivatives for current state from known vector field

    # Update the momenta (p1, p2) using the current derivatives
    y[2:] = y[2:] + h * derivatives[2:]

    # Compute the new derivatives after momentum update
    if func_type == 'HNN':
        derivatives = get_vector_field(func, y, device)  # Get derivatives for current state from the trained model
    elif func_type == 'FFNN':
        with torch.no_grad():
            derivatives = func(y) # Get derivatives for current state from the trained model
    else:
        derivatives = func(y)  # Get derivatives for current state from known vector field

    # Update the positions (q1, q2) using the updated momentum
    y[:2] = y[:2] + h * derivatives[:2]

    return y


def solve(
        func,
        func_type: str,
        y0: torch.Tensor,
        t_span: tuple,
        h: float = 0.001,
        device: torch.device = torch.device("cpu")
) -> tuple:
    """
    Solve the system of ODEs using the Leapfrog method.

    Args:
        func (Union[nn.Module, Callable]): Function or model that computes the time derivatives (vector field).
        func_type (str): Either "HNN", "FFNN" or "Numerical"
        y0 (torch.Tensor): Initial state, a tensor of shape (4) containing [q1, q2, p1, p2].
        t_span (tuple): A tuple (t_start, t_end) defining the time interval.
        h (float, optional): Step size. Defaults to 0.01.
        device (torch.device, optional): The device on which the computation will be performed. Defaults to "cpu".

    Returns:
        tuple: A tuple containing:
            - t_values (torch.Tensor): Tensor of time points.
            - y_values (torch.Tensor): Tensor of state values at the corresponding time points.
    """
    t_start, t_end = t_span
    t = t_start
    y = y0.clone().detach().to(device)

    t_values = [t]
    y_values = [y.clone().detach()]

    while t < t_end:
        # Ensure we don't step past the end time
        if t + h > t_end:
            h = t_end - t

        # Perform a single Leapfrog step
        y = step(func, func_type, y, h, device)
        t += h

        # Store the results
        t_values.append(t)
        y_values.append(y.clone().detach().cpu())

    return torch.tensor(t_values, dtype=torch.float32), torch.stack(y_values)