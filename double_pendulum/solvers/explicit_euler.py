import torch

def step(func, y, h):
    """
    Perform a single step of the explicit Euler method.

    Args:
        func (Callable): Function that computes the time derivatives (vector field).
        y (torch.Tensor): Current state, a tensor of shape (4) containing [q1, q2, p1, p2].
        h (float): Step size.

    Returns:
        torch.Tensor: Updated state, a tensor of shape (4).
    """
    # Compute the current derivatives from the vector field
    derivatives = func(y)

    # Update the momenta (p1, p2) using the current derivatives
    y[2:] = y[2:] + h * derivatives[2:]

    # Update the positions (q1, q2) using the current derivatives
    y[:2] = y[:2] + h * derivatives[:2]

    return y


def solve(func, y0, t_span, h=0.01):
    """
    Solve the system of ODEs using the explicit Euler method.

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

        # Perform a single explicit Euler step
        y = step(func, y, h)
        t += h

        # Store the results
        t_values.append(t)
        y_values.append(y.clone().detach())

    return torch.tensor(t_values, dtype=torch.float32), torch.stack(y_values)
