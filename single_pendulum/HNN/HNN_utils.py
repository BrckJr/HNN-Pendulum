import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

def plot_gradients(
        true_gradients: tuple,
        predicted_gradients: tuple,
        epoch: int
) -> None:
    """
    Plot the true gradients vs. the predicted gradients.

    Args:
        true_gradients (tuple): A tuple containing true gradients [dq/dt, dp/dt].
        predicted_gradients (tuple): A tuple containing predicted gradients [dq/dt, dp/dt].
        epoch (int): Current training epoch for labeling the plot.
    """
    # Convert tuples to numpy arrays for plotting
    true_gradients = [tg.detach().numpy() for tg in true_gradients]
    predicted_gradients = [pg.detach().numpy() for pg in predicted_gradients]

    # Create subplots for each gradient component
    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    labels = ['dq/dt', 'dp/dt']

    for i, ax in enumerate(axes.flatten()):
        # Scatter plot of true vs predicted gradients
        ax.scatter(true_gradients[i], predicted_gradients[i], alpha=0.6, label='Predicted vs True')

        # Plot a perfect match line (y=x) for reference
        ax.plot(true_gradients[i], true_gradients[i], 'r--', label='Perfect Match')  # y=x line

        # Set plot labels and titles
        ax.set_title(f'Gradient: {labels[i]}')
        ax.set_xlabel('True Gradient')
        ax.set_ylabel('Predicted Gradient')
        ax.legend()

    # Set the main title for the entire figure
    plt.suptitle(f'Gradient Comparison at Epoch {epoch}')
    plt.tight_layout()
    plt.show()

def data_loss(
        model: nn.Module,
        X_measured: torch.Tensor,
        H_measured: torch.Tensor
) -> torch.Tensor:
    """ Compute the mean squared error data loss based on the provided data.

    This ensures that our model also fits to data points we measured in reality

    Args:
        model (nn.Module): The trained Hamiltonian neural network.
        X_measured (torch.tensor): Points in phase space for which true Hamiltonian is known (n_samples, 2).
        H_measured (torch.tensor): True Hamiltonian at the X_measured points (n_samples).

    Returns:
        torch.Tensor: data loss
    """

    H_pred = model(X_measured).squeeze()
    return torch.mean((H_pred - H_measured) ** 2)

def physics_loss(
        model: nn.Module,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        epoch: int,
        counter: int
) -> torch.Tensor:
    """
    MSE between predicted and true derivatives in the phase space.

    Compute the MSE loss between the predicted values of the derivatives dq_t and dp_dt and the true
    values stored in Y_train.

    Args:
        model: The Hamiltonian Neural Network.
        X_train (torch.Tensor): Input data containing (normalized) [q, p] (batch_size, 2)
        Y_train (torch.Tensor): Target data containing (normalized) [dq_t, dp_dt] (batch_size, 2)
        epoch (int): Current training epoch.
        counter (int): Current batch number to ensure gradient plots are shown only once

    Returns:
        torch.Tensor: physics loss
    """

    # Extract single features
    q, p = X_train[:, 0], X_train[:, 1]

    # Enable autograd for q and p to compute partial derivatives
    q.requires_grad_(), p.requires_grad_()

    # Compute the Hamiltonian
    hamiltonian = model(torch.stack((q, p), dim=1)).squeeze()

    # Calculate dH/dq and dH/dp
    grad_H = torch.autograd.grad(hamiltonian, (q, p), grad_outputs=torch.ones_like(hamiltonian), create_graph=True)
    dH_dq, dH_dp = grad_H[0], grad_H[1]

    # Calculate dq_dt and dp_dt from the gradients of the Hamiltonian
    dq_dt = dH_dp
    dp_dt = -dH_dq

    # Compute the true vector field (dq_dt, dp_dt) for the single pendulum
    dq_dt_true, dp_dt_true = Y_train[:, 0], Y_train[:, 1]

    # Plot gradients every 100 epochs
    if epoch % 100 == 0 and counter == 0:
        plot_gradients((dq_dt_true, dp_dt_true), (dq_dt, dp_dt), epoch)

    # Calculate the sum of mean squared error between true and computed values of the derivatives (dq/dt and dp/dt)
    loss = torch.mean((dq_dt - dq_dt_true) ** 2 + (dp_dt - dp_dt_true) ** 2)

    return loss

def train_hnn(
        model: nn.Module,
        num_epochs: int,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        X_measured: torch.Tensor,
        H_measured: torch.Tensor,
        lam_data: float = 0.1,
        lam_phy: float = 1.0
) -> list:
    """
    Train the given Hamiltonian Neural Network.

    Args:
        model (nn.Module): The model to train.
        num_epochs (int): Number of epochs to train the model.
        X_train (torch.Tensor): Batch of points in phase space containing [q, p] (batch_size, 2)
        Y_train (torch.Tensor): Batch of derivatives containing [dq_t, dp_dt] (batch_size, 2)
        X_measured (torch.tensor): Points in phase space for which true Hamiltonian is known (n_samples, 2).
        H_measured (torch.tensor): True Hamiltonian at the X_measured points (n_samples).
        lam_data (float): Weighting of the supervised (data) loss.
        lam_phy (float): Weighting of the unsupervised (physics) loss.

    Returns:
        list: A list containing the loss value for each epoch.
    """

    # Create Data Loader
    batch_size = 64
    training_data = torch.utils.data.TensorDataset(X_train, Y_train)
    dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    # Set optimizer for parameter adjustment
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Use a learning rate scheduler for better training
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.25)  # Halve LR every 250 epochs

    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_data_loss = 0.0
        epoch_physics_loss = 0.0
        for i, (X_batch, Y_batch) in enumerate(dataloader):

            # Reset the optimizer
            optimizer.zero_grad()

            # Forward pass
            p_loss = physics_loss(model, X_batch, Y_batch, epoch, i)
            d_loss = data_loss(model, X_measured, H_measured)
            batch_loss = lam_data * d_loss + lam_phy * p_loss

            # Backward pass and optimization
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()

            # Check the dominating part in loss
            epoch_data_loss += lam_data * d_loss.item()
            epoch_physics_loss += lam_phy * p_loss.item()

        # Step with the scheduler
        scheduler.step()

        # Collect averaged batch loss for each epoch
        loss_history.append(epoch_loss / batch_size)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss={epoch_loss:.2f} with physics loss {epoch_physics_loss:.2f} and data loss {epoch_data_loss:.2f}")

    return loss_history
