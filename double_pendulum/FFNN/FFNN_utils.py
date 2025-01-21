import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_ffnn(
        model: nn.Module,
        num_epochs: int,
        X: torch.Tensor,
        Y: torch.Tensor,
        device: torch.device = torch.device("cpu")
) -> list:
    """
    Train a given model using the given dataset with X and Y.

    Args:
        model (nn.Module): The model to train.
        num_epochs (int): Number of epochs to train the model.
        X (torch.Tensor): Input data as a tensor of shape (n_samples, 4).
        Y (torch.Tensor): Target data as a tensor of shape (n_samples, 4).
        device (torch.device, optional): The device on which the computation will be performed. Defaults to "cpu".

    Returns:
        list: A list containing the loss value for each epoch.
    """

    # Move model to the device
    model = model.to(device)

    # Move data to the device
    X, Y = X.to(device), Y.to(device)

    # Create Data Loader
    training_data = torch.utils.data.TensorDataset(X, Y)
    dataloader = DataLoader(training_data, batch_size=32, shuffle=True)

    # Loss function and optimizer
    loss_fn = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, Y_batch in dataloader:
            # Move batch data to the device
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            # Reset the optimizer
            optimizer.zero_grad()

            # Forward pass
            Y_pred = model(X_batch)
            batch_loss = loss_fn(Y_pred, Y_batch)

            # Backward pass and optimization
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()

        # Collect all epoch losses
        loss_history.append(epoch_loss)

        if epoch % 100 == 0:
            print(f"Loss at epoch {epoch}: {epoch_loss:.4f}")

    return loss_history
