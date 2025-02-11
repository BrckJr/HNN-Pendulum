import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_ffnn(model: nn.Module, num_epochs: int, X: torch.tensor, Y: torch.tensor) -> list:
    """
    Train a given model using the given dataset with X and Y.

    Args:
        model (nn.Module): The model to train.
        num_epochs (int): Number of epochs to train the model.
        X (torch.tensor): Input data as a tensor of shape (n_samples, 2).
        Y (torch.tensor): Target data as a tensor of shape (n_samples, 2).

    Returns:
        list: A list containing the loss value for each epoch.
    """

    # Create Data Loader
    training_data = torch.utils.data.TensorDataset(X, Y)
    dataloader = DataLoader(training_data, batch_size=32, shuffle=True)

    # loss function and optimizer
    loss_fn = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for i, (X_batch, Y_batch) in enumerate(dataloader):

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
            print(f"Loss at epoch {epoch}: {epoch_loss}")

    return loss_history