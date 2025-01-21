import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

# Automatically select the device (CUDA, MPS, or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Function to plot gradients
def plot_gradients(true_gradients, predicted_gradients, epoch):
    true_gradients = true_gradients.cpu().detach().numpy()
    predicted_gradients = predicted_gradients.cpu().detach().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    labels = ['dq1/dt', 'dq2/dt', 'dp1/dt', 'dp2/dt']

    for i, ax in enumerate(axes.flatten()):
        ax.scatter(true_gradients[:, i], predicted_gradients[:, i], alpha=0.6, label='Predicted vs True')
        ax.plot(true_gradients[:, i], true_gradients[:, i], 'r--', label='Perfect Match')
        ax.set_title(f'Gradient: {labels[i]}')
        ax.set_xlabel('True Gradient')
        ax.set_ylabel('Predicted Gradient')
        ax.legend()

    plt.suptitle(f'Gradient Comparison at Epoch {epoch}')
    plt.tight_layout()
    plt.show()


def data_loss(model: nn.Module, X_measured: torch.Tensor, H_measured: torch.Tensor) -> torch.Tensor:
    H_pred = model(X_measured).squeeze()
    return torch.mean((H_pred - H_measured) ** 2)


def physics_loss(model: nn.Module, X_train: torch.Tensor, Y_train: torch.Tensor, epoch: int, counter: int) -> torch.Tensor:
    X_train = X_train.requires_grad_()
    q1, q2, p1, p2 = X_train[:, 0], X_train[:, 1], X_train[:, 2], X_train[:, 3]

    H = model(torch.stack([q1, q2, p1, p2], dim=1))

    grad_H = torch.autograd.grad(H, (q1, q2, p1, p2), grad_outputs=torch.ones_like(H), create_graph=True)

    dH_dq1, dH_dq2, dH_dp1, dH_dp2 = grad_H
    dq1_dt, dq2_dt, dp1_dt, dp2_dt = dH_dp1, dH_dp2, -dH_dq1, -dH_dq2

    dq1_dt_true, dq2_dt_true, dp1_dt_true, dp2_dt_true = Y_train[:, 0], Y_train[:, 1], Y_train[:, 2], Y_train[:, 3]

    predicted_gradients = torch.stack([dq1_dt, dq2_dt, dp1_dt, dp2_dt], dim=1)
    true_gradients = torch.stack([dq1_dt_true, dq2_dt_true, dp1_dt_true, dp2_dt_true], dim=1)

    if epoch % 250 == 0 and counter == 0:
        plot_gradients(true_gradients, predicted_gradients, epoch)

    MSE_loss = torch.mean((dH_dp1 - dq1_dt_true) ** 2 +
                          (dH_dp2 - dq2_dt_true) ** 2 +
                          (dH_dq1 + dp1_dt_true) ** 2 +
                          (dH_dq2 + dp2_dt_true) ** 2)

    return MSE_loss


def train_hnn(model: nn.Module, num_epochs: int, X_train: torch.Tensor, Y_train: torch.Tensor,
              X_measured: torch.Tensor, H_measured: torch.Tensor, lam_data: float = 0.1, lam_phy: float = 0.9) -> list:

    batch_size = 128
    training_data = torch.utils.data.TensorDataset(X_train, Y_train)
    dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

    model = model.to(device)
    X_measured, H_measured = X_measured.to(device), H_measured.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)

    loss_history = []
    for epoch in range(num_epochs):
        if epoch == 0:
            print(f"Training on device: {device}")

        model.train()
        epoch_loss = 0.0
        epoch_data_loss = 0.0
        epoch_physics_loss = 0.0
        for i, (X_batch, Y_batch) in enumerate(dataloader):
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()

            p_loss = physics_loss(model, X_batch, Y_batch, epoch, i)
            d_loss = data_loss(model, X_measured, H_measured)
            batch_loss = lam_data * d_loss + lam_phy * p_loss

            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()
            epoch_data_loss += lam_data * d_loss.item()
            epoch_physics_loss += lam_phy * p_loss.item()

        scheduler.step()

        loss_history.append(epoch_loss / len(dataloader))

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss={epoch_loss:.2f} with physics loss {epoch_physics_loss:.2f} and data loss {epoch_data_loss:.2f}")

    return loss_history
