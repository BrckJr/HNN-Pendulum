import torch
import torch.nn as nn

import utils
import constants
import double_pendulum

import FFNN.FFNN as FFNN
import FFNN.FFNN_utils as FFNN_utils
import HNN.HNN as HNN

import solvers.explicit_euler as explicit_euler
import solvers.symplectic_euler as symplectic_euler
from double_pendulum import hamiltonian

# Automatically select the device (CUDA, MPS, or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Convert initial state into a torch tensor
Y0 = torch.tensor(constants.Y0, dtype=torch.float32)
T_SPAN = constants.T_SPAN

def solve_numerically(selected_solver: str) -> None:
    """
    Solve the PDE of the double pendulum numerically with a selected solver.

    Args:
        selected_solver (str): Numerical solver to use. Options: "Explicit Euler", "Symplectic Euler"

    Returns:
        None
    """

    # Set the numerical solver stated in the selected_solver argument
    match selected_solver:
        case "Explicit Euler":
            func = explicit_euler.solve
        case "Symplectic Euler":
            func = symplectic_euler.solve
        case _:
            raise ValueError(f"{selected_solver} is not a known solver.")

    # Solve PDE for initial state Y0 and time span t_span
    t_values, y_values = func(double_pendulum.vector_field, '_', Y0, T_SPAN)

    # Plot the complete trajectory for the double pendulum over the whole time range
    utils.plot_positions_in_cartesian(t_values, y_values, selected_solver)

    # Plot the Hamiltonian over time to see if it stays constant
    utils.plot_hamiltonian_deviation_over_time(t_values, y_values, selected_solver)

def learn_hamiltonian_and_solve(selected_model: str) -> nn.Module:
    """
    Learn the Hamiltonian function from data and use the learned model to solve the PDE with the Symplectic Euler method.

    Args:
        selected_model (str): Model which should be used for learning the Hamiltonian

    Returns:
        nn.Module: The trained model.
    """

    data_samples = double_pendulum.monte_carlo_sampling(num_samples=10000)  # Generate training data
    X_train, Y_train = data_samples['states'], data_samples['derivatives']  # Y_train is only used for the FFNN

    print(f"\n --- Start Using of {selected_model} --- \n")

    match selected_model:
        case "FFNN":
            model = FFNN.FFNN(input_dim=4, hidden_dim=256, output_dim=4) # Use simple FFNN as model
            """
            loss_history = FFNN_utils.train_ffnn(
                model=model,
                num_epochs=1000,
                X=X_train,
                Y=Y_train,
                device=device
            )
            torch.save(model.state_dict(),"FFNN/FFNN_model.pth")  # Only save the learned parameters
            """

            # Load the trained model
            model.load_state_dict(torch.load("FFNN/FFNN_model.pth"))

        case "HNN":
            # Simulate "measuring" of data points for the supervised (data) loss
            X_measured = double_pendulum.monte_carlo_sampling(num_samples=50)['states']
            H_measured = hamiltonian(X_measured)

            model = HNN.HNN(input_dim=4, hidden_dim=1028, output_dim=1) # Use the HNN as model
            """
            loss_history = HNN_utils.train_hnn(
                model=model,
                num_epochs=5000,
                X_train=X_train,
                Y_train=Y_train,
                X_measured=X_measured,
                H_measured=H_measured,
                device=device
            )
            torch.save(model.state_dict(),"HNN/HNN_model.pth")  # Only save the learned parameters
            """

            # Load the trained model
            model.load_state_dict(torch.load("HNN/HNN_model.pth"))
        case _:
            raise ValueError(f"{selected_model} is not a known solver.")

    # Plot loss function over the training epochs
    # utils.plot_losses(loss_history, selected_model)

    # Use the trained network and solve with Symplectic Euler solver
    model.eval()
    t_values, y_values = symplectic_euler.solve(model, selected_model, Y0, T_SPAN)

    # Plot the complete trajectory for the double pendulum over the whole time range
    utils.plot_positions_in_cartesian(t_values, y_values, f"{selected_model} with Symplectic Euler")

    # Plot the Hamiltonian over time to see if it stays constant
    utils.plot_hamiltonian_deviation_over_time(t_values, y_values, f"{selected_model} with Symplectic Euler")

    return model

if __name__ == '__main__':
    # Set the numerical solver for solving the known PDE
    use_solver = "Symplectic Euler" # Alternatives: "Symplectic Euler" and "Explicit Euler"

    # Numerically solve the known PDE with the selected solver
    solve_numerically(use_solver)

    # Set the model to learn the Hamiltonian
    use_model = "HNN" # Alternatives: "HNN", "FFNN"

    # Learn the Hamiltonian and solve the learned PDE with the Symplectic Euler method
    trained_model = learn_hamiltonian_and_solve(use_model)
