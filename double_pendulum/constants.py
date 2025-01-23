import numpy as np

# System constants defined globally
M1, M2 = 1.0, 1.0  # Masses of the pendulums
L1, L2 = 1.0, 1.0  # Lengths of the pendulums
G = 9.81           # Gravity constant

# Define initial state and time span for simulation
# q1: Angle at which pendulum 1 is position. (0, -1) is default position for q1=0.
# q2: Angle at which pendulum 2 is positioned relative to pendulum 1. q2=0 means pendulum 2 is vertically under pendulum 1
# p1: Momentum in x direction
# p2: Momentum in y direction
Y0 = [0.8, 0.0, 0.0, 0.0]  # Initial conditions [q1, q2, p1, p2]
T_SPAN = (0, 1)           # Time range for the simulation
