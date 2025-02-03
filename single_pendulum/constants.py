import numpy as np

# System constants defined globally
M = 1.0  # Mass of the pendulum
L = 1.0  # Length of the pendulum
G = 9.81  # Gravitational constant

# Define initial state and time span for simulation
# q1: Angle at which pendulum 1 is position. (0, -1) is default position for q1=0.
# q2: Angle at which pendulum 2 is positioned relative to pendulum 1. q2=0 means pendulum 2 is vertically under pendulum 1
# p1: Momentum in x direction
# p2: Momentum in y direction
Y0 = [0.4, 0.0]  # Initial conditions [q, p]
T_SPAN = (0, 10)           # Time range for the simulation
