import numpy as np
import matplotlib.pyplot as plt


# System definition
k1, k2, c1, c2 = 5, 3, 10, 2  # Model constants
m = 2  # Mass

# Initial Conditions - System model
x0, x0_dot = 2, 0
u = 0

# Initial guesses for model constants
m_hat, k1_hat, k2_hat, c1_hat, c2_hat = 0.5, 8, 5, 5, 1


# Reference definition
m_m = 1  # Ref model mass
k_m1, c_m1 = 6, 2.53  # Ref model constants
xm0, xm0_dot = 0.5, -1  # Initial Conditions - Ref model


# Adaptive Controller Constants
K = 1  # Constant
lamda0 = 1  # Constant of the z system dynamics
gamma = 50  # Adaptation gain


# Process Time & Reference Signal
continuous_rate = 1000  # Hz
dt = 1 / continuous_rate

t_start = 0
t_end = 50
t = np.arange(t_start, t_end + dt, dt)

ones = np.ones(t.shape)

first_input = 3.5 * np.sin(1.2 * t[:int(0.6 * len(t))])
second_input = 0.5 * np.ones(t[int(0.6 * len(t)):].shape)
r_sin = np.hstack((first_input, second_input))

plt.figure(figsize=(14, 6))
plt.xlabel("t [sec]", fontsize=20)
plt.ylabel("signal(t)", fontsize=20)
plt.title("Reference Signal", fontsize=24)
plt.plot(t, r_sin, linewidth=2)
plt.grid()
plt.savefig(f'images/Reference Signal.png')
plt.close()
