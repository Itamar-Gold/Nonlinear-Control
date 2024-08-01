from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import calculations as calc


def process_with_unknown_params(System, t, k1, k2, c1, c2, m, K, lamda0, k_m1, c_m1, gamma):
    # Array represents the initial state space model: [x0, x0_dot]
    x = System[0]
    x_dot = System[1]
    xm = System[2]
    xm_dot = System[3]
    m_hat = System[4]
    k1_hat = System[5]
    k2_hat = System[6]
    c1_hat = System[7]
    c2_hat = System[8]

    # Reference signal to the model reference output
    r = 5 * np.sin(2 * t)
    if t > 30:
        r = -3

    # # Step Response
    # r = 2

    # Reference model dynamics
    xm_ddot = -(k_m1 / m_m) * xm - (c_m1 / m_m) * xm_dot + r / m_m

    # controlled input for the model to track the output of the reference model
    u, xr_ddot, z = calc.controller(x, xm, x_dot, xm_dot, xm_ddot, m_hat, lamda0, K, k1_hat, k2_hat, c1_hat, c2_hat,
                                    params="unknown")

    # Saving data to plot the input u
    time_lst.append(t)
    input_lst.append(u)

    # Model dynamics
    x_ddot = -(k1 / m) * x - (k2 / m) * (x ** 3) - (c1 / m) * x_dot - (c2 / m) * (x_dot ** 2) + u / m

    # Adaptation law
    m_hat_dot, k1_hat_dot, k2_hat_dot, c1_hat_dot, c2_hat_dot = calc.adaptation_law(gamma, z, xr_ddot, x, x_dot)

    return [x_dot, x_ddot, xm_dot, xm_ddot, m_hat_dot, k1_hat_dot, k2_hat_dot, c1_hat_dot, c2_hat_dot]


# System definition
k1, k2, c1, c2 = 98, 80, 15, 0.34  # Model constants
m = 2  # Mass

# Initial Conditions - System model
x0, x0_dot = 0, 0
u = 0

# Reference definition
m_m = 1  # Ref model mass
k_m1, c_m1 = 70, 9  # Ref model constants
xm0, xm0_dot = 0, 0  # Initial Conditions - Ref model

# Initial Guess
m_ig = 1
k_1_ig, k_2_ig = 85, 100
c_1_ig, c_2_ig = 10, 0.1

# Process Time & Reference Signal
continuous_rate = 1000  # Hz
dt = 1 / continuous_rate

t_start = 0
t_end = 50
t = np.arange(t_start, t_end + dt, dt)

# Adaptive Controller Constants
K = 1  # Constant
lamda0 = 1  # Constant of the z system dynamics
gamma = 200  # Adaptation gain

time_lst, input_lst = [], []
# x0, x0_dot, xm0, xm0_dot = 0, 0, 0, 0  # Initial Conditions

System = [x0, x0_dot, xm0, xm0_dot, m_ig, k_1_ig, k_2_ig, c_1_ig, c_2_ig]  # Initial conditions

sys_response = odeint(process_with_unknown_params, System, t, args=(k1, k2, c1, c2, m, K, lamda0, k_m1, c_m1, gamma))
print(sys_response[-1, :])
# [x_dot, x_ddot, xm_dot, xm_ddot, m_hat_dot, k1_hat_dot, k2_hat_dot, c1_hat_dot, c2_hat_dot]


# Plot results
plt.tight_layout()

plt.figure(figsize=(9, 6))
plt.xlabel("t [sec]", fontsize=20)
plt.ylabel("x [m]", fontsize=20)
plt.plot(t, sys_response[:, 2], label=r"$x_r(t)$", linewidth=2)
plt.plot(t, sys_response[:, 0], label=r"$x(t)$", linewidth=2)
plt.legend(prop={'size': 17}, framealpha=1)
plt.title("Reference Tracking performance", fontsize=24)
plt.grid()
plt.savefig(f'images/Q4/Reference Tracking performance.png')
plt.close()


err = np.abs(sys_response[:, 0] - sys_response[:, 2])

plt.figure(figsize=(9, 6))
plt.xlabel("t [sec]", fontsize=20)
plt.ylabel("e(t)", fontsize=20)
plt.plot(t, err, label=r"$Error = |x(t)-x_r(t)|$", linewidth=2)
plt.legend(prop={'size': 17}, framealpha=1)
plt.grid()
plt.savefig(f'images/Q4/Error.png')
plt.close()


plt.figure(figsize=(14, 6))
plt.xlabel("t [sec]", fontsize=20)
plt.ylabel("Input", fontsize=20)
plt.plot(time_lst, input_lst, label=r"$u(t)$", linewidth=2)
plt.legend(prop={'size': 17}, framealpha=1)
plt.grid()
plt.savefig(f'images/Q4/Input.png')
plt.close()
