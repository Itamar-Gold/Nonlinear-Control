import numpy as np
import os
import matplotlib.pyplot as plt
import calculations as calc


def discrete_adaptive_control(x0, x0_dot, xm0, xm0_dot, ones, t, discrete_control_rate, Improvement=True):

    err_lst = np.zeros((len(t), len(discrete_control_rate)))
    x_rate = np.zeros((len(t), len(discrete_control_rate)))

    for index, control_rate in enumerate(discrete_control_rate):

        # Initial Guess
        m_ig = 1
        k_1_ig, k_2_ig = 85, 100
        c_1_ig, c_2_ig = 10, 0.1
        u = 0

        # Reset lists
        x_lst, xm_lst, input_lst = [], [], []
        m_hat_lst, k1_hat_lst, k2_hat_lst, c1_hat_lst, c2_hat_lst = [], [], [], [], []

        # Initial Conditions
        x = x0
        x_dot = x0_dot
        xm = xm0
        xm_dot = xm0_dot

        for i, ts in enumerate(t):

            # Save the systems outputs
            x_lst.append(x)
            xm_lst.append(xm)
            m_hat_lst.append(m_ig)
            k1_hat_lst.append(k_1_ig)
            k2_hat_lst.append(k_2_ig)
            c1_hat_lst.append(c_1_ig)
            c2_hat_lst.append(c_2_ig)

            # Discrete Controller
            if i > 0 and i % (continuous_rate / control_rate) == 0:

                # Reference model dynamics
                xm_ddot = -(k_m1 / m) * xm - (c_m1 / m) * xm_dot + ones[i] / m

                # Integration
                xm_dot = calc.euler_integration(xm_dot, xm_ddot, dt * (continuous_rate / control_rate))
                xm = calc.euler_integration(xm, xm_dot, dt * (continuous_rate / control_rate))

                # controlled input for the model to track the output of the reference model
                if Improvement:
                    x_np1 = x + 1 * (1 / control_rate) * x_dot
                    u, xr_ddot, z = calc.controller(x_np1, xm, x_dot, xm_dot, xm_ddot, m_ig, lamda0, K, k_1_ig, k_2_ig,
                                                    c_1_ig, c_2_ig, params="unknown")

                else:
                    # controlled input for the model to track the output of the reference model
                    u, xr_ddot, z = calc.controller(x, xm, x_dot, xm_dot, xm_ddot, m_ig, lamda0, K,
                                                    k_1_ig, k_2_ig, c_1_ig, c_2_ig, params="unknown")

                # Adaptation Law
                m_hat_dot, k1_hat_dot, k2_hat_dot, c1_hat_dot, c2_hat_dot = calc.adaptation_law(gamma, z, xr_ddot, x, x_dot)

                # Integration
                m_ig = calc.euler_integration(m_ig, m_hat_dot, dt * (continuous_rate / control_rate))
                k_1_ig = calc.euler_integration(k_1_ig, k1_hat_dot, dt * (continuous_rate / control_rate))
                k_2_ig = calc.euler_integration(k_2_ig, k2_hat_dot, dt * (continuous_rate / control_rate))
                c_1_ig = calc.euler_integration(c_1_ig, c1_hat_dot, dt * (continuous_rate / control_rate))
                c_2_ig = calc.euler_integration(c_2_ig, c2_hat_dot, dt * (continuous_rate / control_rate))

            # Model dynamics
            x_ddot = -(k1 / m) * x - (k2 / m) * (x ** 3) - (c1 / m) * x_dot - (c2 / m) * (x_dot ** 2) + u / m

            # Integration
            x_dot = calc.euler_integration(x_dot, x_ddot, dt)
            x = calc.euler_integration(x, x_dot, dt)

            input_lst.append(u)

        err = np.abs(np.array(x_lst) - np.array(xm_lst))

        x_rate[:, index] = x_lst
        err_lst[:, index] = err

    return t, np.array(xm_lst), x_rate, err_lst


# System definition
k1, k2, c1, c2 = 98, 80, 15, 0.34  # Model constants
m = 2  # Mass

# Initial Conditions - System model
x0, x0_dot = 0, 0
u = 0

# Reference definition
m_m = 1  # Ref model mass
k_m1, c_m1 = 3.8, 2.3  # Ref model constants
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
ones = np.ones_like(t)

# Adaptive Controller Constants
K = 1  # Constant
lamda0 = 1  # Constant of the z system dynamics
gamma = 100  # Adaptation gain

discrete_control_rate = [1000, 500, 50, 10]  # Hz

im_t, im_xm_lst, im_x_rate, im_err_lst = discrete_adaptive_control(x0, x0_dot, xm0, xm0_dot,
                                                                   ones, t, discrete_control_rate, Improvement=True)
reg_t, reg_xm_lst, reg_x_rate, reg_err_lst = discrete_adaptive_control(x0, x0_dot, xm0, xm0_dot,
                                                                       ones, t, discrete_control_rate, Improvement=False)
# Plot the graphs for each control rate
for index, control_rate in enumerate(discrete_control_rate):
    path = f'images/Compare/results_for_{control_rate}hz'
    if path:
        # Create the directory
        os.makedirs(path, exist_ok=True)
        # Plot results
        plt.tight_layout()

        plt.figure(figsize=(9, 6))
        plt.xlabel("t [sec]", fontsize=20)
        plt.ylabel("x [m]", fontsize=20)
        plt.plot(t, im_xm_lst, label=r"$x_r(t)$", linewidth=2)
        plt.plot(t, im_x_rate[:, index], label='improved x(t)', linewidth=2)
        plt.plot(t, reg_x_rate[:, index], label='regular x(t)', linewidth=2)
        plt.legend(prop={'size': 17}, framealpha=1)
        plt.title("improved / regular tracking", fontsize=24)
        plt.grid()
        plt.savefig(f'{path}/Step Reference Tracking performance.png')
        plt.close()

        plt.figure(figsize=(9, 6))
        plt.xlabel("t [sec]", fontsize=20)
        plt.ylabel("e(t)", fontsize=20)
        plt.plot(t, im_err_lst[:, index], label='Improved Error', linewidth=2)
        plt.plot(t, reg_err_lst[:, index], label='Regular Error', linewidth=2)
        plt.legend(prop={'size': 17}, framealpha=1)
        plt.title("improved / regular error tracking", fontsize=24)
        plt.grid()
        plt.savefig(f'{path}/Step Error performance.png')
        plt.close()
