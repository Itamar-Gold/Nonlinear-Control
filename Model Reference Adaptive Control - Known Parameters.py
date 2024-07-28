from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import calculations as calc


def process_with_known_params(System, t, k1, k2, c1, c2, m, K, lamda0, k_m1, c_m1):
    # Array represents the initial state space model: [x0, x0_dot]
    x = System[0]
    x_dot = System[1]
    xm = System[2]
    xm_dot = System[3]

    # Reference signal to the model reference output
    r = 3.5 * np.sin(1.2 * t)
    if t > 30:
        r = 0.5

    # Reference model dynamics
    xm_ddot = -(k_m1 / m_m) * xm - (c_m1 / m_m) * xm_dot + r / m_m

    # controlled input for the model to track the output of the reference model
    u = calc.controller(x, xm, x_dot, xm_dot, xm_ddot, m, lamda0, K, k1, k2, c1, c2, params="known")

    # Saving data to plot the input u
    time_lst.append(t)
    input_lst.append(u)

    # Model dynamics
    x_ddot = -(k1 / m) * x - (k2 / m) * (x ** 3) - (c1 / m) * x_dot - (c2 / m) * (x_dot ** 2) + u / m

    return [x_dot, x_ddot, xm_dot, xm_ddot]


def generate_plot(t, xm, x, ts, u, **kwargs):


    if kwargs.__len__() > 1:
        plt.figure(figsize=(16, 16))
        plt.subplot(3, 2, 1)
        plt.xlabel("t [sec]", fontsize=17)
        plt.ylabel("Parameter estimation", fontsize=17)
        plt.plot(t, kwargs["m"], label=r"$m$", linewidth=2)
        plt.plot(t, kwargs["m_hat"], label=r"$\hat{m}$", linewidth=2)
        plt.legend(prop={'size': 17}, framealpha=1)
        plt.grid()

        plt.subplot(3, 2, 2)
        # plt.figure(figsize=(9,6))
        plt.xlabel("t [sec]", fontsize=17)
        plt.ylabel("Parameter estimation", fontsize=17)
        plt.plot(t, kwargs["k1"], label=r"$k_1$", linewidth=2)
        plt.plot(t, kwargs["k1_hat"], label=r"$\hat{k_1}$", linewidth=2)
        plt.legend(prop={'size': 17}, framealpha=1)
        plt.grid()

        plt.subplot(3, 2, 3)
        # plt.figure(figsize=(9,6))
        plt.xlabel("t [sec]", fontsize=17)
        plt.ylabel("Parameter estimation", fontsize=17)
        plt.plot(t, kwargs["k2"], label=r"$k_2$", linewidth=2)
        plt.plot(t, kwargs["k2_hat"], label=r"$\hat{k_2}$", linewidth=2)
        plt.legend(prop={'size': 17}, framealpha=1)
        plt.grid()

        plt.subplot(3, 2, 4)
        # plt.figure(figsize=(9,6))
        plt.xlabel("t [sec]", fontsize=17)
        plt.ylabel("Parameter estimation", fontsize=17)
        plt.plot(t, kwargs["c1"], label=r"$c_1$", linewidth=2)
        plt.plot(t, kwargs["c1_hat"], label=r"$\hat{c_1}$", linewidth=2)
        plt.legend(prop={'size': 17}, framealpha=1)
        plt.grid()

        plt.subplot(3, 2, 5)
        # plt.figure(figsize=(9,6))
        plt.xlabel("t [sec]", fontsize=17)
        plt.ylabel("Parameter estimation", fontsize=17)
        plt.plot(t, kwargs["c2"], label=r"$c_2$", linewidth=2)
        plt.plot(t, kwargs["c2_hat"], label=r"$\hat{c_2}$", linewidth=2)
        plt.legend(prop={'size': 17}, framealpha=1)
        plt.grid()
    plt.show()


# System definition
k1, k2, c1, c2 = 5, 3, 10, 2  # Model constants
m = 2  # Mass

# Initial Conditions - System model
x0, x0_dot = 2, 0
u = 0

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

time_lst, input_lst = [], []
System = [x0, x0_dot, xm0, xm0_dot]  # Initial conditions
sys_response = odeint(process_with_known_params, System, t, args=(k1, k2, c1, c2, m, K, lamda0, k_m1, c_m1))

# Plot results
plt.tight_layout()

plt.figure(figsize=(9, 6))
plt.xlabel("t [sec]", fontsize=20)
plt.ylabel("Tracking performance", fontsize=20)
plt.plot(t, sys_response[:, 2], label=r"$x_m(t)$", linewidth=2)
plt.plot(t, sys_response[:, 0], label=r"$x(t)$", linewidth=2)
plt.legend(prop={'size': 17}, framealpha=1)
plt.title("Reference Tracking performance", fontsize=24)
plt.grid()
plt.savefig(f'images/Q3/Reference Tracking performance.png')
plt.close()


err = np.abs(sys_response[:, 0] - sys_response[:, 2])

plt.figure(figsize=(9, 6))
plt.xlabel("t [sec]", fontsize=20)
plt.ylabel("Error", fontsize=20)
plt.plot(t, err, label=r"$E = |x(t)-x_r(t)|$", linewidth=2)
plt.legend(prop={'size': 17}, framealpha=1)
plt.grid()
plt.savefig(f'images/Q3/Error.png')
plt.close()


plt.figure(figsize=(14, 6))
plt.xlabel("t [sec]", fontsize=20)
plt.ylabel("Input", fontsize=20)
plt.plot(time_lst, input_lst, label=r"$u(t)$", linewidth=2)
plt.legend(prop={'size': 17}, framealpha=1)
plt.grid()
plt.savefig(f'images/Q3/Input.png')
plt.close()

# generate_plot(t=t, xm=sys_response[:, 2], x=sys_response[:, 0], ts=time_lst, u=np.array(input_lst))
