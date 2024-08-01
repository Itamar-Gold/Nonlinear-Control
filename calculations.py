

def error_calc(x, xr, x_dt, xr_dt):
    """

    :param x:
    :param xr:
    :param x_dt:
    :param xr_dt:
    :return:
    """
    err = x - xr
    err_d = x_dt - xr_dt
    return err, err_d


def controller(x, xm, x_dot, xm_dot, xm_ddot, m, lamda0, K, k1, k2, c1, c2, params):
    """

    :param x:
    :param xm:
    :param x_dot:
    :param xm_dot:
    :param xm_ddot:
    :param m:
    :param lamda0:
    :param K:
    :param k1:
    :param k2:
    :param c1:
    :param c2:
    :param params:
    :return:
    """
    e, e_dot = error_calc(x, xm, x_dot, xm_dot)
    xr_ddot = xm_ddot - lamda0 * e_dot
    z = e_dot + lamda0 * e
    u = m * xr_ddot - K * z + k1 * x + k2 * (x ** 3) + c1 * x_dot + c2 * (x_dot ** 2)
    if params == "known":
        return u
    return u, xr_ddot, z


def euler_integration(y, y_dt, dt):
    """

    :param y:
    :param y_dot:
    :param dt:
    :return:
    """
    return y + y_dt * dt


def adaptation_law(gamma, z, xr_ddot, x, x_dot):
    m_hat_dot = - gamma * z * xr_ddot
    k1_hat_dot = - gamma * z * x
    k2_hat_dot = - gamma * z * (x ** 3)
    c1_hat_dot = - gamma * z * x_dot
    c2_hat_dot = - gamma * z * (x_dot ** 2)
    return m_hat_dot, k1_hat_dot, k2_hat_dot, c1_hat_dot, c2_hat_dot
