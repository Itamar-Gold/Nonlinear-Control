

def error_calc(x, xm, x_dot, xm_dot):
    return x - xm, x_dot - xm_dot


def controller(x, xm, x_dot, xm_dot, xm_ddot, m, lamda0, K, k1, k2, c1, c2, params):
    e, e_dot = error_calc(x, xm, x_dot, xm_dot)
    xr_ddot = xm_ddot - lamda0 * e_dot
    z = e_dot + lamda0 * e
    u = m * xr_ddot - K * z + k1 * x + k2 * (x ** 3) + c1 * x_dot + c2 * (x_dot ** 2)
    if params == "known":
        return u
    return u, xr_ddot, z


def euler_integration(y, y_dot, dt):
    return y + y_dot * dt

