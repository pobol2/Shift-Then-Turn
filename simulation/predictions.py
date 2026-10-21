"""Predictions of the paper's equations for the simulated plate, in mm and degrees.

- shift tilt: d theta/ds = (e - K theta)/(L^2 + rho^2), L = lambda - s  (Eq. relax)
- turn slip: first-order Delta rho^2/l^2 and drift |Delta| rho^2/l (Prop. turn slip),
  and the same ellipsoidal limit-surface model integrated without linearization.
"""

import numpy as np
from scipy.optimize import brentq

D0 = np.array([20.0, -5.0])
S = float(np.linalg.norm(D0))
RHO = 2.0
K0 = 48.507125  # R(alpha_a + alpha_b) - d0_hat^T P_i for the 80 x 30 mm plate


def Q(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def relax(lam, e, n=400):
    """Tilt (deg) along grasp travel s from Eq. (relax), starting untilted."""
    K = K0 - lam
    s = np.linspace(0, S, n)
    th = np.zeros(n)
    for k in range(1, n):
        ds = s[k] - s[k - 1]
        L = lam - 0.5 * (s[k] + s[k - 1])
        th[k] = th[k - 1] + ds * (e - K * th[k - 1]) / (L**2 + RHO**2)
    return s, np.degrees(th)


def slip(rho, l, Delta=np.pi / 2, n=2000):
    """Under-rotation (deg) of a 90 deg turn about one free rod on the top edge,
    ellipsoidal limit surface integrated without linearization."""
    if rho == 0:
        return 0.0
    qw = np.array([-20.0 - l, 20.0])
    r0 = np.array([-20.0, 5.0])
    gy = 5.0
    th = 0.0
    for k in range(1, n + 1):
        r = qw + Q(k / n * Delta) @ (r0 - qw)
        th0, gy0 = th, gy
        for _ in range(3):
            ell = -(Q(th).T @ (qw - r))[0]
            th = brentq(
                lambda t: (Q(t).T @ (qw - r))[1] + gy0 + rho**2 / ell * (t - th0) - 20,  # noqa: B023 (called within this iteration)
                th0 - 0.2,
                th0 + 0.2,
            )
        gy = gy0 + rho**2 / ell * (th - th0)
    return np.degrees(Delta - th)


def first_order_slip(rho, l, Delta=np.pi / 2):
    """Under-rotation (deg) Delta rho^2/l^2 of Prop. turn slip."""
    return np.degrees(Delta) * rho**2 / l**2


def first_order_drift(rho, l, Delta=np.pi / 2):
    """Grasp drift (mm) |Delta| rho^2/l along the contact normal, Prop. turn slip."""
    return abs(Delta) * rho**2 / l
