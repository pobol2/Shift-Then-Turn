"""The paper's shift construction on a convex polygon, in mm: rod placement, stiffness and lever arms."""

import numpy as np

J = np.array([[0, -1], [1, 0]])


def Q(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def cr(a, b):
    return a[0] * b[1] - a[1] * b[0]


def poly_info(P):
    n = len(P)
    T = []
    N = []
    for i in range(n):
        e = P[(i + 1) % n] - P[i]
        t = e / np.linalg.norm(e)
        T.append(t)
        N.append(-J @ t)
    return np.array(T), np.array(N)


def seg_dist(q, a, b):
    ab = b - a
    u = np.clip((q - a) @ ab / (ab @ ab), 0, 1)
    return np.linalg.norm(q - (a + u * ab))


def inside(q, P):
    n = len(P)
    _, N = poly_info(P)
    return all(N[i] @ (q - P[i]) < 0 for i in range(n))


def dist(q, P):
    d = min(seg_dist(q, P[i], P[(i + 1) % len(P)]) for i in range(len(P)))
    return -d if inside(q, P) else d


def design(P, R, Delta, d, lam):
    T, N = poly_info(P)
    n = len(P)
    d0 = Q(Delta).T @ d
    dh = d0 / np.linalg.norm(d0)
    i = int(np.argmax([-d0 @ p for p in P]))
    ta, tb = T[i - 1], T[i]
    na, nb = N[i - 1], N[i]
    aa, ab = np.linalg.solve(np.column_stack([na, nb]), -dh)
    x = -lam * dh
    pa = P[i] + ((x - P[i]) @ ta) * ta
    pb = P[i] + ((x - P[i]) @ tb) * tb
    qa = pa + R * na
    qb = pb + R * nb
    out = dict(d0=d0, dh=dh, i=i, Pi=P[i], na=na, nb=nb, aa=aa, ab=ab, x=x, pa=pa, pb=pb, qa=qa, qb=qb)

    # admissibility
    def on_edge(p, a, b):
        u = (p - a) @ (b - a) / ((b - a) @ (b - a))
        return 0.02 < u < 0.98

    out['pa_on'] = on_edge(pa, P[i - 1], P[i])
    out['pb_on'] = on_edge(pb, P[i], P[(i + 1) % n])
    out['x_inside'] = inside(x, P)
    out['dist_a'] = dist(qa, P)
    out['dist_b'] = dist(qb, P)
    out['grasp_end_inside'] = dist(-d0, P)
    K0 = -dh @ P[i] + R * (aa + ab)
    out['K'] = K0 - lam
    out['D'] = np.linalg.norm(qa - qb)
    # post-shift lever arms (grasp at d0 in object frame)
    sig = np.sign(Delta)
    la = -sig * cr(pa + d0, na)
    lb = -sig * cr(pb + d0, nb)
    out['la'] = la
    out['lb'] = lb
    return out
