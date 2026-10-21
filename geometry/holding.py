"""fig:holding, paper/figures/holding.tex: one rod, two rods on one edge, and two rods across a vertex.

Same pentagon and shift as fig:tworod.
"""

import numpy as np

from .construction import design, poly_info

P = np.array([[-34, -17], [30, -21], [46, 2], [18, 22], [-40, 13]], float)
R = 5


def figure():
    T, N = poly_info(P)

    def f(v):
        return f'({v[0]:.2f},{v[1]:.2f})'

    def poly(Pts, style):
        return f'\\draw[{style}] ' + ' -- '.join(f(p) for p in Pts) + ' -- cycle;'

    def rod(q):
        return f'  \\draw[feature] {f(q)} circle[radius={R}]; \\fill {f(q)} circle(.6pt);'

    S = 0.042
    e = 3
    t = T[e]
    n = N[e]
    foot = P[e] + ((-P[e]) @ t) * t
    # (a) one rod
    A = [
        poly(P, 'object'),
        f'  \\draw[guide] {f(-18 * n)} -- {f(foot + 16 * n)};',
        rod(foot + R * n),
        f'  \\draw[force] {f(foot + 19 * n)} -- {f(foot + 0.5 * n)};',
    ]
    q = foot + R * n
    ang = np.degrees(np.arctan2(n[1], n[0]))
    A.append(
        f'  \\draw[->,black!70,line width=.5pt] ([shift={{({ang - 55:.1f}:10)}}]{q[0]:.2f},{q[1]:.2f}) arc[start angle={ang - 55:.1f},end angle={ang + 55:.1f},radius=10];'
    )
    A.append('  \\gripper{0}{0}{}')
    A.append(f'  \\draw[motion,line width=.9pt] {f(5 * n)} -- {f(12 * n)};')
    ptsA = list(P) + [foot + 19 * n, q + np.array([10, 10]), -18 * n]
    # (b) two rods on one edge
    B = [poly(P, 'object'), f'  \\draw[guide] {f(-18 * n)} -- {f(foot + 16 * n)};']
    for u in (-16, 16):
        p = foot + u * t
        B += [rod(p + R * n), f'  \\draw[force] {f(p + 19 * n)} -- {f(p + 0.5 * n)};']
    B.append('  \\gripper{0}{0}{}')
    B.append(f'  \\draw[motion,line width=.9pt] {f(5 * n)} -- {f(12 * n)};')
    ptsB = list(P) + [foot + 16 * t + 19 * n, foot - 16 * t + 19 * n, -18 * n]
    # (c) trailing vertex, Fig. 1 shift
    o = design(P, R, np.radians(80), np.array([6, 18.0]), 34)
    x, dh, pa, pb, qa, qb, na, nb = o['x'], o['dh'], o['pa'], o['pb'], o['qa'], o['qb'], o['na'], o['nb']
    a1 = np.degrees(np.arctan2(-na[1], -na[0]))
    a2 = np.degrees(np.arctan2(-nb[1], -nb[0]))
    if (a2 - a1) % 360 > 180:
        a1, a2 = a2, a1
    a2 = a1 + ((a2 - a1) % 360)
    C = [
        poly(P, 'object'),
        f'  \\fill[forceorange!18] {f(x)} -- ([shift={{({a1:.1f}:12)}}]{x[0]:.2f},{x[1]:.2f}) arc[start angle={a1:.1f},end angle={a2:.1f},radius=12] -- cycle;',
        f'  \\draw[guide] {f(x - 14 * dh)} -- {f(30 * dh)};',
    ]
    for p, nn, qq in ((pa, na, qa), (pb, nb, qb)):
        C += [
            f'  \\draw[guide] {f(p + 2 * nn)} -- {f(x - 3 * nn)};',
            rod(qq),
            f'  \\draw[force] {f(p + 17 * nn)} -- {f(p + 0.5 * nn)};',
        ]
    C.append(
        f'  \\fill {f(x)} circle(1.2pt); \\node[anchor=south west,inner sep=1pt] at {f(x + np.array([0.8, 0.8]))} {{$\\mathbf x$}};'
    )
    C.append('  \\gripper{0}{0}{}')
    C.append(f'  \\draw[motion,line width=.9pt] {f(-4 * dh)} -- {f(-13 * dh)};')
    ptsC = list(P) + [pa + 17 * na, pb + 17 * nb, x - 14 * dh, 30 * dh]
    gr = [np.array([-14, -22]), np.array([6, 5])]
    out = [
        r"""\begin{figure}[htbp]
\centering
% Computed with the hold condition and the trailing-vertex construction on the pentagon of Fig. 1 (mm, 1 mm = 0.042 cm).
% Written by python -m geometry; do not edit coordinates by hand.
\begin{tikzpicture}[paperfigure,
  gripbody/.style={fill=black!55,draw=black!80,line width=.4pt},
  grippad/.style={fill=black!75,draw=black}]
\newcommand{\gripper}[3]{%
  \begin{scope}[shift={(#1,#2)},#3]
    \fill[gripbody,rotate=30] (-3,0) rectangle (3,-24);
    \fill[grippad] (0,0) circle[radius=4];
    \fill[white] (0,0) circle[radius=1];
  \end{scope}}"""
    ]
    tops = []
    body = []
    specs = (('(a) One rod', A, ptsA), ('(b) Same edge', B, ptsB), ('(c) Across a vertex', C, ptsC))
    boxes = [(np.array(p + gr).min(0), np.array(p + gr).max(0)) for _, _, p in specs]
    base = 3.0 - max(S * (hi[1] - (lo[1] - 1)) for lo, hi in boxes)  # common lower edge
    for k, ((title, cont, _), (lo, hi)) in enumerate(zip(specs, boxes)):
        cx = 2.6 + 5.2 * k
        sx = cx - S * (lo[0] + hi[0]) / 2
        sy = base - S * (lo[1] - 1)
        body.append(f'\\node[paneltitle] at ({5.2 * k:.1f},3.35) {{{title}}};')
        body.append(f'\\begin{{scope}}[shift={{({sx:.3f},{sy:.3f})}},x={S}cm,y={S}cm]')
        body += cont
        body.append('\\end{scope}')
        tops.append(base - 0.2)
    ymin = min(tops)
    out.append(f'\\path[use as bounding box] (0,{ymin:.2f}) rectangle (15.6,3.55);')
    out += body
    out.append(
        f'\\draw[black!15] (5.05,{ymin:.2f}) -- (5.05,3.45); \\draw[black!15] (10.25,{ymin:.2f}) -- (10.25,3.45);'
    )
    out.append(r"""\end{tikzpicture}
\caption{Holding the plate during a shift. Orange arrows: rod forces along their push lines; teal arrows: gripper motion, which drags the plate through the grasp; dashed lines: grasp line and push lines. (a)~One rod holds only when its push line coincides with the grasp line, allowing finitely many shift directions. The hold has no rotational stiffness: the plate can turn freely about the rod centre (curved arrow; Section~\ref{sec:minimal}). (b)~For a shift straight into an edge, two rods on that edge straddle the grasp line. (c)~Across the trailing vertex, the two push lines meet at $\mathbf x$ on the grasp line, as placed by \eqref{eq:v_construction}. Their forces span the shaded cone, and the hold has positive second-order stiffness $K_{\mathrm{sh}}>0$ (Proposition~\ref{prop:stiffness}).}
\label{fig:holding}
\end{figure}""")
    return '\n'.join(out)
