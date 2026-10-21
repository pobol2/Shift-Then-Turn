"""fig:tworod, paper/figures/tworod-example.tex: engage, shift, turn and result for a convex pentagon."""

import numpy as np

from simulation.core import ROD_RADIUS_MM as R

from .construction import Q, design
from .examples import DELTA as Delta
from .examples import LAMBDA as lam
from .examples import D as d
from .examples import P


def figure():
    o = design(P, R, Delta, d, lam)
    d0 = o['d0']
    dh = o['dh']
    qa, qb = o['qa'], o['qb']
    pa, pb = o['pa'], o['pb']
    na, nb = o['na'], o['nb']
    x = o['x']

    def f(v):
        return f'({v[0]:.2f},{v[1]:.2f})'

    def poly(Pts, style):
        return f'\\draw[{style}] ' + ' -- '.join(f(p) for p in Pts) + ' -- cycle;'

    def rot(Pts, c, a):
        return np.array([c + Q(a) @ (p - c) for p in Pts])

    def rods():
        return '\n'.join(
            [f'  \\draw[feature] {f(q)} circle[radius={R:g}]; \\fill {f(q)} circle(.6pt);' for q in (qa, qb)]
        )

    w = (na + nb) / np.linalg.norm(na + nb)
    eps = 7
    g1 = -d0  # post-shift grasp (world)
    # ---------- panel contents (mm, world = initial gripper frame)
    A = []
    A.append(poly(P - eps * w, 'ghostobj'))
    A.append(poly(P, 'object'))
    A.append(rods())
    s = np.array([30, -26.0])
    A.append(f'  \\draw[motion,line width=.9pt] {f(s)} -- {f(s + 12 * w)};')
    A.append('  \\gripper{0.00}{0.00}{}')
    A.append(f'  \\node[anchor=south] at {f(qa + np.array([0, 5.5]))} {{$a$}};')
    A.append(f'  \\node[anchor=east] at {f(qb - np.array([5.5, 0]))} {{$b$}};')
    B = []
    B.append(poly(P, 'object'))
    gl0 = x - 6 * dh
    gl1 = np.array([0, 0]) + 26 * dh
    B.append(f'  \\draw[guide] {f(gl0)} -- {f(gl1)};')
    B.append(
        f'  \\node[anchor=south west,inner sep=1pt,text=black!60] at {f(gl1 + np.array([-14, 1.5]))} {{grasp line}};'
    )
    B.append(rods())
    for p, n in ((pa, na), (pb, nb)):
        B.append(f'  \\draw[guide] {f(p)} -- {f(x)};')
        B.append(f'  \\draw[force] {f(p + 16 * n)} -- {f(p + 0.5 * n)};')
    B.append(
        f'  \\fill {f(x)} circle(1.2pt); \\node[anchor=north west,inner sep=1pt] at {f(x + np.array([0.8, -0.8]))} {{$\\mathbf x$}};'
    )
    B.append('  \\gripper{0.00}{0.00}{,opacity=.28}')
    B.append(f'  \\draw[motion,line width=1pt] {f(-3 * dh)} -- {f(g1 + 3 * dh)};')
    B.append(f'  \\node[anchor=south,inner sep=2pt] at {f(g1 * 0.5 + np.array([0, 2.0]))} {{$-\\mathbf d_0$}};')
    B.append(f'  \\gripper{{{g1[0]:.2f}}}{{{g1[1]:.2f}}}{{}}')
    C = []
    C.append(poly(P, 'ghostobj'))
    C.append(poly(rot(P, qa, Delta / 2), 'object,fill opacity=.45,draw opacity=.45'))
    C.append(poly(rot(P, qa, Delta), 'object'))
    C.append(rods())
    r = np.linalg.norm(g1 - qa)
    a0 = np.degrees(np.arctan2(*(g1 - qa)[::-1]))
    C.append(
        f'  \\draw[motion,line width=1pt] {f(g1)} arc[start angle={a0:.3f},end angle={a0 + np.degrees(Delta):.3f},radius={r:.3f}];'
    )
    for sfrac, op in ((0, ',opacity=.28'), (0.5, ',opacity=.28'), (1, '')):
        gp = qa + Q(sfrac * Delta) @ (g1 - qa)
        C.append(f'  \\gripper{{{gp[0]:.2f}}}{{{gp[1]:.2f}}}{{{op}}}')
    C.append(f'  \\node[anchor=east] at {f(qa - np.array([5.5, -3]))} {{$a$}};')
    C.append(f'  \\node[anchor=east] at {f(qb - np.array([5.5, 0]))} {{$b$}};')
    Pf = rot(P, qa, Delta)
    D = []
    D.append(poly(P, 'ghostobj'))
    Pr = np.array([Q(Delta) @ p + d for p in P])
    D.append(poly(Pr, 'object'))
    D.append(f'  \\fill[objectblue] {f(d)} circle(1.3pt);')
    D.append(f'  \\draw[motion,line width=.9pt] (0,0) -- {f(d * 0.92)};')
    D.append(f'  \\node[anchor=east,inner sep=2pt] at {f(d * 0.55)} {{$\\mathbf d$}};')
    D.append(f'  \\draw[motion,line width=.9pt] (26,-4) arc[start angle=-10,end angle={-10 + 80},radius=26];')
    D.append(f'  \\node[anchor=west] at (31,14) {{$\\Delta={80}^\\circ$}};')
    D.append('  \\gripper{0.00}{0.00}{}')

    # ---------- bounding boxes (mm)
    def bb(pts):
        pts = np.array(pts)
        return pts.min(0), pts.max(0)

    def gr(c):
        return [c + np.array([-14, -22]), c + np.array([6, 5])]  # gripper bar extent

    bA = bb(list(P - eps * w) + list(P) + [qa + [0, 10], qb - [16, 0], qb + [5, 5]] + gr(np.zeros(2)) + [s, s + 12 * w])
    bB = bb(list(P) + [pa + 16 * na, pb + 16 * nb, qa + [0, 6], qb - [6, 0], gl0, gl1] + gr(np.zeros(2)) + gr(g1))
    pts = list(P) + list(rot(P, qa, Delta / 2)) + list(Pf) + [qb - [18, 0], qa - [18, 0]]
    for sf in (0, 0.5, 1):
        pts += gr(qa + Q(sf * Delta) @ (g1 - qa))
    bC = bb(pts)
    bD = bb(list(P) + list(Pr) + gr(np.zeros(2)) + [np.array([45, 12])])
    S = 0.05

    def place(b, cx, ytop):
        # Center horizontally at cx (cm), with the top of the content at ytop (cm).
        lo, hi = b
        sx = cx - S * (lo[0] + hi[0]) / 2
        sy = ytop - S * hi[1]
        return sx, sy, lo[1] - 3

    hA = S * (bA[1][1] - bA[0][1]) + 0.75
    hB = S * (bB[1][1] - bB[0][1]) + 0.75
    top1 = 10.55
    div = top1 - max(hA, hB) - 0.05
    t2 = div - 0.25
    top2 = t2 - 0.3
    cells = {'A': (3.9, top1), 'B': (11.7, top1), 'C': (3.9, top2), 'D': (11.7, top2)}
    out = []
    ys = {}
    for key, title, body, b in (
        ('A', '(a) Engage', A, bA),
        ('B', '(b) Shift', B, bB),
        ('C', '(c) Turn', C, bC),
        ('D', '(d) Result', D, bD),
    ):
        cx, yt = cells[key]
        sx, sy, capy = place(b, cx, yt)
        ys[key] = sy + S * capy - 0.45
        tx = 0 if key in 'AC' else 8.0
        ty = 10.9 if key in 'AB' else t2
        out.append(f'\\node[paneltitle] at ({tx},{ty}) {{{title}}};')
        out.append(f'\\begin{{scope}}[shift={{({sx:.3f},{sy:.3f})}},x=0.05cm,y=0.05cm]')
        out += body
        out.append('\\end{scope}')
    ymin = min(ys.values()) - 0.1
    out.append(f'\\draw[black!15] (7.8,{ymin:.2f}) -- (7.8,11.0); \\draw[black!15] (0,{div:.2f}) -- (15.6,{div:.2f});')
    return '\n'.join(
        [
            r"""\begin{figure}[htbp]
\centering
% Computed from the shift and turn constructions for a convex pentagon (mm, 1 mm = 0.05 cm):
% Delta = 80 deg, d = (6,18) mm, R = 5 mm, lambda = 34 mm. Written by python -m geometry; do not edit coordinates by hand.
\begin{tikzpicture}[paperfigure,
  gripbody/.style={fill=black!55,draw=black!80,line width=.4pt},
  grippad/.style={fill=black!75,draw=black},
  ghostobj/.style={draw=objectblue!60,densely dashed,fill=none,line width=.5pt}]
\newcommand{\gripper}[3]{%
  \begin{scope}[shift={(#1,#2)},#3]
    \fill[gripbody,rotate=30] (-3,0) rectangle (3,-24);
    \fill[grippad] (0,0) circle[radius=4];
    \fill[white] (0,0) circle[radius=1];
  \end{scope}}""",
            f'\\path[use as bounding box] (0,{ymin:.2f}) rectangle (15.6,11.1);',
            '',
            *out,
            r"""\end{tikzpicture}
\caption{Changing the grasp with two rods, using the constructions of Sections~\ref{sec:method} and~\ref{sec:turn}. (a)~The gripper brings the plate against two installed rods that spin freely on bearings. Dark disk: gripper pad; bar: wrist, whose orientation stays fixed throughout. (b)~The rod forces (orange) act along push lines meeting at $\mathbf x$ on the grasp line: the plate stays still while the gripper slides by $-\mathbf d_0$. (c)~The gripper orbits rod~$a$; the plate pivots within the grip, turns by $\Delta$, and separates from rod~$b$. (d)~In the gripper frame, the plate has turned by $\Delta=80^\circ$ and shifted by the requested $\mathbf d=\mathbf Q(\Delta)\mathbf d_0$. Teal arrows: gripper motion in (a--c), relative object displacement and rotation in (d); dashed outlines: initial poses.}
\label{fig:tworod}
\end{figure}""",
        ]
    )
