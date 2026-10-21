"""fig:approach, paper/figures/approach.tex: collision-free approach to one rod and to the installed pair.

Nonconvex part with a wide pocket (reachable) and a narrow slot (closed by the offset).
"""

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon

R = 5


def figure():
    P = Polygon(
        [
            (-40, -15),
            (-12, -15),
            (-12, -7),
            (8, -7),
            (8, -15),
            (40, -15),
            (40, 15),
            (14, 15),
            (14, 5),
            (8, 5),
            (8, 15),
            (-40, 15),
        ]
    )
    G = P.buffer(R, quad_segs=32)
    assert len(G.interiors) == 0

    def f(v):
        return f'({v[0]:.2f},{v[1]:.2f})'

    def path(poly, style):
        return f'\\draw[{style}] ' + ' -- '.join(f(p) for p in list(poly.exterior.coords)[:-1]) + ' -- cycle;'

    qa = np.array([-36.0, 20])
    qb = np.array([-45.0, 9])
    for q in (qa, qb):
        assert abs(P.distance(Point(q)) - R) < 1e-9
    sh = -(qb - qa)
    G2 = affinity.translate(G, *sh)
    assert G2.exterior.distance(Point(qa)) < 1e-6 and G.exterior.distance(Point(qa)) < 1e-6
    # (a) one rod: planned path into the pocket
    pocket_q = np.array([-2.0, -12])
    assert abs(P.distance(Point(pocket_q)) - R) < 1e-9
    slot_q = np.array([11.0, 19.0])
    assert abs(P.distance(Point(slot_q)) - R) < 1e-9
    waypts = [np.array([46.0, -38]), np.array([14.0, -36]), np.array([-2.0, -30]), np.array([-2.0, -21])]
    assert LineString(waypts).distance(P) > R + 0.5
    A = [path(G, 'cobs'), path(P, 'object')]
    for q in ((-45, 0), (45, 4), (25, 20)):
        assert abs(P.distance(Point(q)) - R) < 1e-9, q
        A.append(f'\\draw[rodghost] {f(q)} circle[radius=5];')
    A.append(f'\\draw[rodghost] {f(slot_q)} circle[radius=5];')
    A.append('\\node at (-22,2) {$P$};')
    A.append('\\node[anchor=west] at (46,-12) {$P\\oplus B_R$};')
    A.append('\\draw[motion,line width=.9pt] ' + ' -- '.join(f(p) for p in waypts) + ';')
    A.append(f'\\fill[motionteal] {f(waypts[0])} circle(1.3pt);')
    A.append(f'\\draw[force] {f(waypts[-1])} -- {f(pocket_q - [0, 5.6])};')
    A.append(
        f'\\draw[draw=forceorange,fill=forceorange!12] {f(pocket_q)} circle[radius=5]; \\fill[forceorange] {f(pocket_q)} circle(1.3pt);'
    )
    A.append('\\node[anchor=east] at (-9,-27) {reachable};')
    A.append('\\node[anchor=south] at (11,25) {closed};')
    B = [
        path(G, 'cobs'),
        path(G2, 'cobs,fill=black!7'),
        path(G, 'draw=black!55,line width=.5pt,fill=none'),
        path(P, 'object'),
    ]
    B.append('\\node at (-16,2) {$P$};')
    s0 = np.array([-62.0, 40])
    B.append(f'\\draw[draw=forceorange] {f(s0)} circle[radius=5]; \\draw[rodghost] {f(s0 + qb - qa)} circle[radius=5];')
    B.append(f'\\draw[black!40,line width=.4pt] {f(s0)} -- {f(s0 + qb - qa)};')
    wp = [s0, np.array([-53.0, 33]), np.array([-43.0, 27])]
    B.append('\\draw[motion,line width=.9pt] ' + ' -- '.join(f(p) for p in wp) + ';')
    B.append(f'\\draw[force] {f(wp[-1])} -- {f(qa + np.array([-0.4, 0.6]))};')
    B.append(
        f'\\draw[draw=forceorange,fill=forceorange!12] {f(qa)} circle[radius=5]; \\fill[forceorange] {f(qa)} circle(1.3pt);'
    )
    B.append(f'\\draw[feature] {f(qb)} circle[radius=5]; \\fill {f(qb)} circle(.6pt);')
    B.append(f'\\draw[black!40,line width=.4pt] {f(qa)} -- {f(qb)};')
    B.append(
        '\\node[anchor=south west,inner sep=1pt] at (-34,22) {$\\mathbf q_a$}; \\node[anchor=east] at (-50,9) {$\\mathbf q_b$};'
    )
    out = [
        r"""\begin{figure}[htbp]
\centering
% Gripper frame, mm, 1 mm = 0.05 cm. Offsets computed exactly (shapely buffer). Written by python -m geometry; do not edit coordinates by hand.
\begin{tikzpicture}[paperfigure,
  cobs/.style={fill=black!12,draw=black!55,line width=.5pt},
  rodghost/.style={draw=black!40,densely dashed,line width=.4pt}]
\path[use as bounding box] (0,-0.1) rectangle (15.6,5.05);
\node[paneltitle] at (0,4.85) {(a) One rod};
\begin{scope}[shift={(3.9,2.25)},x=.05cm,y=.05cm]"""
    ]
    out += A
    out.append(r"""\end{scope}
\node[paneltitle] at (8.0,4.85) {(b) Two rods};
\begin{scope}[shift={(12.0,2.05)},x=.05cm,y=.05cm]""")
    out += B
    out.append(r"""\end{scope}
\draw[black!15] (7.8,-0.05) -- (7.8,4.95);
\end{tikzpicture}
\caption{Collision-free approach in the object frame: the object stays fixed and the rods move (adapted from our earlier C-space construction). (a)~Plan the rod centre outside $P\oplus B_R$, the outline grown by a disk of radius $R$. The wide pocket stays reachable, while the slot narrower than the rod closes; dashed circles show contact configurations. (b)~With the wrist fixed, the rod pair translates rigidly together. Planning the centre of rod~$a$ therefore requires avoiding both offset regions: $P\oplus B_R$ and its upper-right copy $(P\oplus B_R)-(\mathbf q_b-\mathbf q_a)$, as in \eqref{eq:cobs}. The engagement point lies where their boundaries meet. Teal paths end with a straight approach (orange) to contact, closing both gaps together in (b); the gripper moves opposite to the rod-centre path.}
\label{fig:approach}
\end{figure}""")
    return '\n'.join(out)
