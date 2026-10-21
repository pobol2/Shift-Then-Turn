"""Writes paper/figures/sim-shift.tex and sim-turn.tex from results/simulation.json.

Sign convention: the paper's e = d0_hat x (g - x). The simulator offsets the
gripper by -e_sim w, so e = -e_sim. Edit this script, not the generated files.
"""

import numpy as np

from . import store
from .predictions import K0, relax, slip

HEADER = '% Written by python -m simulation figures from results/simulation.json; do not edit by hand.'


class Axes:
    def __init__(self, x0, y0, w, h, xlim, ylim):
        self.x0, self.y0, self.w, self.h, self.xlim, self.ylim = x0, y0, w, h, xlim, ylim
        self.out = []

    def p(self, x, y):
        X = self.x0 + (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.w
        Y = self.y0 + (y - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * self.h
        return f'({X:.3f},{Y:.3f})'

    def frame(self, xticks, yticks, xlabel, ylabel, xfmt, yfmt, zero=False):
        o = self.out
        for t in yticks:
            o.append(f'\\draw[black!12,line width=.3pt] {self.p(self.xlim[0], t)} -- {self.p(self.xlim[1], t)};')
        if zero:
            o.append(f'\\draw[black!35,line width=.4pt] {self.p(self.xlim[0], 0)} -- {self.p(self.xlim[1], 0)};')
            o.append(f'\\draw[black!35,line width=.4pt] {self.p(0, self.ylim[0])} -- {self.p(0, self.ylim[1])};')
        o.append(
            f'\\draw[black!60,line width=.4pt] {self.p(self.xlim[0], self.ylim[0])} rectangle {self.p(self.xlim[1], self.ylim[1])};'
        )
        for t in xticks:
            o.append(
                f'\\draw[black!60,line width=.4pt] {self.p(t, self.ylim[0])} -- ++(0,.08) '
                f'node[below=3pt,black!80] {{${xfmt(t)}$}};'
            )
        for t in yticks:
            o.append(
                f'\\draw[black!60,line width=.4pt] {self.p(self.xlim[0], t)} -- ++(.08,0) '
                f'node[left=3pt,black!80] {{${yfmt(t)}$}};'
            )
        o.append(f'\\node[below=14pt] at {self.p(sum(self.xlim) / 2, self.ylim[0])} {{{xlabel}}};')
        o.append(f'\\node[rotate=90,above=18pt] at {self.p(self.xlim[0], sum(self.ylim) / 2)} {{{ylabel}}};')

    def line(self, xs, ys, style):
        pts = ' -- '.join(self.p(x, y) for x, y in zip(xs, ys))
        self.out.append(f'\\draw[{style}] {pts};')

    def marks(self, xs, ys, shape, color):
        for x, y in zip(xs, ys):
            if shape == 'o':
                self.out.append(
                    f'\\filldraw[fill={color},draw=white,line width=.4pt] {self.p(x, y)} circle[radius=2.1pt];'
                )
            else:
                self.out.append(
                    f'\\filldraw[fill={color},draw=white,line width=.4pt] {self.p(x, y)} ++(-1.9pt,-1.9pt) rectangle ++(3.8pt,3.8pt);'
                )

    def text(self, x, y, s, opts):
        self.out.append(f'\\node[{opts}] at {self.p(x, y)} {{{s}}};')


def figure(body, bbox, caption, label, header):
    return '\n'.join(
        [
            '\\begin{figure}[htbp]',
            '\\centering',
            header,
            '\\begin{tikzpicture}[paperfigure]',
            f'\\path[use as bounding box] {bbox};',
            *body,
            '\\end{tikzpicture}',
            f'\\caption{{{caption}}}',
            f'\\label{{{label}}}',
            '\\end{figure}',
            '',
        ]
    )


def figures_main():
    runs = store.load_runs()

    def primary(experiment):
        return [r for r in runs.values() if r['experiment'] == experiment and not r['parent']]

    blue, orange = 'objectblue', 'forceorange'

    # ---- Shift figure -------------------------------------------------------
    A = Axes(1.3, 1.0, 5.8, 4.6, (-1.15, 1.15), (-5.6, 5.6))
    A.frame(
        [-1, -0.5, 0, 0.5, 1],
        [-4, -2, 0, 2, 4],
        r'offset $e$ (mm)',
        r'endpoint tilt ($^{\circ}$)',
        lambda t: f'{t:g}',
        lambda t: f'{t:g}',
        zero=True,
    )
    lams = [(37.10795, blue, 'o'), (30.0, orange, 's')]
    for lam, col, shp in lams:
        K = K0 - lam
        ee = np.array([-1.1, 1.1])
        A.line(ee, np.degrees(ee / K), f'{col},densely dashed,line width=.6pt')
        frac = relax(lam, 1.0)[1][-1]
        A.line(ee, ee * frac, f'{col},line width=.8pt')
        rows = [r for r in primary('E2') if np.isclose(r['lambda_mm'], lam, atol=1e-3)]
        A.marks([-r['e_mm'] for r in rows], [r['world_shift_angle_deg'] for r in rows], shp, col)
    A.text(-1.15, 5.6, r'(a)', 'anchor=south west,font=\\small\\bfseries')
    A.marks([-1.0], [4.6], 'o', blue)
    A.text(-0.92, 4.6, r'$\lambda=37.1$ mm', 'anchor=west,font=\\scriptsize')
    A.marks([-1.0], [3.7], 's', orange)
    A.text(-0.92, 3.7, r'$\lambda=30$ mm', 'anchor=west,font=\\scriptsize')

    B = Axes(9.2, 1.0, 5.8, 4.6, (0, 21), (0, 6.0))
    B.frame(
        [0, 5, 10, 15, 20],
        [0, 1, 2, 3, 4, 5],
        r'grasp travel $s$ (mm)',
        r'tilt ($^{\circ}$)',
        lambda t: f'{t:g}',
        lambda t: f'{t:g}',
    )
    for lam, col, shp in lams:
        K = K0 - lam
        B.line([0, 21], [np.degrees(1 / K)] * 2, f'{col},densely dashed,line width=.6pt')
        s, th = relax(lam, 1.0)
        B.line(s[::8], th[::8], f'{col},line width=.8pt')
        r = next(r for r in primary('E2') if np.isclose(r['lambda_mm'], lam, atol=1e-3) and r['e_mm'] == -1)
        trav, tilt = np.array(r['shift_travel_mm']), np.array(r['shift_tilt_deg'])
        pick = [np.argmin(abs(trav - t)) for t in np.arange(1, 21, 2)]
        B.marks(trav[pick], tilt[pick], shp, col)
    B.text(0, 6.0, r'(b)', 'anchor=south west,font=\\small\\bfseries')
    B.text(
        20.6,
        np.degrees(1 / (K0 - 37.10795)) + 0.12,
        r'$e/K_{\mathrm{sh}}$, $\lambda=37.1$',
        f'anchor=south east,{blue}!90!black,font=\\scriptsize',
    )
    B.text(
        20.6,
        np.degrees(1 / (K0 - 30)) + 0.12,
        r'$e/K_{\mathrm{sh}}$, $\lambda=30$',
        f'anchor=south east,{orange}!90!black,font=\\scriptsize',
    )
    # legend for line styles
    lg = [
        '\\filldraw[fill=black!60,draw=white] (3.2,-0.35) circle[radius=2.1pt]; \\node[anchor=west] at (3.35,-0.35) {simulation};',
        '\\draw[black!70,line width=.8pt] (5.6,-0.35) -- (6.1,-0.35); \\node[anchor=west] at (6.15,-0.35) {prediction, Eqs.~(\\ref{eq:relax})--(\\ref{eq:relax_end})};',
        '\\draw[black!70,densely dashed,line width=.6pt] (10.4,-0.35) -- (10.9,-0.35); \\node[anchor=west] at (10.95,-0.35) {asymptote $e/K_{\\mathrm{sh}}$};',
    ]
    cap = (
        r'Shift tilt in simulation for the placements $\lambda=37.1$ and 30 mm. (a)~Tilt at the end of the '
        r'shift against the grasp offset $e$, with the finite-shift prediction \eqref{eq:relax_end} (solid) and '
        r'the asymptote $e/K_{\mathrm{sh}}$ of Proposition~\ref{prop:stiffness} (dashed). (b)~Tilt along the shift '
        r'for $e=1$ mm, with the solution of \eqref{eq:relax}: the plate approaches $e/K_{\mathrm{sh}}$ at the '
        r'predicted rate and does not reach it.'
    )
    store.save_text(
        store.FIGURES / 'sim-shift.tex',
        figure(A.out + B.out + lg, '(0,-0.6) rectangle (15.6,6.0)', cap, 'fig:sim_shift', HEADER),
    )

    # ---- Turn figure --------------------------------------------------------
    T = Axes(1.3, 1.0, 9.0, 5.4, (0, 0.1), (0, 9.4))
    T.frame(
        [0, 0.02, 0.04, 0.06, 0.08, 0.1],
        [0, 2, 4, 6, 8],
        r'$\rho^2/\ell^2$',
        r'under-rotation ($^{\circ}$)',
        lambda t: f'{t:g}',
        lambda t: f'{t:g}',
    )
    T.line([0, 0.1], [0, 9], 'black!75,line width=.6pt')
    rr = np.linspace(0, 5, 26)
    T.line((rr / 16) ** 2, [slip(r, 16) for r in rr], f'{blue},line width=.8pt')
    ll = np.linspace(7.5, 20, 26)
    T.line((2 / ll) ** 2, [slip(2, l) for l in ll], f'{orange},line width=.8pt')
    for prefix, shp, col in [('E3_l', 's', orange), ('E3_rho', 'o', blue)]:
        rows = [r for r in primary('E3') if r['name'].startswith(prefix)]
        T.marks(
            [(r['rho_mm'] / r['lever_mm']) ** 2 for r in rows], [-r['signed_angle_error_deg'] for r in rows], shp, col
        )
    T.text(0.068, 6.9, r'first order, $\Delta\rho^2/\ell^2$', 'anchor=south east,black!80,font=\\scriptsize')
    T.marks([0.005], [8.6], 'o', blue)
    T.text(0.0075, 8.6, r'$\rho$ sweep, $\ell=16$ mm', 'anchor=west,font=\\scriptsize')
    T.marks([0.005], [7.8], 's', orange)
    T.text(0.0075, 7.8, r'$\ell$ sweep, $\rho=2$ mm', 'anchor=west,font=\\scriptsize')
    T.text(
        0.099,
        0.35,
        r'markers: simulation; coloured lines: integrated model',
        'anchor=south east,black!70,font=\\scriptsize',
    )
    cap = (
        r'Turn slip in simulation for a $90^{\circ}$ turn about one free rod: the $\rho$ sweep at $\ell=16$ mm '
        r'(circles) and the $\ell$ sweep at $\rho=2$ mm (squares), with the first-order estimate of '
        r'Proposition~\ref{prop:slip} (black) and the same limit-surface model integrated without '
        r'linearization for each sweep (coloured).'
    )
    store.save_text(
        store.FIGURES / 'sim-turn.tex', figure(T.out, '(-0.1,0.0) rectangle (10.6,6.6)', cap, 'fig:sim_turn', HEADER)
    )
