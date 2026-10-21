"""Writes paper/tables/sim-turn.tex and sim-sensing.tex from results/simulation.json."""

from decimal import ROUND_HALF_UP, Decimal

import numpy as np

from . import store
from .predictions import first_order_drift, first_order_slip, slip

HEADER = '% Written by python -m simulation tables from results/simulation.json; do not edit by hand.'

TURN = (
    HEADER
    + r"""
\begin{table}[htbp]
    \centering
    \renewcommand{\arraystretch}{1.1}
    \begin{tabular}{cccccccc}
        \hline
        $\rho$ (mm) & $\ell$ (mm) & $\rho/\ell$ & \multicolumn{3}{c}{Under-rotation ($^{\circ}$)} & \multicolumn{2}{c}{Grasp drift (mm)} \\
        & & & first order & integrated & simulated & first order & simulated \\
        \hline
ROWS
        \hline
    \end{tabular}
    \caption{Turn slip in simulation for a $90^{\circ}$ turn about one free rod. First order: the estimates $\Delta\rho^2/\ell^2$ and $|\Delta|\rho^2/\ell$ of Proposition~\ref{prop:slip}; integrated: the same limit-surface model without linearization. Drift is measured along the contact normal. The row $\ell=20$ mm is the corner contact.}
    \label{tab:sim_turn}
\end{table}
"""
)

SENSING = (
    HEADER
    + r"""
\begin{table}[htbp]
    \centering
    \renewcommand{\arraystretch}{1.1}
    \begin{tabular}{lcccc}
        \hline
        Strategy & Mean ($^{\circ}$) & SD ($^{\circ}$) & RMS ($^{\circ}$) & Slope \\
        \hline
NOMINAL
        \hline
CALIBRATION
        \hline
    \end{tabular}
    \caption{Final in-grasp angle error of the turn under grasp uncertainty, 50 trials per row with the same perturbations in each block. Slope: regression of the error on the injected initial orientation error.}
    \label{tab:sim_sensing}
\end{table}
"""
)

STRATEGIES = {'A': 'Open loop', 'B': 'Touch', 'C': 'Touch and direction'}


def fixed(x, digits):
    """x with the given number of decimals, rounding halves up."""
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-digits), ROUND_HALF_UP))


def row(cells):
    return '        ' + ' & '.join(cells) + r' \\'


def turn_rows(runs):
    """E3 cases by increasing rho/l, one row per (rho, l)."""
    cases = {}
    for r in sorted(runs.values(), key=lambda r: r['name']):
        if r['experiment'] == 'E3' and not r['parent']:
            cases.setdefault((r['rho_mm'], r['lever_mm']), r)
    rows = []
    for (rho, l), r in sorted(cases.items(), key=lambda c: (c[0][0] / c[0][1], c[0][0])):
        rows.append(
            row(
                [
                    f'{rho:g}',
                    f'{l:g}',
                    fixed(rho / l, 3),
                    fixed(first_order_slip(rho, l), 2),
                    fixed(slip(rho, l), 2),
                    fixed(-r['signed_angle_error_deg'], 2),
                    fixed(first_order_drift(rho, l), 3),
                    fixed(r['turn_grasp_drift_body_mm'][1], 3),
                ]
            )
        )
    return '\n'.join(rows)


def sensing_rows(runs, calibration):
    """E7 angle error for each strategy, without or with rod calibration errors."""
    rows = []
    for strategy, label in STRATEGIES.items():
        block = [
            r
            for r in runs.values()
            if r['experiment'] == 'E7'
            and not r['parent']
            and r['strategy'] == strategy
            and bool(np.any(r['rod_error'])) == calibration
        ]
        error = np.array([r['signed_angle_error_deg'] for r in block])
        slope = np.polyfit([r['initial_angle_deg'] for r in block], error, 1)[0]
        rows.append(
            row(
                [
                    label + (', calibration error' if calibration else ''),
                    f'${fixed(error.mean(), 2)}$',
                    fixed(error.std(ddof=1), 2),
                    fixed(np.sqrt(np.mean(error**2)), 2),
                    '--' if strategy == 'A' else fixed(slope, 2),
                ]
            )
        )
    return '\n'.join(rows)


def tables_main():
    runs = store.load_runs()
    store.save_text(store.TABLES / 'sim-turn.tex', TURN.replace('ROWS', turn_rows(runs)))
    store.save_text(
        store.TABLES / 'sim-sensing.tex',
        SENSING.replace('NOMINAL', sensing_rows(runs, False)).replace('CALIBRATION', sensing_rows(runs, True)),
    )
