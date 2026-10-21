# Saved results

[`simulation.json`](simulation.json) holds the 367 MuJoCo runs behind the
[paper](../paper/main.pdf)'s simulation tables, figures and quoted values. It is
written by `python -m simulation run` (see the [commands](../README.md#commands)),
which overwrites it; `tables` and `figures` read it and write into `paper/`.

## Format

`mujoco` is the engine version used; `runs` is a list with one object per run. Every
run has `name`, `experiment`, `status` (`complete` for all saved runs), `timestep`,
`speed_scale` and `parent`. A run with an empty `parent` is a case of the paper; a run
with a `parent` repeats that case with the time step halved (`_dt_half`) or quartered
(`_dt_quarter`), or at half speed (`_half_speed`), to check convergence. The other
fields are the inputs and measurements that the experiment uses, listed in `INPUTS` and
`MEASURED` in [`simulation/experiments.py`](../simulation/experiments.py). Lengths are
in millimetres, angles in degrees, forces in newtons and times in seconds, as the field
names state.

| Experiment | Runs | Paper (Section 8) |
| --- | --- | --- |
| `E1` | Complete nominal sequence, engagement position perturbed by up to 0.3 mm | Complete sequences |
| `creep` | 9 s hold, bounding the solver's creep | Model |
| `E2` | Shift alone with grasp offset $e\in\{0,\pm0.5,\pm1\}$ mm, at $\lambda=37.1$ and 30 mm; includes `shift_travel_mm` and `shift_tilt_deg` along the shift for $e=1$ mm | Shift, Figure `fig:sim_shift` |
| `E3` | Turn alone about one rod: $\rho$ sweep at $\ell=16$ mm, $\ell$ sweep at $\rho=2$ mm | Turn, Table `tab:sim_turn`, Figure `fig:sim_turn` |
| `E4` | Turn with rod friction, free or welded rod | Turn |
| `E5` | Shift into the top edge with one or two rods offset by 0.5 mm | Shift |
| `E6` | Complete sequence with re-engagement | Complete sequences |
| `E7` | Turn under grasp uncertainty: strategies `A` (open loop), `B` (touch), `C` (touch and direction), 50 paired trials each, without (`nominal`) or with (`mount`) rod calibration errors | Sensing at first contact, Table `tab:sim_sensing` |

`python -m unittest discover -s tests -t .` checks that the file holds one complete run
for each case and refinement defined in the code, and that the generated tables and
figures in `paper/` are what it produces.
