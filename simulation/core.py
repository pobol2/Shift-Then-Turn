"""Shared E1--E7 trial settings, first-touch commands and trial execution.

Distances are millimetres in trial metadata and SI in the physics engine.
Plant perturbations are removed before constructing the controller.
"""

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NOMINAL = np.array([[-36.0, 20.0], [-45.0, 9.0]])
D0 = np.array([20.0, -5.0])
WAPP = np.array([-1.0, 1.0]) / np.sqrt(2)


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


W = rotation(np.pi / 2) @ (D0 / np.linalg.norm(D0))


class PaperSensors:
    """Rod bearing forces in the world frame; no plate state is observed."""

    def __init__(self, exp, trial):
        self.exp = exp
        self.filter_time_s = trial.sensor_filter_s
        self.site_ids = [exp.model.site(f'rod_{n}_loadcell').id for n in ('lower', 'upper')]
        self.addresses = [exp.model.sensor(f'rod_{n}_load_force').adr[0] for n in ('lower', 'upper')]
        self.filtered = np.zeros((2, 2))
        self.norms = np.zeros(2)
        self.update()

    def update(self):
        dt = self.exp.model.opt.timestep
        alpha = dt / (self.filter_time_s + dt)
        forces = np.zeros((2, 2))
        for i, force_address in enumerate(self.addresses):
            frame = self.exp.data.site_xmat[self.site_ids[i]].reshape(3, 3)
            force = frame @ self.exp.data.sensordata[force_address : force_address + 3]
            forces[i] = force[:2]
        self.filtered += alpha * (forces - self.filtered)
        self.norms = np.linalg.norm(self.filtered, axis=1)


@dataclass
class Trial:
    name: str
    experiment: str
    rods: list = None
    active_rods: int = 2
    free: bool = True
    rod_mu: float = 0.0
    rho_mm: float = 2.0
    lever_mm: float = 16.0
    object_error: list = None
    grasp_error: list = None
    rod_error: list = None
    initial_angle_deg: float = 0.0
    strategy: str = 'A'
    seed: int = 7
    noise_deg: float = 1.0
    touch_threshold_N: float = 0.5
    speed_scale: float = 1.0
    e_mm: float = 0.0
    lambda_mm: float = None
    timestep: float = 0.001
    parent: str = ''
    impratio: float = 100.0
    approach_clearance_mm: float = 1.0
    acquisition_clearance_mm: float = 5.0
    sensor_filter_s: float = 0.0
    initial_hold_s: float = 0.1
    final_hold_s: float = 0.0

    def __post_init__(self):
        if self.rods is None:
            self.rods = NOMINAL.tolist()
        if self.object_error is None:
            self.object_error = [0.0, 0.0]
        if self.grasp_error is None:
            self.grasp_error = [0.0, 0.0]
        if self.rod_error is None:
            self.rod_error = [[0.0, 0.0], [0.0, 0.0]]


class Commands:
    def __init__(self, t, encoder, sensors):
        # Callers supply t with all geometry errors removed.
        self.t = t
        self.encoder = encoder.copy()
        self.sensors = sensors
        self.status = 'running'
        self.rng = np.random.default_rng(t.seed)

    def emit(self, phase, command):
        self.encoder = np.array(command).copy()
        return phase, self.encoder.copy()

    def line(self, phase, end):
        end = np.array(end)
        duration = max(0.2, np.linalg.norm(end - self.encoder) / (0.005 * self.t.speed_scale))
        yield from self.move(phase, end, duration)

    def acquire(self):
        for _ in range(round(5 / self.t.timestep)):
            if self.sensors.norms[0] > self.t.touch_threshold_N:
                # Only bearing force/encoder/nominal center; no object state.
                force = self.sensors.filtered[0].copy()
                normal = -force / np.linalg.norm(force)
                noise = float(self.rng.normal(0, math.radians(self.t.noise_deg)))
                measured = rotation(noise) @ normal
                theta = float(wrap(math.atan2(measured[1], measured[0]) - np.pi / 2))
                center = np.array(self.t.rods[0]) / 1000
                point = center - 0.005 * measured
                lever = -((point - self.encoder)[0] * measured[1] - (point - self.encoder)[1] * measured[0])
                if self.t.strategy == 'C' and (lever < 0.005 or abs(theta) > math.radians(15)):
                    self.status = 'acquisition_abort'
                    return None
                return theta
            yield self.emit('seek_touch', self.encoder + np.array([0, 0.005 * self.t.timestep]))
        self.status = 'acquisition_timeout'
        return None

    def run(self):
        t = self.t
        yield from self.move('initial_settle', self.encoder, t.initial_hold_s)
        if t.experiment in ('E1', 'E2'):
            yield from self.line('approach', [0, 0])
            yield from self.line('shift', -D0 / 1000)
            if t.experiment == 'E1':
                yield from self.orbit()
        elif t.experiment in ('E3', 'E4'):
            yield from self.orbit()
        elif t.experiment == 'E5':
            yield from self.line('shift', [0, 0.010])
        elif t.experiment == 'E6':
            yield from self.line('shift', [0, 0.010])
            yield from self.line('retreat', [0, 0.007])
            yield from self.line('carry', [0.050, 0.007])
            yield from self.line('reengage', [0.050, 0.010])
            yield from self.orbit(rod=1)
        elif t.experiment == 'creep':
            yield from self.line('shift', -D0 / 1000)
            yield from self.move('creep_hold', self.encoder, 9.0)
        elif t.experiment == 'E7':
            if t.strategy == 'A':
                yield from self.line('nominal_approach', [-0.020, 0.005])
                angle = 90.0
            else:
                theta = yield from self.acquire()
                if theta is None:
                    yield from self.move('settle', self.encoder, t.final_hold_s)
                    return
                angle = 90 - math.degrees(theta) if t.strategy == 'C' else 90.0
            yield from self.orbit(angle=angle)
        yield from self.move('settle', self.encoder, t.final_hold_s)
        self.status = 'complete'


PAPER_FIELDS = [
    'time_s',
    'phase',
    'gripper_x_mm',
    'gripper_y_mm',
    'gripper_theta_deg',
    'object_x_mm',
    'object_y_mm',
    'object_theta_deg',
    'relative_x_mm',
    'relative_y_mm',
    'relative_theta_deg',
    'grasp_body_x_mm',
    'grasp_body_y_mm',
    'rod_a_normal_N',
    'rod_b_normal_N',
    'rod_a_tangent_N',
    'rod_b_tangent_N',
    'pad_overlap',
    'grasp_lost',
    'pinch_preload_N',
    'grasp_fx_N',
    'grasp_fy_N',
    'grasp_torque_about_pad_Nm',
]


def row(exp, phase):
    force, tangent, _, _, _ = exp.contacts(True)
    relative = exp.data.qpos[:2] - exp.gripper
    body = rotation(exp.data.qpos[2]).T @ (-relative)
    applied = exp.data.qfrc_applied[:3]
    lever = exp.gripper - exp.data.qpos[:2]
    torque = applied[2] - (lever[0] * applied[1] - lever[1] * applied[0])
    return [
        float(exp.data.time),
        phase,
        *exp.gripper * 1000,
        0.0,
        *exp.data.qpos[:2] * 1000,
        math.degrees(exp.data.qpos[2]),
        *relative * 1000,
        math.degrees(exp.data.qpos[2]),
        *body * 1000,
        *force,
        *tangent,
        exp.overlap,
        int(exp.lost),
        20.0,
        *applied[:2],
        torque,
    ]


def run_trial(t, *, plant_class, row_function, commands_class):
    """Run one trial; return its final measurements and its 100 Hz log rows."""
    exp = plant_class(t)
    sensors = PaperSensors(exp, t)
    nominal = replace(t, object_error=[0, 0], grasp_error=[0, 0], rod_error=[[0, 0], [0, 0]], initial_angle_deg=0)
    controller = commands_class(nominal, exp.gripper, sensors)
    rows = [row_function(exp, 'initial')]
    phases = []
    turn_start = None
    shift_end = None
    turn_body = None
    creep_start = None
    settle_start = None
    separation = None
    force_unload = None
    separation_state = None
    ever = False
    for k, (phase, command) in enumerate(controller.run(), 1):
        if phase == 'turn' and turn_start is None:
            turn_start = float(exp.data.time)
            turn_body = rotation(exp.data.qpos[2]).T @ (exp.gripper - exp.data.qpos[:2]) * 1000
            separation_rod_index = 0 if t.experiment == 'E6' else 1
            other_loaded = bool(exp.contacts()[separation_rod_index] > 0.02)
            separation_state = 'awaiting_separation' if other_loaded else 'initially_unloaded'
        if phase == 'settle' and settle_start is None:
            settle_start = exp.data.qpos[:3].copy()
        if phase == 'creep_hold' and creep_start is None:
            creep_start = exp.data.qpos[:3].copy()
        exp.step(np.array(command))
        sensors.update()
        if not np.all(np.isfinite(exp.data.qpos)) or np.any(exp.data.warning.number):
            raise FloatingPointError(f'{t.name}: MuJoCo nonfinite state / warning')
        force = exp.contacts()
        ever |= bool(np.any(force > 0.02))
        if phase == 'shift':
            shift_end = exp.data.qpos[:3].copy()
        if phase == 'turn' and separation_state != 'initially_unloaded':
            if force_unload is None and force[separation_rod_index] < 1e-9:
                force_unload = float(exp.data.time) - turn_start
            if separation is None and exp.rod_gaps()[separation_rod_index] > 1e-7:
                separation = float(exp.data.time) - turn_start
                separation_state = 'separated'
        if phase not in phases:
            phases.append(phase)
        if k % round(0.01 / t.timestep) == 0:
            rows.append(row_function(exp, phase))
    rows.append(row_function(exp, phases[-1] if phases else 'initial'))
    final = rows[-1]
    relative = np.array(final[8:10])
    theta = final[10]
    body = np.array(final[11:13])
    delta = 90 if t.experiment in ('E1', 'E3', 'E4', 'E6', 'E7') else 0
    d = D0 if t.experiment in ('E1', 'E2', 'E3', 'E4', 'E7', 'creep') else np.array([0.0, -10.0])
    position_error = relative - rotation(math.radians(delta)) @ d
    result = dict(
        status=controller.status,
        signed_angle_error_deg=float((theta - delta + 180) % 360 - 180),
        position_error_mm=float(np.linalg.norm(position_error)),
        world_shift_angle_deg=None if shift_end is None else math.degrees(shift_end[2]),
        grasp_position_error_against_target_mm=(body - (-d)).tolist(),
        turn_grasp_drift_body_mm=None if turn_body is None else (body - turn_body).tolist(),
        nonturning_rod_force_zero_after_turn_start_s=force_unload,
        separation_after_turn_start_s=separation,
        grasp_retained=not exp.lost,
        ever_contact=ever,
        abort=controller.status == 'acquisition_abort',
        creep_drift_pose=None
        if creep_start is None
        else [*((exp.data.qpos[:2] - creep_start[:2]) * 1000), math.degrees(exp.data.qpos[2] - creep_start[2])],
        E2_hold_angle_change_deg=None if settle_start is None else theta - math.degrees(settle_start[2]),
    )
    return result, rows
