"""Shared E1--E7 trial settings, sensors and gripper commands.

Distances are millimetres in trial metadata and SI in the physics engine.
Plant perturbations are removed before constructing the controller.
"""

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from geometry.construction import Q as rotation

HERE = Path(__file__).resolve().parent
_MODEL = ET.parse(HERE / 'model.xml').getroot()
ROD_RADIUS_M = float(_MODEL.find(".//default[@class='rod']/geom").get('size').split()[0])
ROD_RADIUS_MM = 1000 * ROD_RADIUS_M
PLATE_HALF_M = np.fromstring(_MODEL.find(".//geom[@name='object_geom']").get('size'), sep=' ')[:2]
NOMINAL = np.array(
    [
        np.fromstring(_MODEL.find(f".//body[@name='rod_{label}_body']").get('pos'), sep=' ')[:2] * 1000
        for label in ('lower', 'upper')
    ]
)
RHO_MM = 1000 * float(_MODEL.find(".//default[@class='pad']/geom").get('friction').split()[1])
PINCH_N = 5.0  # Each pad; total sliding capacity 10 N with mu=1.
CONTACT_FORCE_N = 0.02
CONTACT_GAP_M = 1e-7
D0 = np.array([20.0, -5.0])
WAPP = np.array([-1.0, 1.0]) / np.sqrt(2)


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


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
    rods: list = field(default_factory=lambda: NOMINAL.tolist())
    active_rods: int = 2
    free: bool = True
    rod_mu: float = 0.0
    rho_mm: float = RHO_MM
    lever_mm: float = 16.0
    object_error: list = field(default_factory=lambda: [0.0, 0.0])
    grasp_error: list = field(default_factory=lambda: [0.0, 0.0])
    rod_error: list = field(default_factory=lambda: [[0.0, 0.0], [0.0, 0.0]])
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
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError('name must be a nonempty string')
        if self.experiment not in ('E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'creep'):
            raise ValueError(f'unknown experiment: {self.experiment!r}')
        if self.strategy not in ('A', 'B', 'C'):
            raise ValueError(f'unknown strategy: {self.strategy!r}')
        if (
            isinstance(self.active_rods, bool)
            or not isinstance(self.active_rods, (int, np.integer))
            or self.active_rods not in (1, 2)
        ):
            raise ValueError('active_rods must be 1 or 2')
        if not isinstance(self.free, bool):
            raise ValueError('free must be a boolean')
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        for key, shape in (('rods', (2, 2)), ('object_error', (2,)), ('grasp_error', (2,)), ('rod_error', (2, 2))):
            try:
                value = np.asarray(getattr(self, key), dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'{key} must contain finite numbers with shape {shape}') from exc
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f'{key} must contain finite numbers with shape {shape}')
            setattr(self, key, value.tolist())
        positive = ('timestep', 'speed_scale', 'lever_mm', 'touch_threshold_N', 'impratio')
        nonnegative = (
            'rod_mu',
            'rho_mm',
            'noise_deg',
            'approach_clearance_mm',
            'acquisition_clearance_mm',
            'sensor_filter_s',
            'initial_hold_s',
            'final_hold_s',
        )
        for key in (*positive, *nonnegative, 'initial_angle_deg', 'e_mm'):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)
            ):
                raise ValueError(f'{key} must be finite')
            if key in positive and value <= 0:
                raise ValueError(f'{key} must be positive')
            if key in nonnegative and value < 0:
                raise ValueError(f'{key} must be nonnegative')
        if self.timestep > 0.01:
            raise ValueError('timestep must be at most 0.01 s for 100 Hz logging')
        if self.lambda_mm is not None:
            if (
                isinstance(self.lambda_mm, bool)
                or not isinstance(self.lambda_mm, (int, float, np.integer, np.floating))
                or not np.isfinite(self.lambda_mm)
                or self.lambda_mm <= 0
            ):
                raise ValueError('lambda_mm must be finite and positive')
        if self.experiment == 'E2' and self.lambda_mm is None:
            raise ValueError('E2 requires lambda_mm')


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
        if self.t.experiment == 'E2':
            end = end - self.t.e_mm * W / 1000
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
                point = center - ROD_RADIUS_M * measured
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

    def move(self, phase, end, duration):
        start = self.encoder.copy()
        end = np.array(end)
        n = max(1, round(duration / self.t.timestep))
        if duration == 0:
            return
        for k in range(1, n + 1):
            yield self.emit(phase, start + (end - start) * (k / n))

    def orbit(self, rod=0, angle=90):
        center = np.array(self.t.rods[rod]) / 1000
        start = self.encoder.copy()
        n = round(abs(angle) / (10 * self.t.speed_scale) / self.t.timestep)
        for k in range(1, n + 1):
            yield self.emit('turn', center + rotation(np.deg2rad(angle) * k / n) @ (start - center))
