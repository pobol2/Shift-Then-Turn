"""MuJoCo plant, E1--E7 cases and parallel runner for the paper's simulations."""

import math
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace

import mujoco
import numpy as np

from .core import (
    CONTACT_FORCE_N,
    CONTACT_GAP_M,
    D0,
    HERE,
    PINCH_N,
    PLATE_HALF_M,
    ROD_RADIUS_M,
    WAPP,
    Commands,
    PaperSensors,
    Trial,
    W,
    rotation,
)


class Plant:
    def update_overlap(self):
        # The grasp wrench is supplied exclusively by native contacts.
        local = rotation(self.data.qpos[2]).T @ (self.gripper - self.data.qpos[:2])
        self.overlap = float(np.all(np.abs(local) < self.half))
        # Retention is updated from both pad forces after initial pinch settling.

    def __init__(self, t):
        self.trial = t
        self.half = PLATE_HALF_M.copy()
        self.overlap = 1.0
        self.lost = False
        # Unknown geometry enters only the plant's initial state.
        if t.experiment in ('E3', 'E4', 'E7'):
            start = -D0 - (np.array([0.0, t.acquisition_clearance_mm]) if t.experiment == 'E7' else 0)
            theta = np.deg2rad(t.initial_angle_deg)
            true_grasp = -D0 + t.grasp_error
            center = start - rotation(theta) @ true_grasp
        else:
            start = -t.approach_clearance_mm * WAPP if t.experiment in ('E1', 'E2') else np.zeros(2)
            theta = 0.0
            center = start + np.array(t.object_error)
        # E2 perturbs the grasp, preserving nominal rod/plate clearance.
        self.gripper = (start - t.e_mm * W if t.experiment == 'E2' else start) / 1000
        initial = np.array([*np.array(center) / 1000, theta])
        root = ET.parse(HERE / 'model.xml').getroot()
        root.find('option').set('impratio', str(t.impratio))
        root.find('option').set('timestep', str(t.timestep))
        centers = np.array(t.rods) + np.array(t.rod_error)
        for i, label in enumerate(('lower', 'upper')):
            body = root.find(f".//body[@name='rod_{label}_body']")
            body.set('pos', f'{centers[i, 0] / 1000} {centers[i, 1] / 1000} 0.015')
            geom = root.find(f".//geom[@name='rod_{label}']")
            geom.set('friction', f'{t.rod_mu} 0 0')
            if not t.free:
                rotor = root.find(f".//body[@name='rod_{label}_rotor']")
                rotor.remove(rotor.find('joint'))
            if i >= t.active_rods:
                geom.set('contype', '0')
                geom.set('conaffinity', '0')
        for label in ('top', 'bottom'):
            root.find(f".//geom[@name='pad_{label}']").set('friction', f'1 {t.rho_mm / 1000} 0')
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:3] = initial
        self.gripper_addresses = [self.model.joint('gripper_' + a).qposadr[0] for a in ['x', 'y']]
        self.gripper_dofs = [self.model.joint('gripper_' + a).dofadr[0] for a in ['x', 'y']]
        self.data.qpos[self.gripper_addresses] = self.gripper
        self.update_overlap()
        self.data.ctrl[:] = [-PINCH_N, PINCH_N]
        self.rod_ids = [self.model.geom('rod_' + n).id for n in ('lower', 'upper')]
        self.pad_ids = [self.model.geom('pad_' + n).id for n in ('top', 'bottom')]
        self.object_id = self.model.geom('object_geom').id
        mujoco.mj_forward(self.model, self.data)

    def contacts(self, diagnostics=False):
        forces = np.zeros(2)
        tangential = np.zeros(2)
        couples = np.zeros(2)
        present = np.zeros(2, dtype=int)
        upper_arm_weighted = 0.0
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            pair = {contact.geom1, contact.geom2}
            for i, rod_id in enumerate(self.rod_ids):
                if pair == {self.object_id, rod_id}:
                    present[i] |= int(contact.dist <= CONTACT_GAP_M and contact.efc_address >= 0)
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(self.model, self.data, contact_id, wrench)
                    forces[i] += max(0.0, wrench[0])
                    tangential[i] += np.linalg.norm(wrench[1:3])
                    couples[i] += np.linalg.norm(wrench[3:6])
                    if i == 1:
                        normal = contact.frame.reshape(3, 3)[0, :2]
                        lever = contact.pos[:2] - self.gripper
                        arm = abs(lever[0] * normal[1] - lever[1] * normal[0])
                        upper_arm_weighted += arm * max(0.0, wrench[0])
        arm = upper_arm_weighted / forces[1] if forces[1] > 1e-12 else 0.0
        return (forces, tangential, couples, arm, present) if diagnostics else forces

    def physical_contacts(self):
        normal = np.zeros(2)
        count = np.zeros(4, dtype=int)
        for k in range(self.data.ncon):
            c = self.data.contact[k]
            if c.efc_address < 0 or c.dist > CONTACT_GAP_M:
                continue
            for i, g in enumerate(self.pad_ids + self.rod_ids):
                if {c.geom1, c.geom2} == {g, self.object_id}:
                    count[i] += 1
                    if i < 2:
                        f = np.zeros(6)
                        mujoco.mj_contactForce(self.model, self.data, k, f)
                        normal[i] += max(0.0, f[0])
        return normal, count

    def rod_gaps(self):
        # Independent planar circle/rectangle signed distance, including corners.
        local = (self.data.geom_xpos[self.rod_ids, :2] - self.data.qpos[:2]) @ rotation(self.data.qpos[2])
        q = np.abs(local) - self.half
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(np.max(q, axis=1), 0) - ROD_RADIUS_M

    def rod_spins(self):
        return [
            float(self.data.qpos[self.model.joint('rod_' + n + '_spin').qposadr[0]]) if self.trial.free else 0.0
            for n in ('lower', 'upper')
        ]

    def step(self, command):
        # Prescribed position-control boundary with matching velocity. The huge
        # slide armature makes contact reaction displacement negligible; only
        # the gripper state is commanded, never the dynamic plate or rods.
        self.data.qvel[self.gripper_dofs] = (command - self.gripper) / self.model.opt.timestep
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.gripper = self.data.qpos[self.gripper_addresses].copy()
        self.update_overlap()
        normal, count = self.physical_contacts()
        if self.data.time >= self.trial.initial_hold_s:
            self.lost |= not bool(self.overlap and np.all(normal > CONTACT_FORCE_N) and np.all(count[:2] > 0))


def row(exp, phase):
    relative = exp.data.qpos[:2] - exp.gripper
    body = rotation(exp.data.qpos[2]).T @ (-relative)
    rods, tangent, _, _, _ = exp.contacts(True)
    normal, count = exp.physical_contacts()
    force = np.zeros(3)
    torque = np.zeros(3)
    for k in range(exp.data.ncon):
        c = exp.data.contact[k]
        if exp.object_id not in (c.geom1, c.geom2) or not any(p in (c.geom1, c.geom2) for p in exp.pad_ids):
            continue
        f = np.zeros(6)
        mujoco.mj_contactForce(exp.model, exp.data, k, f)
        sign = 1 if c.geom2 == exp.object_id else -1
        world = sign * c.frame.reshape(3, 3).T @ f[:3]
        force += world
        torque += sign * c.frame.reshape(3, 3).T @ f[3:] + np.cross(c.pos - np.array([*exp.gripper, 0.015]), world)
    gaps = exp.rod_gaps() * 1000
    spins = exp.rod_spins()
    return dict(
        time_s=float(exp.data.time),
        phase=phase,
        gripper_x_mm=exp.gripper[0] * 1000,
        gripper_y_mm=exp.gripper[1] * 1000,
        gripper_theta_deg=0.0,
        object_x_mm=exp.data.qpos[0] * 1000,
        object_y_mm=exp.data.qpos[1] * 1000,
        object_theta_deg=math.degrees(exp.data.qpos[2]),
        relative_x_mm=relative[0] * 1000,
        relative_y_mm=relative[1] * 1000,
        relative_theta_deg=math.degrees(exp.data.qpos[2]),
        grasp_body_x_mm=body[0] * 1000,
        grasp_body_y_mm=body[1] * 1000,
        rod_a_normal_N=rods[0],
        rod_b_normal_N=rods[1],
        rod_a_tangent_N=tangent[0],
        rod_b_tangent_N=tangent[1],
        pad_overlap=exp.overlap,
        grasp_lost=int(exp.lost),
        pinch_preload_N=2 * PINCH_N,
        grasp_fx_N=force[0],
        grasp_fy_N=force[1],
        grasp_torque_about_pad_Nm=torque[2],
        pad_top_normal_N=normal[0],
        pad_bottom_normal_N=normal[1],
        pad_top_contacts=count[0],
        pad_bottom_contacts=count[1],
        rod_a_contacts=count[2],
        rod_b_contacts=count[3],
        rod_a_gap_mm=gaps[0],
        rod_b_gap_mm=gaps[1],
        pad_top_supported=int(normal[0] > CONTACT_FORCE_N),
        pad_bottom_supported=int(normal[1] > CONTACT_FORCE_N),
        rod_a_supported=int(rods[0] > CONTACT_FORCE_N),
        rod_b_supported=int(rods[1] > CONTACT_FORCE_N),
        rod_a_spin_rad=spins[0],
        rod_b_spin_rad=spins[1],
    )


def run_trial(t):
    """Run one trial; return its final measurements and its 100 Hz log rows."""
    exp = Plant(t)
    sensors = PaperSensors(exp, t)
    nominal = replace(t, object_error=[0, 0], grasp_error=[0, 0], rod_error=[[0, 0], [0, 0]], initial_angle_deg=0)
    controller = Commands(nominal, exp.gripper, sensors)
    rows = [row(exp, 'initial')]
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
            other_loaded = bool(exp.contacts()[separation_rod_index] > CONTACT_FORCE_N)
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
        ever |= bool(np.any(force > CONTACT_FORCE_N))
        if phase == 'shift':
            shift_end = exp.data.qpos[:3].copy()
        if phase == 'turn' and separation_state != 'initially_unloaded':
            if force_unload is None and force[separation_rod_index] < 1e-9:
                force_unload = float(exp.data.time) - turn_start
            if separation is None and exp.rod_gaps()[separation_rod_index] > CONTACT_GAP_M:
                separation = float(exp.data.time) - turn_start
                separation_state = 'separated'
        if phase not in phases:
            phases.append(phase)
        if k % round(0.01 / t.timestep) == 0:
            rows.append(row(exp, phase))
    rows.append(row(exp, phases[-1] if phases else 'initial'))
    final = rows[-1]
    relative = np.array([final['relative_x_mm'], final['relative_y_mm']])
    theta = final['relative_theta_deg']
    body = np.array([final['grasp_body_x_mm'], final['grasp_body_y_mm']])
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


def trials():
    """The E1--E7 and creep cases of the paper; E7 strategies share paired seeds."""
    cases = []
    rng = np.random.default_rng(7)
    for i in range(10):
        cases.append(Trial(f'E1_trial{i:02}', 'E1', object_error=rng.uniform(-0.3, 0.3, 2).tolist()))
    cases.append(replace(cases[0], name='E1_trial00_half_speed', speed_scale=0.5, parent='E1_trial00'))
    cases.append(Trial('creep_9s', 'creep'))
    for lam in [float(np.linalg.norm([-36, 9])), 30.0]:
        x = -lam * D0 / np.linalg.norm(D0)
        for e in [-1, -0.5, 0, 0.5, 1]:
            cases.append(
                Trial(
                    f'E2_lambda{lam:.4f}_e{e}',
                    'E2',
                    rods=[[x[0], 20], [-45, x[1]]],
                    e_mm=e,
                    lambda_mm=lam,
                    final_hold_s=3.0,
                )
            )
    for rho in [0.5, 1, 2, 3, 4, 5]:
        cases.append(Trial(f'E3_rho{rho}_l16', 'E3', active_rods=1, rho_mm=rho))
    for lever in [8, 12, 16, 20]:
        cases.append(
            Trial(f'E3_l{lever}_rho2', 'E3', rods=[[-20 - lever, 20], [-45, 9]], active_rods=1, lever_mm=lever)
        )
    for free, mus in [(True, [0.5, 1.0]), (False, [0.1, 0.2, 0.3])]:
        for mu in mus:
            cases.append(Trial(f'E4_{"free" if free else "fixed"}_mu{mu}', 'E4', active_rods=1, free=free, rod_mu=mu))
    for active in [1, 2]:
        rods = [[0, 20], [30, 20]] if active == 1 else [[-15, 20], [15, 20]]
        cases.append(Trial(f'E5_{active}rod', 'E5', rods=rods, active_rods=active, rod_error=[[0.5, 0], [0.5, 0]]))
    cases.append(Trial('E6_reengage', 'E6', rods=[[-15, 20], [15, 20]]))
    rng = np.random.default_rng(7)
    for i in range(50):
        grasp = rng.uniform(-2, 2, 2).tolist()
        theta = float(rng.uniform(-3, 3))
        rod = rng.uniform(-0.5, 0.5, 2).tolist()
        for mount in [False, True]:
            for strategy in ['A', 'B', 'C']:
                cases.append(
                    Trial(
                        f'E7_{strategy}_{"mount" if mount else "nominal"}_trial{i:02}',
                        'E7',
                        active_rods=1,
                        grasp_error=grasp,
                        initial_angle_deg=theta,
                        rod_error=[rod if mount else [0, 0], [0, 0]],
                        strategy=strategy,
                        seed=7 + i,
                    )
                )
    return cases


# Trial inputs and measurements that the paper uses, by experiment.
INPUTS = {
    'E2': ['e_mm', 'lambda_mm'],
    'E3': ['rho_mm', 'lever_mm'],
    'E4': ['free', 'rod_mu'],
    'E5': ['active_rods'],
    'E7': ['strategy', 'initial_angle_deg', 'grasp_error', 'rod_error'],
}
MEASURED = {
    'E1': [
        'world_shift_angle_deg',
        'grasp_position_error_against_target_mm',
        'turn_grasp_drift_body_mm',
        'nonturning_rod_force_zero_after_turn_start_s',
        'separation_after_turn_start_s',
        'grasp_retained',
    ],
    'creep': ['creep_drift_pose'],
    'E2': ['world_shift_angle_deg', 'E2_hold_angle_change_deg'],
    'E3': ['turn_grasp_drift_body_mm'],
    'E4': ['turn_grasp_drift_body_mm'],
    'E5': ['position_error_mm'],
    'E6': ['position_error_mm'],
    'E7': ['position_error_mm', 'ever_contact', 'abort'],
}


def run(tdict):
    """One trial, reduced to the quantities the paper reports."""
    t = Trial(**tdict)
    m, rows = run_trial(t)
    record = dict(
        name=t.name,
        experiment=t.experiment,
        parent=t.parent,
        timestep=t.timestep,
        speed_scale=t.speed_scale,
        **{k: getattr(t, k) for k in INPUTS.get(t.experiment, [])},
        status=m['status'],
        signed_angle_error_deg=m['signed_angle_error_deg'],
        **{k: m[k] for k in MEASURED[t.experiment]},
    )
    if t.experiment == 'E6':
        # Load on the other rod after the retreat, to show it is not touched again.
        record['rod_a_peak_after_retreat_N'] = max(
            r['rod_a_normal_N'] for r in rows if r['phase'] in ('carry', 'reengage', 'turn')
        )
    if t.experiment == 'E2' and t.e_mm == -1 and not t.parent:
        # Tilt along the shift against grasp travel, for the shift figure.
        def xy(r):
            return np.array([r['gripper_x_mm'], r['gripper_y_mm']])

        start = xy([r for r in rows if r['phase'] == 'approach'][-1])
        shift = [r for r in rows if r['phase'] == 'shift']
        record['shift_travel_mm'] = [float((xy(r) - start) @ (-D0 / np.linalg.norm(D0))) for r in shift]
        record['shift_tilt_deg'] = [r['object_theta_deg'] for r in shift]
    return record


def execute(cases, workers):
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    if len({t.name for t in cases}) != len(cases):
        raise ValueError('trial names must be unique')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(run, asdict(t)): t.name for t in cases}
        records = {}
        for n, f in enumerate(as_completed(jobs), 1):
            try:
                r = f.result()
            except Exception as exc:
                raise RuntimeError(f'{jobs[f]} failed: {exc}') from exc
            records[r['name']] = r
            if n % 10 == 0 or n == len(jobs):
                print(f'{n}/{len(jobs)} {r["name"]} {r["status"]}', flush=True)
    return [records[t.name] for t in cases]


def refinements(cases):
    refs = []
    for t in cases:
        if t.experiment == 'E4' or t.name in [
            'E1_trial00',
            'E3_rho2_l16',
            'E7_C_nominal_trial00',
            'E6_reengage',
            'E3_rho0.5_l16',
            'E3_rho1_l16',
            'E3_l20_rho2',
            'E2_lambda37.1080_e1',
            'E2_lambda30.0000_e1',
        ]:
            refs.append(replace(t, name=t.name + '_dt_half', timestep=0.0005, parent=t.name))
        if t.name in ['E3_rho0.5_l16', 'E3_rho1_l16', 'E3_l20_rho2']:
            refs.append(replace(t, name=t.name + '_dt_quarter', timestep=0.00025, parent=t.name))
    for t in cases:
        if t.experiment == 'E2':
            refs.append(replace(t, name=t.name + '_half_speed', speed_scale=0.5, parent=t.name))
    return refs
