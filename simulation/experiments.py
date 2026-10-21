"""MuJoCo plant, E1--E7 cases and parallel runner for the paper's simulations."""

import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace

import mujoco
import numpy as np

from . import core as pe

PINCH_N = 5.0  # Each pad; total sliding capacity 10 N with mu=1.
PAD_RADIUS_M = 0.001
CONTACT_FORCE_N = 0.02
CONTACT_GAP_M = 1e-7
FIELDS = pe.PAPER_FIELDS + [
    'pad_top_normal_N',
    'pad_bottom_normal_N',
    'pad_top_contacts',
    'pad_bottom_contacts',
    'rod_a_contacts',
    'rod_b_contacts',
    'rod_a_gap_mm',
    'rod_b_gap_mm',
    'pad_top_supported',
    'pad_bottom_supported',
    'rod_a_supported',
    'rod_b_supported',
    'rod_a_spin_rad',
    'rod_b_spin_rad',
]


class NativePlant:
    def update_overlap(self):
        # The grasp wrench is supplied exclusively by native contacts.
        local = pe.rotation(self.data.qpos[2]).T @ (self.gripper - self.data.qpos[:2])
        self.overlap = float(np.all(np.abs(local) < self.half))
        # Retention is updated from both pad forces after initial pinch settling.

    def __init__(self, t):
        self.trial = t
        self.half = np.array([0.04, 0.015])
        self.overlap = 1.0
        self.lost = False
        # Unknown geometry enters only the plant's initial state.
        if t.experiment in ('E3', 'E4', 'E7'):
            start = np.array([-20.0, 5.0]) - (
                np.array([0.0, t.acquisition_clearance_mm]) if t.experiment == 'E7' else 0
            )
            theta = np.deg2rad(t.initial_angle_deg)
            true_grasp = np.array([-20.0, 5.0]) + t.grasp_error
            center = start - pe.rotation(theta) @ true_grasp
        else:
            start = -t.approach_clearance_mm * pe.WAPP if t.experiment in ('E1', 'E2') else np.zeros(2)
            theta = 0.0
            center = start + np.array(t.object_error)
        # E2 perturbs the grasp, preserving nominal rod/plate clearance.
        self.gripper = (start - t.e_mm * pe.W if t.experiment == 'E2' else start) / 1000
        initial = np.array([*np.array(center) / 1000, theta])
        root = ET.parse(pe.HERE / 'model.xml').getroot()
        root.find('option').set('cone', 'elliptic')
        root.find('option').set('impratio', str(t.impratio))
        root.find('option').set('timestep', str(t.timestep))
        root.find('option').set('iterations', '100')
        root.find('option').set('integrator', 'implicitfast')
        obj = root.find(".//geom[@name='object_geom']")
        obj.set('size', '0.04 0.015 0.0025')
        obj.set('contype', '1')
        obj.set('conaffinity', '6')
        centers = np.array(t.rods) + np.array(t.rod_error)
        for i, label in enumerate(('lower', 'upper')):
            body = root.find(f".//body[@name='rod_{label}_body']")
            body.set('pos', f'{centers[i, 0] / 1000} {centers[i, 1] / 1000} 0.015')
            geom = root.find(f".//geom[@name='rod_{label}']")
            # Analytic capsule/box collision avoids convex-cylinder tangency artifacts.
            # Caps lie outside the plate z range; the contact region is cylindrical.
            geom.set('type', 'capsule')
            geom.set('size', '0.005 0.025')
            geom.set('condim', '3')
            rotor = root.find(f".//body[@name='rod_{label}_rotor']")
            ET.SubElement(
                rotor,
                'inertial',
                pos='0 0 0',
                mass='.02',
                diaginertia='4.291666666666667e-6 4.291666666666667e-6 2.5e-7',
            )
            geom.set('friction', f'{t.rod_mu} 0 0')
            geom.set('priority', '1')
            geom.set('contype', '2')
            geom.set('conaffinity', '1')
            if not t.free:
                rotor = root.find(f".//body[@name='rod_{label}_rotor']")
                rotor.remove(rotor.find('joint'))
            if i >= t.active_rods:
                geom.set('contype', '0')
                geom.set('conaffinity', '0')
        grip = root.find(".//body[@name='gripper']")
        grip.attrib.pop('mocap')
        grip.set('pos', '0 0 0.015')
        for axis, vector in [('x', '1 0 0'), ('y', '0 1 0')]:
            ET.SubElement(grip, 'joint', name='gripper_' + axis, type='slide', axis=vector, armature='1000000')
        actuator = ET.SubElement(root, 'actuator')
        for label, sign in [('top', 1), ('bottom', -1)]:
            b = ET.SubElement(grip, 'body', name='pad_' + label + '_body', pos=f'0 0 {sign * 0.0035}')
            ET.SubElement(
                b,
                'joint',
                name='pinch_' + label,
                type='slide',
                axis='0 0 1',
                limited='true',
                range='-.001 .001',
                damping='.01',
            )
            ET.SubElement(
                b,
                'geom',
                name='pad_' + label,
                type='sphere',
                size=str(PAD_RADIUS_M),
                mass='.01',
                condim='4',
                friction=f'1 {t.rho_mm / 1000} 0',
                priority='2',
                contype='4',
                conaffinity='1',
                rgba='.9 .4 .1 1',
            )
            ET.SubElement(actuator, 'motor', name='pinch_' + label, joint='pinch_' + label, gear='1')
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
        local = (self.data.geom_xpos[self.rod_ids, :2] - self.data.qpos[:2]) @ pe.rotation(self.data.qpos[2])
        q = np.abs(local) - self.half
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(np.max(q, axis=1), 0) - 0.005

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


def native_row(exp, phase):
    values = pe.row(exp, phase)
    values[19] = 2 * PINCH_N
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
    values[20:23] = [*force[:2], torque[2]]
    return (
        values
        + normal.tolist()
        + count.tolist()
        + (exp.rod_gaps() * 1000).tolist()
        + (normal > CONTACT_FORCE_N).astype(int).tolist()
        + (exp.contacts() > CONTACT_FORCE_N).astype(int).tolist()
        + exp.rod_spins()
    )


class NativeCommands(pe.Commands):
    def line(self, phase, end):
        # Preserve E2's grasp offset throughout the nominal translation path.
        if self.t.experiment == 'E2':
            end = np.array(end) - self.t.e_mm * pe.W / 1000
        yield from super().line(phase, end)

    def move(self, phase, end, duration):
        # Constant speed along the specified line (no bowed paths).
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
            yield self.emit('turn', center + pe.rotation(np.deg2rad(angle) * k / n) @ (start - center))


def trials():
    """The E1--E7 and creep cases of the paper; E7 strategies share paired seeds."""
    Trial = pe.Trial
    cases = []
    rng = np.random.default_rng(7)
    for i in range(10):
        cases.append(Trial(f'E1_trial{i:02}', 'E1', object_error=rng.uniform(-0.3, 0.3, 2).tolist()))
    cases.append(replace(cases[0], name='E1_trial00_half_speed', speed_scale=0.5, parent='E1_trial00'))
    cases.append(Trial('creep_9s', 'creep'))
    for lam in [float(np.linalg.norm([-36, 9])), 30.0]:
        x = -lam * pe.D0 / np.linalg.norm(pe.D0)
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
    t = pe.Trial(**tdict)
    m, rows = pe.run_trial(t, plant_class=NativePlant, row_function=native_row, commands_class=NativeCommands)
    record = dict(
        name=t.name,
        experiment=t.experiment,
        parent=t.parent,
        timestep=t.timestep,
        speed_scale=t.speed_scale,
        **{k: tdict[k] for k in INPUTS.get(t.experiment, [])},
        status=m['status'],
        signed_angle_error_deg=m['signed_angle_error_deg'],
        **{k: m[k] for k in MEASURED[t.experiment]},
    )
    column = {f: i for i, f in enumerate(FIELDS)}
    if t.experiment == 'E6':
        # Load on the other rod after the retreat, to show it is not touched again.
        record['rod_a_peak_after_retreat_N'] = max(
            r[column['rod_a_normal_N']] for r in rows if r[column['phase']] in ('carry', 'reengage', 'turn')
        )
    if t.experiment == 'E2' and t.e_mm == -1 and not t.parent:
        # Tilt along the shift against grasp travel, for the shift figure.
        def xy(r):
            return np.array([r[column['gripper_x_mm']], r[column['gripper_y_mm']]])

        start = xy([r for r in rows if r[column['phase']] == 'approach'][-1])
        shift = [r for r in rows if r[column['phase']] == 'shift']
        record['shift_travel_mm'] = [float((xy(r) - start) @ (-pe.D0 / np.linalg.norm(pe.D0))) for r in shift]
        record['shift_tilt_deg'] = [r[column['object_theta_deg']] for r in shift]
    return record


def execute(cases, workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(run, asdict(t)) for t in cases]
        records = {}
        for n, f in enumerate(as_completed(jobs), 1):
            r = f.result()
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
