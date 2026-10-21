"""Invalid runs and failed writes must leave the saved results intact."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest import mock

from simulation import store
from simulation.__main__ import main, run_main
from simulation.core import Trial
from simulation.experiments import execute


class Settings(unittest.TestCase):
    def test_list_defaults_are_independent(self):
        first, second = Trial('first', 'E1'), Trial('second', 'E1')
        for key in ('rods', 'object_error', 'grasp_error', 'rod_error'):
            with self.subTest(key=key):
                self.assertIsNot(getattr(first, key), getattr(second, key))
        first.rods[0][0] = 100
        first.rod_error[0][0] = 100
        self.assertNotEqual(first.rods, second.rods)
        self.assertNotEqual(first.rod_error, second.rod_error)

    def test_invalid_settings_report_the_field(self):
        for key, value in (
            ('name', ''),
            ('experiment', 'unknown'),
            ('strategy', 'D'),
            ('active_rods', 0),
            ('active_rods', 1.0),
            ('free', 1),
            ('seed', -1),
            ('rods', [[0, 0]]),
            ('rod_error', [[0, 0], [0, float('nan')]]),
            ('object_error', None),
            ('grasp_error', [0, float('inf')]),
            ('timestep', 0),
            ('timestep', 0.1),
            ('speed_scale', -1),
            ('speed_scale', True),
            ('rho_mm', -1),
            ('noise_deg', float('nan')),
            ('initial_hold_s', -1),
            ('lambda_mm', 'invalid'),
            ('lambda_mm', 0),
        ):
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                Trial(**({'name': 'trial', 'experiment': 'E1'} | {key: value}))
        with self.assertRaisesRegex(ValueError, 'lambda_mm'):
            Trial('shift', 'E2')

    def test_runner_rejects_invalid_workers_and_duplicate_names(self):
        for workers in (0, -1, True, 1.5):
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, 'workers'):
                execute([], workers)
        with self.assertRaisesRegex(ValueError, 'unique'):
            execute([Trial('same', 'E1'), Trial('same', 'E1')], 1)

    def test_worker_failure_identifies_the_trial(self):
        failed = Future()
        failed.set_exception(FloatingPointError('nonfinite state'))
        with mock.patch('simulation.experiments.ProcessPoolExecutor') as pool:
            pool.return_value.__enter__.return_value.submit.return_value = failed
            with self.assertRaisesRegex(RuntimeError, 'broken failed: nonfinite state'):
                execute([Trial('broken', 'E1')], 1)


class SavedResults(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'simulation.json'
        patch = mock.patch.object(store, 'RESULTS', self.path)
        patch.start()
        self.addCleanup(patch.stop)

    def test_overwrite_requires_explicit_flag(self):
        store.save_results([], mujoco='test')
        original = self.path.read_bytes()
        with mock.patch('simulation.experiments.execute') as run:
            with self.assertRaisesRegex(FileExistsError, '--overwrite'):
                run_main(1)
            run.assert_not_called()
        with self.assertRaises(FileExistsError):
            store.save_results([])
        self.assertEqual(self.path.read_bytes(), original)
        store.save_results([], overwrite=True, mujoco='replacement')
        self.assertEqual(json.loads(self.path.read_text())['mujoco'], 'replacement')

    def test_write_failure_preserves_old_results_and_cleans_temporary_file(self):
        self.path.write_text('original')
        with mock.patch.object(Path, 'replace', side_effect=OSError('failed write')):
            with self.assertRaisesRegex(OSError, 'failed write'):
                store.save_results([], overwrite=True)
        self.assertEqual(self.path.read_text(), 'original')
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_file_created_during_save_is_not_replaced(self):
        link = os.link

        def concurrent_save(source, destination):
            self.path.write_text('other run')
            link(source, destination)

        with mock.patch.object(store.os, 'link', side_effect=concurrent_save):
            with self.assertRaises(FileExistsError):
                store.save_results([])
        self.assertEqual(self.path.read_text(), 'other run')
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_incomplete_runs_do_not_replace_results(self):
        self.path.write_text('original')
        for status in ('acquisition_abort', 'acquisition_timeout'):
            with (
                self.subTest(status=status),
                mock.patch('simulation.experiments.execute', return_value=[{'name': 'failed', 'status': status}]),
                self.assertRaisesRegex(RuntimeError, 'failed'),
            ):
                run_main(1, overwrite=True)
            self.assertEqual(self.path.read_text(), 'original')

    def test_duplicate_and_incomplete_saved_runs_are_rejected(self):
        complete = {'name': 'trial', 'status': 'complete'}
        self.path.write_text(json.dumps({'runs': [complete, complete]}))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            store.load_runs()
        self.path.write_text(json.dumps({'runs': [{'name': 'failed', 'status': 'acquisition_abort'}]}))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            store.load_runs()


class CommandLine(unittest.TestCase):
    def test_failure_has_nonzero_exit_and_useful_message(self):
        error = io.StringIO()
        with (
            mock.patch('sys.argv', ['simulation', 'run']),
            mock.patch('simulation.__main__.run_main', side_effect=RuntimeError('trial failed')),
            contextlib.redirect_stderr(error),
            self.assertRaises(SystemExit) as result,
        ):
            main()
        self.assertEqual(result.exception.code, 1)
        self.assertIn('trial failed', error.getvalue())

    def test_invalid_worker_count_is_rejected(self):
        with (
            mock.patch('sys.argv', ['simulation', 'run', '--workers', '0']),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as result,
        ):
            main()
        self.assertEqual(result.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
