"""The saved results cover every trial, and the generated paper files are what the code produces."""

import unittest
from unittest import mock

from geometry.__main__ import FIGURES, WRITERS
from simulation import store
from simulation.experiments import refinements, trials
from simulation.figures import figures_main
from simulation.tables import tables_main

SIMULATED = [
    store.TABLES / 'sim-turn.tex',
    store.TABLES / 'sim-sensing.tex',
    store.FIGURES / 'sim-shift.tex',
    store.FIGURES / 'sim-turn.tex',
]
GENERATED = SIMULATED + [FIGURES / name for name in WRITERS]


class Results(unittest.TestCase):
    def test_one_complete_run_per_trial(self):
        runs = store.load_runs()
        cases = trials()
        self.assertEqual(set(runs), {t.name for t in cases + refinements(cases)})
        for name, r in runs.items():
            with self.subTest(name):
                self.assertEqual(r['status'], 'complete')
                if r['parent']:
                    self.assertIn(r['parent'], runs)


class Generated(unittest.TestCase):
    def test_files_match_saved_results(self):
        written = {}
        with mock.patch.object(store, 'save_text', written.__setitem__):
            tables_main()
            figures_main()
        self.assertEqual(sorted(written), sorted(SIMULATED))
        for path, text in written.items():
            with self.subTest(path.name):
                self.assertEqual(path.read_text(encoding='utf-8'), text)

    def test_geometric_figures_match_constructions(self):
        for name, figure in WRITERS.items():
            with self.subTest(name):
                self.assertEqual((FIGURES / name).read_text(encoding='utf-8'), figure())

    def test_every_generated_file_is_included(self):
        sources = ''.join(p.read_text(encoding='utf-8') for p in (store.ROOT / 'paper' / 'sections').glob('*.tex'))
        for path in GENERATED:
            with self.subTest(path.name):
                self.assertIn(f'\\input{{{path.parent.name}/{path.stem}}}', sources)


if __name__ == '__main__':
    unittest.main()
