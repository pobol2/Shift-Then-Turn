"""Locations of the saved results and of the paper's generated files."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results' / 'simulation.json'
TABLES = ROOT / 'paper' / 'tables'
FIGURES = ROOT / 'paper' / 'figures'


def save_results(records, **info):
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(dict(**info, runs=records), indent=1) + '\n', encoding='utf-8')


def load_runs():
    """Saved runs by name."""
    return {r['name']: r for r in json.loads(RESULTS.read_text(encoding='utf-8'))['runs']}


def save_text(path, text):
    path.parent.mkdir(exist_ok=True)
    path.write_text(text, encoding='utf-8', newline='\n')
    print('wrote', path.relative_to(ROOT))
