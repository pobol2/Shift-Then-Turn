"""Locations of the saved results and of the paper's generated files."""

import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results' / 'simulation.json'
TABLES = ROOT / 'paper' / 'tables'
FIGURES = ROOT / 'paper' / 'figures'


def check_results_path(*, overwrite=False):
    if RESULTS.exists() and not overwrite:
        raise FileExistsError(f'{RESULTS} already exists; use --overwrite to replace it')


def save_results(records, *, overwrite=False, **info):
    check_results_path(overwrite=overwrite)
    text = json.dumps(dict(**info, runs=records), indent=1, allow_nan=False) + '\n'
    RESULTS.parent.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=RESULTS.parent, prefix='.simulation-', suffix='.json', delete=False) as temp:
        temporary = Path(temp.name)
    try:
        temporary.write_text(text, encoding='utf-8', newline='\n')
        if overwrite:
            temporary.replace(RESULTS)
        else:
            # Atomically publish without replacing a file created during the run.
            os.link(temporary, RESULTS)
    finally:
        temporary.unlink(missing_ok=True)


def load_runs():
    """Saved runs by name."""
    runs = json.loads(RESULTS.read_text(encoding='utf-8'))['runs']
    if len({r['name'] for r in runs}) != len(runs):
        raise ValueError(f'{RESULTS}: duplicate run names')
    failed = [r['name'] for r in runs if r['status'] != 'complete']
    if failed:
        raise ValueError(f'{RESULTS}: incomplete runs: {", ".join(failed)}')
    return {r['name']: r for r in runs}


def save_text(path, text):
    path.parent.mkdir(exist_ok=True)
    path.write_text(text, encoding='utf-8', newline='\n')
    print('wrote', path.relative_to(ROOT))
