"""Geometric figures of the paper, recomputed from its constructions.

python -m geometry       paper/figures/tworod-example.tex, holding.tex, approach.tex
"""

import argparse
from pathlib import Path

from . import approach, holding, tworod

FIGURES = Path(__file__).resolve().parent.parent / 'paper' / 'figures'
WRITERS = {'tworod-example.tex': tworod.figure, 'holding.tex': holding.figure, 'approach.tex': approach.figure}


def main():
    argparse.ArgumentParser(
        prog='python -m geometry', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    for name, figure in WRITERS.items():
        (FIGURES / name).write_text(figure(), encoding='utf-8', newline='\n')
        print('wrote', f'paper/figures/{name}')


if __name__ == '__main__':
    main()
