"""Geometric figures of the paper, recomputed from its constructions.

python -m geometry       paper/figures/tworod-example.tex, holding.tex, approach.tex
"""

import argparse

from simulation import store

from . import approach, holding, tworod

FIGURES = store.FIGURES
WRITERS = {'tworod-example.tex': tworod.figure, 'holding.tex': holding.figure, 'approach.tex': approach.figure}


def main():
    argparse.ArgumentParser(
        prog='python -m geometry', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    for name, figure in WRITERS.items():
        store.save_text(FIGURES / name, figure())


if __name__ == '__main__':
    main()
