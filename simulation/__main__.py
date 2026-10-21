"""Simulations of the paper.

python -m simulation run [--workers N]   367 MuJoCo runs -> results/simulation.json
python -m simulation tables              paper/tables/sim-turn.tex, sim-sensing.tex
python -m simulation figures             paper/figures/sim-shift.tex, sim-turn.tex
"""

import argparse


def run_main(workers):
    import mujoco

    from . import store
    from .experiments import execute, refinements, trials

    cases = trials()
    records = execute(cases + refinements(cases), workers)
    store.save_results(records, mujoco=mujoco.__version__)
    failed = [r['name'] for r in records if r['status'] != 'complete']
    print(f'{len(records)} runs saved to results/simulation.json' + (f'; not complete: {failed}' if failed else ''))


def main():
    parser = argparse.ArgumentParser(
        prog='python -m simulation', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest='command')
    commands.add_parser('run', help='run the simulations').add_argument('--workers', type=int, default=6)
    commands.add_parser('tables', help='write the simulation tables of the paper')
    commands.add_parser('figures', help='write the simulation figures of the paper')
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
    elif args.command == 'run':
        run_main(args.workers)
    elif args.command == 'tables':
        from .tables import tables_main

        tables_main()
    else:
        from .figures import figures_main

        figures_main()


if __name__ == '__main__':
    main()
