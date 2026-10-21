"""Simulations of the paper.

python -m simulation run [--workers N] [--overwrite]   367 MuJoCo runs -> results/simulation.json
python -m simulation tables              paper/tables/sim-turn.tex, sim-sensing.tex
python -m simulation figures             paper/figures/sim-shift.tex, sim-turn.tex
"""

import argparse


def run_main(workers, *, overwrite=False):
    import mujoco

    from . import store
    from .experiments import execute, refinements, trials

    store.check_results_path(overwrite=overwrite)
    cases = trials()
    records = execute(cases + refinements(cases), workers)
    failed = [r['name'] for r in records if r['status'] != 'complete']
    if failed:
        raise RuntimeError(f'runs did not complete: {", ".join(failed)}; results were not saved')
    store.save_results(records, overwrite=overwrite, mujoco=mujoco.__version__)
    print(f'{len(records)} runs saved to results/simulation.json')


def positive_workers(value):
    workers = int(value)
    if workers < 1:
        raise argparse.ArgumentTypeError('workers must be a positive integer')
    return workers


def main():
    parser = argparse.ArgumentParser(
        prog='python -m simulation', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest='command')
    run_parser = commands.add_parser('run', help='run the simulations')
    run_parser.add_argument('--workers', type=positive_workers, default=6)
    run_parser.add_argument('--overwrite', action='store_true', help='replace existing simulation results')
    commands.add_parser('tables', help='write the simulation tables of the paper')
    commands.add_parser('figures', help='write the simulation figures of the paper')
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
    elif args.command == 'run':
        try:
            run_main(args.workers, overwrite=args.overwrite)
        except (OSError, ValueError, RuntimeError, FloatingPointError) as exc:
            parser.exit(1, f'error: {exc}\n')
    elif args.command == 'tables':
        from .tables import tables_main

        tables_main()
    else:
        from .figures import figures_main

        figures_main()


if __name__ == '__main__':
    main()
