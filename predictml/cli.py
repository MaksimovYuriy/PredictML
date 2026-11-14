import argparse
from .pipeline import run


def main():
    parser = argparse.ArgumentParser(description='Standalone ServiceFlow forecasting')
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('train-predict', 'predict'):
        p = sub.add_parser(command)
        p.add_argument('scenario', choices=['materials', 'price'])
        p.add_argument('--data', required=True)
        p.add_argument('--mode', choices=['demand-v2', 'legacy-parity'], default='demand-v2')
        p.add_argument('--output', required=True)
        p.add_argument('--months', type=int, choices=[1, 2, 3], default=3)
        if command == 'predict':
            p.add_argument('--model', required=True)
        else:
            p.add_argument('--epochs', type=int)
    p = sub.add_parser('prepare')
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--as-of', required=True, help='YYYY-MM-DD; only preceding complete months by default')
    p.add_argument('--mode', choices=['demand-v2', 'legacy-parity'], default='demand-v2')
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            from .preparation import prepare
            prepare(args.data, args.output, args.as_of, mode=args.mode)
        else:
            run(args.scenario, args.data, args.output,
                train=args.command == 'train-predict',
                model=getattr(args, 'model', None), months=args.months,
                epochs=getattr(args, 'epochs', None), mode=args.mode)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
