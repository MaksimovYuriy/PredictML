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
    p = sub.add_parser('inspect-table')
    p.add_argument('--file', required=True)
    p.add_argument('--sheet')
    p.add_argument('--rows', type=int, default=10)
    p = sub.add_parser('train-table')
    p.add_argument('--file', required=True)
    p.add_argument('--sheet')
    p.add_argument('--config', required=True, help='JSON target/features/exclude/types/seed/temporal')
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=50)
    p = sub.add_parser('predict-table')
    p.add_argument('--file', required=True)
    p.add_argument('--sheet')
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        if args.command == 'inspect-table':
            from .tabular.io import preview, sheets
            from pathlib import Path
            result = preview(args.file, args.sheet, args.rows)
            if Path(args.file).suffix.lower() == '.xlsx':
                print('Sheets:', ', '.join(sheets(args.file)))
            print(f"Rows: {result['rows']}; columns: {result['columns']}")
            print('Missing:', result['missing'])
            print(result['preview'].to_string(index=False))
        elif args.command == 'train-table':
            import json
            from pathlib import Path
            from .config import TrainingConfig
            from .tabular.application import train_file
            config = json.loads(Path(args.config).read_text(encoding='utf-8'))
            train_file(args.file, args.output, config, sheet=args.sheet, settings=TrainingConfig(epochs=args.epochs))
        elif args.command == 'predict-table':
            from .tabular.application import predict_file
            predict_file(args.file, args.model, args.output, sheet=args.sheet)
        elif args.command == 'prepare':
            from .preparation import prepare
            prepare(args.data, args.output, args.as_of, mode=args.mode)
        else:
            run(args.scenario, args.data, args.output,
                train=args.command == 'train-predict',
                model=getattr(args, 'model', None), months=args.months,
                epochs=getattr(args, 'epochs', None), mode=args.mode)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
