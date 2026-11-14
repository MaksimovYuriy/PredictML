import argparse
from .pipeline import run


def add_training_options(parser):
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden-sizes', type=int, nargs='+')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--patience', type=int, default=20, help='validation checks without material improvement')
    parser.add_argument('--min-delta', type=float, default=0)
    parser.add_argument('--disable-early-stopping', action='store_true')
    parser.add_argument('--validation-interval', type=int, default=1)


def training_settings(args):
    from .config import TrainingConfig
    return TrainingConfig(epochs=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size,
                          patience=None if args.disable_early_stopping else args.patience, min_delta=args.min_delta,
                          validation_interval=args.validation_interval, seed=args.seed)


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
    p.add_argument('--model-kind', choices=['mlp', 'linear', 'baseline', 'best'], default='mlp')
    add_training_options(p)
    p = sub.add_parser('predict-table')
    p.add_argument('--file', required=True)
    p.add_argument('--sheet')
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('compare')
    p.add_argument('scenario', choices=['materials', 'price'])
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--months', type=int, choices=[1, 2, 3], default=3)
    p.add_argument('--epochs', type=int, default=200)
    add_training_options(p)
    p.set_defaults(seed=42)
    args = parser.parse_args()
    try:
        if args.command == 'compare':
            from .comparison import compare_service
            report = compare_service(args.scenario, args.data, args.output, settings=training_settings(args),
                                     hidden_sizes=args.hidden_sizes, months=args.months)
            print('Recommended by validation:', report['recommended_model'])
            print('Report:', args.output+'/comparison.html')
        elif args.command == 'inspect-table':
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
            if args.seed is not None:
                config['seed'] = args.seed
            model = train_file(args.file, args.output, config, sheet=args.sheet, settings=training_settings(args),
                               hidden_sizes=tuple(args.hidden_sizes or (32, 16)), model_kind=args.model_kind)
            print('Recommended by validation:', model.report['recommended_model'])
            print('Report:', args.output+'/comparison.html')
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
