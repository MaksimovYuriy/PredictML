import argparse
import sqlite3
from .workflow import run
from .config import (TrainingConfig, DEFAULT_SEED, DEFAULT_HIDDEN_SIZES,
                     TABLE_TRAINING_DEFAULTS, COMPARISON_TRAINING_DEFAULTS,
                     INTERACTIVE_TRAINING_DEFAULTS)


def add_training_options(parser):
    parser.add_argument('--learning-rate', type=float, default=TrainingConfig.learning_rate)
    parser.add_argument('--batch-size', type=int, default=TrainingConfig.batch_size)
    parser.add_argument('--hidden-sizes', type=int, nargs='+')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--patience', type=int, default=INTERACTIVE_TRAINING_DEFAULTS.patience, help='validation checks without material improvement')
    parser.add_argument('--min-delta', type=float, default=TrainingConfig.min_delta)
    parser.add_argument('--disable-early-stopping', action='store_true')
    parser.add_argument('--validation-interval', type=int, default=INTERACTIVE_TRAINING_DEFAULTS.validation_interval)


def training_settings(args):
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
    p.add_argument('--epochs', type=int, default=TABLE_TRAINING_DEFAULTS.epochs)
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
    p.add_argument('--epochs', type=int, default=COMPARISON_TRAINING_DEFAULTS.epochs)
    add_training_options(p)
    p.set_defaults(seed=DEFAULT_SEED)
    for command in ('versions', 'runs'):
        p = sub.add_parser(command)
        p.add_argument('--store', required=True)
    for command in ('show-version', 'export-metrics', 'predict-version', 'recover-run'):
        p = sub.add_parser(command)
        p.add_argument('--store', required=True)
        p.add_argument('--id', required=True, help='version ID, or run ID for recover-run')
        if command == 'export-metrics':
            p.add_argument('--output', required=True)
        if command == 'predict-version':
            p.add_argument('--data', required=True)
            p.add_argument('--output')
            p.add_argument('--sheet')
            p.add_argument('--months', type=int, choices=[1, 2, 3], default=3)
            p.add_argument('--format', choices=['csv', 'xlsx', 'both'], default='both')
    for command in ('retrain', 'fine-tune'):
        p = sub.add_parser(command)
        p.add_argument('--store', required=True)
        p.add_argument('--id', required=True, help='parent version ID')
        p.add_argument('--data', required=True)
        p.add_argument('--sheet')
        p.add_argument('--config', help='optional new table configuration; retrain only')
        p.add_argument('--epochs', type=int, default=INTERACTIVE_TRAINING_DEFAULTS.epochs)
        add_training_options(p)
        p.set_defaults(seed=DEFAULT_SEED)
    p = sub.add_parser('compare-versions')
    p.add_argument('--store', required=True)
    p.add_argument('--ids', required=True, nargs='+')
    p.add_argument('--control', required=True)
    p.add_argument('--sheet')
    p.add_argument('--history', help='optional common demand history CSV before control origin')
    p = sub.add_parser('ui')
    p.add_argument('--workspace')
    p.add_argument('--store')
    p.add_argument('--port',type=int,default=8501)
    p = sub.add_parser('demo')
    p.add_argument('--output',required=True)
    args = parser.parse_args()
    try:
        if args.command == 'demo':
            from .demo import export_examples
            print(export_examples(args.output))
        elif args.command == 'ui':
            from .ui.launcher import launch
            launch(args.workspace,args.store,args.port)
        elif args.command in ('retrain', 'fine-tune'):
            import json
            from pathlib import Path
            from .workflow import update_version
            config = json.loads(Path(args.config).read_text(encoding='utf-8')) if args.config else None
            result = update_version(args.store, args.id, args.data, operation=args.command,
                                    settings=training_settings(args), config=config,
                                    hidden_sizes=args.hidden_sizes, sheet=args.sheet)
            print(json.dumps(result, indent=2))
        elif args.command == 'compare-versions':
            import json
            from .workflow import compare_versions
            report = compare_versions(args.store, args.ids, args.control, sheet=args.sheet, history=args.history)
            print(json.dumps({'run_id': report['run_id'], 'output_path': report['output_path'], 'models': report['models']}, indent=2))
        elif args.command in ('versions', 'runs', 'show-version', 'export-metrics', 'predict-version', 'recover-run'):
            import json
            from .storage import Store
            from .workflow import export_metrics, predict_version
            store = Store(args.store)
            if args.command == 'versions':
                result = store.versions()
            elif args.command == 'runs':
                result = store.runs()
            elif args.command == 'show-version':
                result = store.version(args.id)[2]
            elif args.command == 'export-metrics':
                result = str(export_metrics(args.store, args.id, args.output))
            elif args.command == 'recover-run':
                store.recover(args.id)
                result = {'run_id': args.id, 'status': 'failed'}
            else:
                result = predict_version(args.store, args.id, args.data, args.output, months=args.months,
                                         sheet=args.sheet, export_format=args.format)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.command == 'compare':
            from .workflow import compare_service
            report = compare_service(args.scenario, args.data, args.output, settings=training_settings(args),
                                     hidden_sizes=args.hidden_sizes, months=args.months)
            print('Recommended by validation:', report['recommended_model'])
            print('Report:', args.output+'/comparison.html')
        elif args.command == 'inspect-table':
            from .data import preview, sheets
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
            from .workflow import train_file
            config = json.loads(Path(args.config).read_text(encoding='utf-8'))
            if args.seed is not None:
                config['seed'] = args.seed
            model = train_file(args.file, args.output, config, sheet=args.sheet, settings=training_settings(args),
                               hidden_sizes=tuple(args.hidden_sizes or DEFAULT_HIDDEN_SIZES), model_kind=args.model_kind)
            print('Recommended by validation:', model.report['recommended_model'])
            print('Report:', args.output+'/comparison.html')
        elif args.command == 'predict-table':
            from .workflow import predict_file
            predict_file(args.file, args.model, args.output, sheet=args.sheet)
        elif args.command == 'prepare':
            from .scenarios.serviceflow import prepare
            prepare(args.data, args.output, args.as_of, mode=args.mode)
        else:
            run(args.scenario, args.data, args.output,
                train=args.command == 'train-predict',
                model=getattr(args, 'model', None), months=args.months,
                epochs=getattr(args, 'epochs', None), mode=args.mode)
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.error(str(exc))
