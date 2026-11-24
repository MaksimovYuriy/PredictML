"""Reproducible module-3 experiment; run with the student's installed PredictML.

python run_experiment.py --output results/experiment-01
The destination must be new or empty. Data are synthetic, not real business data.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import numpy as np
import pandas as pd
import torch
from predictml.config import TrainingConfig
from predictml.data import read_table
from predictml.demo import export_examples
from predictml.scenarios.serviceflow import prepare
from predictml.storage import Store
from predictml.storage.bundles import environment
from predictml.workflow import (train_file, predict_version, update_version,
                                compare_versions, compare_service)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if root.exists() and any(root.iterdir()):
        parser.error('output must be a new or empty directory')
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    examples = export_examples(root/'examples')
    config = json.loads((examples/'tabular/housing-config.json').read_text())
    # Keep split seed 42 constant; settings.seed controls MLP initialization.
    config['seed'] = 42
    settings = TrainingConfig(epochs=600, validation_interval=1, patience=30, seed=42)
    table_root = root/'table-store'
    model = train_file(examples/'tabular/housing.xlsx', table_root, config,
                       sheet='training', settings=settings, model_kind='best')
    store = Store(table_root)
    versions = {r['model_kind']: r['id'] for r in store.versions()}
    assert set(versions) == {'baseline', 'linear', 'mlp'}
    source = read_table(examples/'tabular/housing.xlsx', 'training')
    train_rows = model.report['split']['parts']['train']['source_rows']
    _, _, baseline = store.load(versions['baseline'])
    expected_mean = pd.to_numeric(source.iloc[train_rows]['price']).mean()
    np.testing.assert_allclose(baseline.predict(source), expected_mean)
    prediction = predict_version(table_root, versions['mlp'],
                                 examples/'tabular/housing_new.csv', root/'new-prediction')
    internal = []
    for kind, details in model.report['models'].items():
        for partition in ('validation', 'test'):
            internal.append(dict(algorithm=kind, partition=partition,
                                 **details[partition]))
    update_settings = TrainingConfig(epochs=300, validation_interval=1,
                                     patience=30, seed=42)
    retrained = update_version(table_root, versions['mlp'],
                              examples/'tabular/housing_update.csv',
                              operation='retrain', settings=update_settings)
    tuned = update_version(table_root, versions['mlp'],
                          examples/'tabular/housing_update.csv',
                          operation='fine-tune', settings=update_settings)
    labels = {versions['baseline']: 'baseline', versions['linear']: 'linear',
              versions['mlp']: 'mlp-parent', retrained['version_id']: 'mlp-retrain',
              tuned['version_id']: 'mlp-fine-tune'}
    control = compare_versions(table_root, list(labels), examples/'tabular/housing_control.csv')
    control_rows = [dict(variant=labels[version], version_id=version,
                         **details['control']) for version, details in control['models'].items()]
    seed_rows = []
    for seed in (11, 23, 42):
        seeded = train_file(examples/'tabular/housing.csv', root/f'seed-{seed}', config,
                            settings=TrainingConfig(epochs=600, validation_interval=1,
                                                    patience=30, seed=seed), model_kind='mlp')
        details = seeded.report['models']['mlp']
        seed_rows.append(dict(initialization_seed=seed, split_seed=config['seed'],
                              validation_mae=details['validation']['mae'],
                              test_mae=details['test']['mae'],
                              best_epoch=details['best_epoch'], epochs_ran=details['epochs_ran']))
    seed_summary = dict(test_mae_mean=statistics.mean(r['test_mae'] for r in seed_rows),
                        test_mae_sample_std=statistics.stdev(r['test_mae'] for r in seed_rows),
                        selection='all prespecified seeds reported; no best-test-seed selection')
    prepared = root/'prepared'
    prepare(examples/'raw', prepared, '2027-01-01')
    scenario_rows, scenario_reports = [], {}
    for task in ('materials', 'price'):
        scenario_root = root/task
        report = compare_service(task, prepared, scenario_root,
                                 settings=TrainingConfig(epochs=200, validation_interval=1,
                                                         patience=30, seed=42), months=3)
        scenario_reports[task] = report
        for kind, details in report['models'].items():
            for partition in ('validation', 'test'):
                scenario_rows.append(dict(task=task, algorithm=kind, partition=partition,
                                           **details[partition]))
        for version in Store(scenario_root).versions():
            predict_version(scenario_root, version['id'], prepared, months=3)
    tables = dict(internal=pd.DataFrame(internal), control=pd.DataFrame(control_rows),
                  seeds=pd.DataFrame(seed_rows), scenarios=pd.DataFrame(scenario_rows))
    for name, frame in tables.items():
        frame.to_csv(root/f'{name}.csv', index=False)
    with pd.ExcelWriter(root/'experiment.xlsx', engine='openpyxl') as book:
        for name, frame in tables.items():
            frame.to_excel(book, sheet_name=name, index=False)
    summary = dict(environment=environment(), torch_threads=torch.get_num_threads(),
                   data_source='bundled synthetic examples', root=str(root),
                   training_settings=asdict(settings), update_settings=asdict(update_settings),
                   split=model.report['split'], dropped_missing_target=model.report['dropped_missing_target'],
                   baseline_mean=float(expected_mean), selected_model=model.report['selected_model'],
                   recommended_model=model.report['recommended_model'],
                   versions=labels, prediction=prediction, control_path=control['output_path'],
                   seed_summary=seed_summary,
                   scenarios={task: dict(recommended=report['recommended_model'],
                                          units=report['units']) for task, report in scenario_reports.items()},
                   status='passed')
    (root/'experiment.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                  allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
