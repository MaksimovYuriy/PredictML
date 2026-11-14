"""Independent common control evaluation of registered versions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .storage import Store, execute_run
from .storage.bundles import digest
from .storage.operations import export_frames
from .provenance import tabular_provenance, period_provenance, assert_control_independent, hashed
from .updates import family
from .metrics import regression_metrics
from .reporting import export_comparison
from .inference import Predictor
from .tabular.io import read_table
from .tabular.schema import numeric


def compare_versions(root, version_ids, control, *, sheet=None, history=None):
    if len(version_ids) < 2 or len(set(version_ids)) != len(version_ids):
        raise ValueError('select at least two distinct version IDs')
    store = Store(root)
    def worker(stage, run_id):
        loaded = [store.load(version) for version in version_ids]
        manifests, payloads = [item[1] for item in loaded], [item[2] for item in loaded]
        task, schema = manifests[0]['task'], manifests[0]['schema']
        if any(m['task'] != task or family(m['schema']) != family(schema) for m in manifests):
            raise ValueError('versions must have the same task, target and input feature schema')
        frame = read_table(control, sheet)
        if task == 'tabular':
            if schema['target'] not in frame:
                raise ValueError('control target column is required')
            actual = numeric(frame[schema['target']], schema['target'], allow_missing=False).to_numpy()
            provenance = tabular_provenance(frame, schema)
            assert_control_independent(provenance, manifests)
            predicted = [model.predict(frame) for model in payloads]
            identities = [dict(source_row=i) for i in range(len(frame))]
            conditions = dict(policy='same independent rows', units='original', schema=schema)
        elif task == 'price':
            from .config import PRICE
            from .contracts import PRICE_COLUMNS, number
            from .scenarios.price import prepare_features
            from .comparison import price_values
            if set(PRICE_COLUMNS)-set(frame):
                raise ValueError('control price table must contain all price_analysis columns')
            for name in PRICE_COLUMNS:
                number(frame, 'control', name, integer=name in ('service_id','year','month','total_bookings','is_active'),
                       positive=name in ('service_id','year','month','base_price'))
            if (frame.month > 12).any() or (frame.year > 9999).any() or frame.duplicated(['service_id','year','month']).any() or not frame.is_active.isin([0,1]).all() or (frame.first_visit_share > 1).any():
                raise ValueError('control price: invalid calendar, flags, share or duplicate service/month')
            provenance = period_provenance(frame)
            assert_control_independent(provenance, manifests)
            frame = prepare_features(frame)
            actual = frame.target_price.to_numpy()
            predicted = [price_values(frame, Predictor(state).predict(frame[list(PRICE.features)].values)) for state in payloads]
            identities = frame[['service_id', 'year', 'month']].astype(int).to_dict('records')
            conditions = dict(policy='monthly formula-recommendation reconstruction, clipping 85–115%', units='currency')
        else:
            from .demand import COLUMNS, validate_monthly, forecast
            if set(COLUMNS)-set(frame):
                raise ValueError('control demand requires service_id/year/month/total_bookings')
            for name in COLUMNS:
                frame[name] = numeric(frame[name], name, allow_missing=False)
                if (frame[name] % 1 != 0).any() or (frame[name] < (0 if name == 'total_bookings' else 1)).any():
                    raise ValueError(f'control {name}: invalid integer values')
            if (frame.month > 12).any() or (frame.year > 9999).any() or frame.duplicated(['service_id','year','month']).any():
                raise ValueError('control demand: invalid or duplicate month')
            frame['ym'] = frame.year*12+frame.month
            provenance = period_provenance(frame)
            assert_control_independent(provenance, manifests)
            contexts = [state.get('demand_context') for state in payloads]
            if not all(contexts):
                raise ValueError('independent demand comparison requires demand-v2')
            if history is None:
                candidates = [validate_monthly(pd.DataFrame(c['history'])) for c in contexts]
                observed = max(candidates, key=lambda f: f.ym.max())
            else:
                observed = validate_monthly(read_table(history))
            first, last = int(frame.ym.min()), int(frame.ym.max())
            horizon = last-first+1
            services = set(observed.service_id)
            if horizon not in (1,2,3) or observed.ym.max() != first-1 or set(frame.service_id) != services:
                raise ValueError('control must cover 1–3 months immediately after common history for the same services')
            if any({r['service_id'] for r in c['history']} != services for c in contexts):
                raise ValueError('version service sets differ')
            if any(set(group.ym) != set(range(first,last+1)) for _,group in frame.groupby('service_id')):
                raise ValueError('control demand: every service needs every control month')
            frame = frame.sort_values(['service_id','ym']).reset_index(drop=True)
            observed[COLUMNS].to_csv(stage/'history.csv', index=False)
            actual = frame.total_bookings.to_numpy()
            predicted = [[r['predicted_bookings'] for r in forecast(observed, Predictor(state), horizon)] for state in payloads]
            identities = frame[['service_id','year','month']].astype(int).to_dict('records')
            for identity, row in zip(identities, frame.itertuples()):
                identity['lead'] = int(row.ym-first+1)
            conditions = dict(policy='fixed-origin recursive, identical observed history', units='completed_bookings',
                              horizon=horizon, history_rows=len(observed), history_digest=hashed(observed[COLUMNS].astype(int).to_dict('records')),
                              origin=int(observed.ym.max()))
        if not len(frame):
            raise ValueError('control needs labelled observations')
        models, rows = {}, []
        for version, values in zip(version_ids, predicted):
            models[version] = dict(control=regression_metrics(actual, values))
            if task == 'materials':
                models[version]['by_lead'] = {str(lead): regression_metrics(
                    [a for a, row in zip(actual, identities) if row['lead'] == lead],
                    [p for p, row in zip(values, identities) if row['lead'] == lead]) for lead in range(1, horizon+1)}
            rows.extend(dict(**identity, actual=float(a), prediction=float(p), model=version)
                        for identity,a,p in zip(identities,actual,values))
        report = dict(run_id=run_id, version_ids=version_ids,
                      versions=[{k: m.get(k) for k in ('version_id','model_kind','parent_id','created_at','settings','architecture','update_operation')} for m in manifests],
                      control_file_sha256=digest(control),
                      control_rows=len(frame), control_provenance=provenance, conditions=conditions, models=models,
                      independence='checked against all supplied training/validation/test observations and inherited weights',
                      policy='evaluation only; no automatic selection or learning from control')
        frame.to_csv(stage/'control.csv', index=False)
        (stage/'control_comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),encoding='utf-8')
        export_comparison(stage, models, rows, conditions, {k:v for k,v in report.items() if k!='models'})
        export_frames(stage, {'predictions':pd.DataFrame(rows)}, models)
        return report
    report, run_id, _ = execute_run(root,'compare-versions',dict(version_ids=version_ids, control=str(control),
                                                               history=str(history) if history else None,sheet=sheet),worker,publish=False)
    return {**report,'output_path':str(store.root/'runs'/run_id)}
