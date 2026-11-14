"""Run the release scenario with the active installed interpreter, outside source.

Usage: python scripts/acceptance.py --output artifacts/acceptance/scenario
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from predictml.config import TrainingConfig
from predictml.storage import Store
from predictml.updates import update_version
from predictml.version_comparison import compare_versions


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root=Path(args.output).resolve()
    if root.exists() and any(root.iterdir()):
        parser.error('acceptance output must be an empty or new directory')
    root.mkdir(parents=True,exist_ok=True)
    examples=root/'examples'
    def command(*values):
        subprocess.run([sys.executable,'-m','predictml',*map(str,values)],cwd=root,check=True)
    command('demo','--output',examples)
    command('inspect-table','--file',examples/'tabular/housing.xlsx','--sheet','training','--rows','2')
    table=root/'table'
    command('train-table','--file',examples/'tabular/housing.xlsx','--sheet','training',
            '--config',examples/'tabular/housing-config.json','--output',table,'--epochs','5','--seed','42')
    store=Store(table)
    parent=next(row['id'] for row in store.versions() if row['model_kind']=='mlp')
    command('predict-version','--store',table,'--id',parent,'--data',examples/'tabular/housing_new.csv','--format','both')
    settings=TrainingConfig(epochs=3,validation_interval=1,seed=42)
    retrained=update_version(table,parent,examples/'tabular/housing_update.csv',operation='retrain',settings=settings)
    tuned=update_version(table,parent,examples/'tabular/housing_update.csv',operation='fine-tune',settings=settings)
    control=compare_versions(table,[parent,retrained['version_id'],tuned['version_id']],examples/'tabular/housing_control.csv')
    command('export-metrics','--store',table,'--id',parent,'--output',root/'metrics.xlsx')
    prepared=root/'prepared'
    command('prepare','--data',examples/'raw','--output',prepared,'--as-of','2027-01-01')
    result=dict(python=sys.version,table_versions=len(store.versions()),table_runs=len(store.runs()),control_report=control['output_path'])
    for task in ('materials','price'):
        location=root/task
        command('compare',task,'--data',prepared,'--output',location,'--epochs','5','--seed','42')
        versions=Store(location).versions()
        result[task+'_versions']=len(versions)
        for version in versions:
            command('predict-version','--store',location,'--id',version['id'],'--data',prepared,'--months','2')
    result['status']='passed'
    (root/'acceptance.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
