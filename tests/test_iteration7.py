from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
import torch
from predictml.config import TrainingConfig
from predictml.tabular.application import train_file
from predictml.preparation import prepare
from predictml.comparison import compare_service
from predictml.storage import Store
from predictml.storage.bundles import digest
from predictml.updates import update_version
from predictml.version_comparison import compare_versions

ROOT = Path(__file__).resolve().parents[1]
FAST = TrainingConfig(epochs=3, validation_interval=1, seed=4)


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = self.root/'store'
        self.data = self.root/'data.csv'
        pd.DataFrame({'x':np.arange(20), 'y':2*np.arange(20)+3}).to_csv(self.data,index=False)
        train_file(self.data,self.store,{'target':'y'},settings=FAST)
        self.registry=Store(self.store)
        self.mlp=next(v['id'] for v in self.registry.versions() if v['model_kind']=='mlp')
        self.linear=next(v['id'] for v in self.registry.versions() if v['model_kind']=='linear')
        threads=torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,threads)

    def test_fine_tune_preserves_parent_preprocessor_architecture_and_lineage(self):
        bundle,_,parent=self.registry.load(self.mlp)
        checksum=digest(bundle/'model.joblib')
        preprocessing=joblib.hash(parent.preprocessor)
        data=self.root/'new.csv'
        pd.DataFrame({'x':np.arange(30,50),'y':np.arange(30,50)*3}).to_csv(data,index=False)
        from predictml.training import fit_arrays
        with patch('predictml.tabular.preprocessing.Preprocessor.fit',side_effect=AssertionError('must not refit')):
            with patch('predictml.updates.fit_arrays',wraps=fit_arrays) as fitting:
                result=update_version(self.store,self.mlp,data,operation='fine-tune',settings=FAST)
                initial=fitting.call_args.kwargs['initial_state']
                for key,value in parent.checkpoint['model_state'].items():
                    torch.testing.assert_close(initial[key],value)
        _,manifest,child=self.registry.load(result['version_id'])
        self.assertEqual(manifest['parent_id'],self.mlp)
        self.assertEqual(joblib.hash(child.preprocessor),preprocessing)
        self.assertEqual(child.checkpoint['hidden_sizes'],parent.checkpoint['hidden_sizes'])
        self.assertEqual(digest(bundle/'model.joblib'),checksum)
        self.assertEqual(manifest['optimizer_policy'],'fresh Adam; saved weights')
        self.assertTrue(set(manifest['provenance']['used_hashes']) >= set(self.registry.version(self.mlp)[2]['provenance']['used_hashes']))

    def test_retrain_fits_new_preprocessor_and_preserves_parent(self):
        data=self.root/'new.csv'
        pd.DataFrame({'x':np.arange(100,120),'y':np.arange(100,120)*2}).to_csv(data,index=False)
        result=update_version(self.store,self.linear,data,settings=FAST)
        _,manifest,child=self.registry.load(result['version_id'])
        _,_,parent=self.registry.load(self.linear)
        self.assertEqual(manifest['parent_id'],self.linear)
        self.assertNotEqual(joblib.hash(parent.preprocessor),joblib.hash(child.preprocessor))
        self.assertEqual(manifest['optimizer_policy'],'from scratch')

    def test_invalid_fine_tune_and_changed_target_new_model(self):
        with self.assertRaisesRegex(ValueError,'only for MLP'):
            update_version(self.store,self.linear,self.data,operation='fine-tune',settings=FAST)
        with self.assertRaisesRegex(ValueError,'architecture'):
            update_version(self.store,self.mlp,self.data,operation='fine-tune',settings=FAST,hidden_sizes=[3])
        with self.assertRaisesRegex(ValueError,'saved schema'):
            update_version(self.store,self.mlp,self.data,operation='fine-tune',settings=FAST,config={'target':'y'})
        frame=pd.read_csv(self.data);frame['new_target']=frame.y+10
        frame.to_csv(self.root/'changed.csv',index=False)
        result=update_version(self.store,self.linear,self.root/'changed.csv',settings=FAST,
                              config={'target':'new_target','features':['x']})
        self.assertIsNone(self.registry.version(result['version_id'])[2]['parent_id'])
        self.assertEqual(result['operation'],'new-model')

    def test_common_control_and_overlap_rejection(self):
        result=update_version(self.store,self.mlp,self.data,operation='fine-tune',settings=FAST)
        control=self.root/'control.csv'
        pd.DataFrame({'x':[100,101,102],'y':[203,205,207]}).to_csv(control,index=False)
        before=len(self.registry.versions())
        report=compare_versions(self.store,[self.mlp,result['version_id']],control)
        self.assertEqual(report['control_rows'],3)
        self.assertEqual(len(self.registry.versions()),before)
        self.assertTrue((Path(report['output_path'])/'comparison.html').exists())
        reused=pd.read_csv(self.data);reused.y+=999
        reused.to_csv(self.root/'overlap.csv',index=False)
        with self.assertRaisesRegex(ValueError,'overlaps'):
            compare_versions(self.store,[self.mlp,result['version_id']],self.root/'overlap.csv')

    def test_control_never_used_to_fit_and_unknown_provenance_rejected(self):
        control=self.root/'control.csv'
        pd.DataFrame({'x':[99,100],'y':[201,203]}).to_csv(control,index=False)
        with patch('predictml.training.fit_arrays',side_effect=AssertionError('no training')):
            report=compare_versions(self.store,[self.mlp,self.linear],control)
        self.assertEqual(set(report['models']),{self.mlp,self.linear})
        from predictml.provenance import assert_control_independent
        with self.assertRaisesRegex(ValueError,'unavailable'):
            assert_control_independent({'policy':'feature-observation-v1','used_hashes':[]},[{'version_id':'old'}])

    def test_checkpoint_updates_and_future_control(self):
        data=self.root/'prepared';prepare(ROOT/'examples/raw',data,'2027-01-01')
        for task in ('materials','price'):
            store=self.root/task
            compare_service(task,data,store,settings=FAST)
            registry=Store(store)
            parent=next(v['id'] for v in registry.versions() if v['model_kind']=='mlp')
            _,manifest,state=registry.load(parent)
            result=update_version(store,parent,data,operation='fine-tune',settings=FAST)
            _,child_manifest,child=registry.load(result['version_id'])
            self.assertEqual(state['feature_mean'],child['feature_mean'])
            self.assertEqual(state['feature_std'],child['feature_std'])
            self.assertEqual(child_manifest['parent_id'],parent)
            linear=next(v['id'] for v in registry.versions() if v['model_kind']=='linear')
            retrained=update_version(store,linear,data,operation='retrain',settings=FAST)
            self.assertEqual(registry.version(retrained['version_id'])[2]['parent_id'],linear)
            if task=='materials':
                rows=[dict(service_id=s,year=2027,month=m,total_bookings=10+s) for s in (1,2) for m in (1,2,3)]
                control=self.root/'demand-control.csv';pd.DataFrame(rows).to_csv(control,index=False)
            else:
                frame=pd.read_csv(data/'price_analysis.csv');frame=frame[frame.year==frame.year.max()].copy();frame.year=2027
                control=self.root/'price-control.csv';frame.to_csv(control,index=False)
            report=compare_versions(store,[parent,result['version_id']],control)
            self.assertEqual(len(report['models']),2)

    def test_temporal_control_and_inherited_data_are_rejected(self):
        from predictml.provenance import assert_control_independent, merge_provenance
        current=dict(policy='service-month-v1',complete=True,used_hashes=['new'],first_period=20,last_period=30)
        parent=dict(policy='service-month-v1',complete=True,used_hashes=['old'],first_period=1,last_period=40)
        merged=merge_provenance(current,parent,True)
        self.assertEqual(merged['last_period'],40)
        self.assertIn('old',merged['used_hashes'])
        with self.assertRaisesRegex(ValueError,'follow'):
            assert_control_independent(dict(policy='service-month-v1',used_hashes=['unknown'],first_period=35),
                                       [dict(version_id='v',provenance=merged)])

    def test_retrain_mlp_does_not_load_parent_weights(self):
        from predictml.regressors import fit_arrays
        with patch('predictml.regressors.fit_arrays',wraps=fit_arrays) as fitting:
            result=update_version(self.store,self.mlp,self.data,settings=FAST)
            self.assertTrue(fitting.called)
            self.assertTrue(all(call.kwargs.get('initial_state') is None for call in fitting.call_args_list))
        self.assertEqual(self.registry.version(result['version_id'])[2]['parent_id'],self.mlp)
