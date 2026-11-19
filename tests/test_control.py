from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from predictml.config import TrainingConfig
from predictml.workflow import train_file
from predictml.storage import Store
from predictml.workflow import update_version
from predictml.workflow import compare_versions

FAST = TrainingConfig(epochs=3, validation_interval=1, seed=4)


class ControlTests(unittest.TestCase):
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
        from predictml.evaluation import assert_control_independent
        with self.assertRaisesRegex(ValueError,'unavailable'):
            assert_control_independent({'policy':'feature-observation-v1','used_hashes':[]},[{'version_id':'old'}])

    def test_temporal_control_and_inherited_data_are_rejected(self):
        from predictml.evaluation import assert_control_independent, merge_provenance
        current=dict(policy='service-month-v1',complete=True,used_hashes=['new'],first_period=20,last_period=30)
        parent=dict(policy='service-month-v1',complete=True,used_hashes=['old'],first_period=1,last_period=40)
        merged=merge_provenance(current,parent,True)
        self.assertEqual(merged['last_period'],40)
        self.assertIn('old',merged['used_hashes'])
        with self.assertRaisesRegex(ValueError,'follow'):
            assert_control_independent(dict(policy='service-month-v1',used_hashes=['unknown'],first_period=35),
                                       [dict(version_id='v',provenance=merged)])

