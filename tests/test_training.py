import unittest
import numpy as np
import torch
from predictml.config import TrainingConfig
from predictml.training import fit_arrays


class TrainingTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def test_early_stopping_restores_best_weights(self):
        x = np.arange(8, dtype=np.float32).reshape(-1, 1)
        y = 2*x
        losses, snapshots = iter([5., 1., 3., 4., 9.]), []
        def objective(state):
            snapshots.append({k: v.clone() for k, v in state['model_state'].items()})
            return next(losses)
        log = {}
        weights, loss, _ = fit_arrays(x, y, x, y, x, (4,),
                                      TrainingConfig(epochs=10, validation_interval=1, patience=2),
                                      validation_evaluator=objective, mean=np.zeros(1), std=np.ones(1),
                                      features=['x'], training_log=log)
        self.assertEqual(log['epochs_ran'], 4)
        self.assertEqual(log['best_epoch'], 2)
        self.assertTrue(log['stopped_early'])
        self.assertEqual(loss, 1)
        for key in weights:
            torch.testing.assert_close(weights[key], snapshots[1][key], rtol=0, atol=0)

    def test_invalid_stopping_parameters(self):
        for values in ({'patience': 0}, {'min_delta': -1}, {'seed': -1}):
            with self.assertRaises(ValueError):
                TrainingConfig(epochs=50, **values)

