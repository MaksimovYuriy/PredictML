from predictml.demo import examples_dir
import json
import tempfile
import unittest
from pathlib import Path
from predictml.scenarios.serviceflow import prepare_legacy as prepare


class Contracts(unittest.TestCase):
    def test_preparation_known_values(self):
        with tempfile.TemporaryDirectory() as directory:
            prepare(examples_dir()/'raw', directory, '2026-12-31')
            import pandas as pd
            root = Path(directory)
            demand = pd.read_csv(root/'material_forecast.csv')
            self.assertEqual(len(demand), 90)
            row = demand[demand.service_id == 1].iloc[0]
            self.assertEqual(row.total_bookings, 6)
            self.assertAlmostEqual(row.avg_bookings_last_3m,
                                   (row.bookings_lag_1+row.bookings_lag_2+row.bookings_lag_3)/3, places=4)
            price = pd.read_csv(root/'price_analysis.csv').iloc[0]
            # January: service 1 has 7 visits, service 2 has 8;
            # all visits are in each client's first completed calendar month.
            self.assertEqual(price.target_price, 1013.75)
            self.assertEqual(json.loads((root/'materials.json').read_text())[0]['current_quantity'], 10)

    def test_invalid_horizon(self):
        from predictml.workflow import run as current_run

        def run(*args, **kwargs):
            return current_run(*args, mode="legacy-parity", **kwargs)
        with self.assertRaisesRegex(ValueError, 'months'):
            run('materials', '.', '.', months=4)


if __name__ == '__main__':
    unittest.main()
