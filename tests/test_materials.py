"""Causality, recursive evaluation, saved context and purchase semantics."""
import unittest
from predictml.scenarios.materials import calculate, validate_catalog


class PurchaseTests(unittest.TestCase):
    def test_decimal_units_steps_and_single_stock_application(self):
        materials = [dict(id=1, title='Liquid', unit='ml', purchase_step=0.25,
                          current_quantity=0.2, minimal_quantity=0.1, price=2.675)]
        norms = [dict(service_id=1, material_id=1, required_quantity=0.1),
                 dict(service_id=2, material_id=1, required_quantity=0.2)]
        demand = [dict(service_id=s, year=2027, month=m, predicted_bookings=1)
                  for s in (1, 2) for m in (1, 2, 3)]
        result = calculate(demand, materials, norms)[0]
        self.assertAlmostEqual(result['total_predicted_usage'], 0.9)
        self.assertEqual(result['purchase_qty'], 1)
        self.assertEqual(result['purchase_cost'], 2.68)
        self.assertEqual(result['unit'], 'ml')
        materials[0]['current_quantity'] = 2
        self.assertEqual(calculate(demand, materials, norms)[0]['purchase_qty'], 0)
        for bad in (0, -1, float('inf')):
            materials[0]['purchase_step'] = bad
            with self.assertRaisesRegex(ValueError, 'purchase_step'):
                validate_catalog(materials, norms)

