"""Demand → consumption → one purchase for the full horizon."""
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_HALF_UP
import json
from pathlib import Path


def decimal(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise ValueError(f'{field}: expected finite nonnegative number')
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f'{field}: invalid number') from exc
    if not result.is_finite() or result < 0:
        raise ValueError(f'{field}: expected finite nonnegative number')
    return result


def validate_catalog(materials, norms, service_ids=None):
    if not isinstance(materials, list) or not isinstance(norms, list):
        raise ValueError('catalog: expected JSON arrays')
    ids = set()
    for material in materials:
        required = {'id', 'title', 'current_quantity', 'minimal_quantity', 'price', 'unit', 'purchase_step'}
        if not isinstance(material, dict) or required - material.keys():
            raise ValueError('materials: required id/title/quantities/price/unit/purchase_step')
        mid = material['id']
        if type(mid) is not int or mid <= 0 or mid in ids:
            raise ValueError('materials.id: expected unique positive integer')
        ids.add(mid)
        for name in ('title', 'unit'):
            if not isinstance(material[name], str) or not material[name].strip():
                raise ValueError(f'materials.{name}: expected non-empty string')
        for name in ('current_quantity', 'minimal_quantity', 'price', 'purchase_step'):
            value = decimal(material[name], f'materials.{name}')
            if name == 'purchase_step' and not value > 0:
                raise ValueError('materials.purchase_step: must be positive')
    for norm in norms:
        if not isinstance(norm, dict) or {'service_id', 'material_id', 'required_quantity'} - norm.keys():
            raise ValueError('service_materials: required service_id/material_id/required_quantity')
        if type(norm['service_id']) is not int or norm['service_id'] <= 0:
            raise ValueError('service_materials.service_id: expected positive integer')
        if type(norm['material_id']) is not int or norm['material_id'] not in ids:
            raise ValueError('service_materials.material_id: unknown material')
        if service_ids is not None and norm['service_id'] not in service_ids:
            raise ValueError('service_materials.service_id: unknown service')
        decimal(norm['required_quantity'], 'service_materials.required_quantity')


def load_catalog(directory, service_ids=None):
    directory = Path(directory)
    try:
        materials = json.loads((directory/'materials.json').read_text(encoding='utf-8'))
        norms = json.loads((directory/'service_materials.json').read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError('catalog: invalid JSON') from exc
    validate_catalog(materials, norms, service_ids)
    return materials, norms


def consumption(demand, norms):
    """Demand rows carry expected counts; norms use each material's unit."""
    usage = {}
    for row in demand:
        count = decimal(row['predicted_bookings'], 'predicted_bookings')
        for norm in norms:
            if norm['service_id'] == row['service_id']:
                key = (norm['material_id'], row['year'], row['month'])
                usage[key] = usage.get(key, Decimal(0)) + count*decimal(norm['required_quantity'], 'required_quantity')
    return usage


def purchase(material, need):
    """Apply stock once, round quantity up by step, then calculate money."""
    missing = max(Decimal(0), need+decimal(material['minimal_quantity'], 'minimal_quantity')
                  -decimal(material['current_quantity'], 'current_quantity'))
    step = decimal(material['purchase_step'], 'purchase_step')
    if not step > 0:
        raise ValueError('purchase_step: must be positive')
    quantity = (missing/step).to_integral_value(rounding=ROUND_CEILING)*step
    cost = (quantity*decimal(material['price'], 'price')).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
    return quantity, cost


def calculate(demand, materials, norms):
    validate_catalog(materials, norms, {row['service_id'] for row in demand})
    periods = sorted({(row['year'], row['month']) for row in demand})
    usage = consumption(demand, norms)
    result = []
    for material in materials:
        mid = material['id']
        amounts = [usage.get((mid, y, m), Decimal(0)) for y, m in periods]
        need = sum(amounts, Decimal(0))
        quantity, cost = purchase(material, need)
        result.append(dict(material_id=mid, title=material['title'], unit=material['unit'],
                           purchase_step=float(decimal(material['purchase_step'], 'purchase_step')),
                           current_quantity=material['current_quantity'], minimal_quantity=material['minimal_quantity'],
                           unit_price=material['price'],
                           months=[dict(year=y, month=m, predicted_usage=float(v)) for (y, m), v in zip(periods, amounts)],
                           total_predicted_usage=float(need), purchase_qty=float(quantity), purchase_cost=float(cost)))
    return sorted(result, key=lambda row: (row['purchase_qty'] == 0, -row['purchase_cost']))
