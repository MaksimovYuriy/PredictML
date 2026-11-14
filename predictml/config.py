from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioConfig:
    features: tuple[str, ...]
    target: str
    hidden_sizes: tuple[int, ...]
    epochs: int


MATERIALS = ScenarioConfig(
    ('bookings_lag_1', 'bookings_lag_2', 'bookings_lag_3',
     'avg_bookings_last_3m', 'usage_trend', 'month_sin', 'month_cos'),
    'booking_ratio', (32, 16), 1500)
PRICE = ScenarioConfig(
    ('base_price', 'is_active', 'total_bookings', 'revenue', 'avg_price_fact',
     'first_visit_share', 'month_sin', 'month_cos',
     'avg_material_cost_per_service', 'material_cost_share'),
    'price_ratio', (64, 32, 16), 1000)
CONFIGS = {'materials': MATERIALS, 'price': PRICE}


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    learning_rate: float = 0.001
    batch_size: int = 32
    dropout: float = 0.1
    validation_interval: int = 50
    patience: int | None = None
    min_delta: float = 0.0
    seed: int | None = None

    def __post_init__(self):
        import math
        if self.patience is not None and (type(self.patience) is not int or self.patience < 1):
            raise ValueError('patience must be a positive number of validation checks')
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError('min_delta must be finite and nonnegative')
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed < 2**32):
            raise ValueError('seed must be in [0, 2**32)')
        if type(self.epochs) is not int or type(self.validation_interval) is not int or self.validation_interval <= 0 or self.epochs < self.validation_interval:
            raise ValueError('epochs must reach a positive validation_interval')
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError('batch_size must be a positive integer')
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate must be positive and finite')
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
