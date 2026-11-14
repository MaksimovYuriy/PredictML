# Synthetic educational examples

All values are synthetic. They are not real property prices, business observations or external datasets.

- housing.csv / housing.xlsx (training sheet): numeric area, categorical district, explicit date built_at, numeric target price; includes missing feature/target values.
- housing_new.csv: rows without target, ready for prediction with the housing model.
- housing-config.json: selected/excluded columns, semantic types, seed.
- monthly.csv / monthly-config.json: prepared next-period numeric target at explicit monthly forecast origins. The demonstration assumes capacity is known at origin; no automatic target shift is applied.

The XLSX about sheet is informational; choose training for model training. Run commands in the repository README.

- housing_update.csv and housing_control.csv: separate synthetic rows for version updates and an independent control experiment. Do not train on housing_control.csv.
