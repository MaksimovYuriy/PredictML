# PredictML — выделенный модуль ServiceFlow

Самостоятельный Python-пакет: без Rails, Sidekiq, PostgreSQL и файлов исходного репозитория. Выполнены отделение по контрактам, их проверки и выделение общего ядра. Полный объём универсальной программы из ТЗ ещё не реализован.

## Установка

Python 3.12+; проверка выполнена в текущем Linux-окружении. Windows отдельно не проверен.

```bash
python -m venv .venv
# Linux:
source .venv/bin/activate
# Windows PowerShell вместо предыдущей команды:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

requirements.txt фиксирует проверенные версии основных библиотек и новых зависимостей табличного сценария; это пока не полный lock всех транзитивных зависимостей PyTorch. Текущие закреплённые numpy/scipy требуют Python 3.12+.

## Своя таблица CSV/XLSX

```bash
python -m predictml inspect-table --file examples/tabular/housing.xlsx --sheet training --rows 5
python -m predictml train-table --file examples/tabular/housing.xlsx --sheet training --config examples/tabular/housing-config.json --output artifacts/housing --epochs 50
python -m predictml predict-table --file examples/tabular/housing_new.csv --model artifacts/housing/model.joblib --output artifacts/housing/predictions.xlsx
```

Конфигурация задаёт цель, признаки, исключения и типы. Даты нужно выбрать явно как datetime. Пропуски признаков обрабатываются по train; строки без цели исключаются из обучения. Предобработчик, схема и веса сохраняются вместе в model.joblib. Новые таблицы проверяются по сохранённым именам и типам, неизвестные категории кодируются нулями; обучение при прогнозе не запускается. Загружайте только доверенные joblib-файлы.

В examples/tabular есть две разные синтетические схемы и их JSON-конфигурации. Для временного примера:

```bash
python -m predictml train-table --file examples/tabular/monthly.csv --config examples/tabular/monthly-config.json --output artifacts/monthly --epochs 50
```

Здесь цель уже подготовлена для указанного горизонта, признаки известны на дату origin. Система разделяет данные хронологически и исключает пересечение горизонтов на границах, но не сдвигает цель и не строит произвольные временные ряды автоматически. Автоматическая рекурсия остаётся в demand-v2. Подробности и API: [табличный контракт](docs/tabular-contract.md).

Это пока обучение MLP. Линейная модель, baseline и сравнение — итерация 5; Streamlit — итерация 8.

## Самостоятельный пример

Все данные examples/raw синтетические. Режим по умолчанию — demand-v2, только полные месяцы до as-of.

```bash
python -m predictml prepare --data examples/raw --output artifacts/data --as-of 2027-01-01
python -m predictml train-predict materials --data artifacts/data --output artifacts/material-v1 --months 3
python -m predictml train-predict price --data artifacts/data --output artifacts/price-v1
python -m predictml predict materials --data artifacts/data --output artifacts/prediction --model artifacts/material-v1/material_model.pth --months 2
```

Для быстрой проверки обучения добавьте `--epochs 50`; это не настройка для достижения заданного качества. По умолчанию сохранены 1500/1000 эпох. Каждый запуск использует отдельный каталог результатов. Старые версии автоматически не создаются, повторное использование каталога перезаписывает файлы.

as-of=2027-01-01 включает данные по декабрь 2026. as-of=2026-12-31 включает только данные по ноябрь: месяц самой границы всегда считается неполным и исключается.

## Входные данные

`prepare` получает четыре CSV (UTF-8, запятая, заголовок):

| Файл | Обязательные столбцы |
|---|---|
| notes.csv | service_id, client_id, status, start_at, total_price |
| services.csv | id, price, active |
| materials.csv | id, title, quantity, minimal_quantity, price, unit, purchase_step |
| service_materials.csv | service_id, material_id, required_quantity |

Завершённая запись: status=2 или completed. active: true/false либо 1/0. start_at — ISO-дата/время без часового пояса в календаре экспорта. Количества и нормативы задаются в unit материала; price — за единицу. purchase_step >0 задаёт шаг округления закупки вверх. Дробные количества допустимы.

Выход подготовки: monthly_demand.csv с полной месячной историей, preparation.json с датой отсечения, два JSON-справочника и price_analysis.csv. Для обучения спроса нужно минимум десять полных месяцев. Подробности — [контракт v2](docs/contracts-v2.md).

## Спрос, оценка и закупки

MLP прогнозирует количество завершённых записей без нормировки на среднее спроса по всей истории. Масштабирование признаков обучается только на train. Отрицательный выход ограничивается нулём. Рекурсивный прогноз 1–3 месяцев использует свои предыдущие прогнозы.

Лучшие веса выбираются по трёхмесячному рекурсивному прогнозу validation. Test оценивается отдельно от одной точки; дополнительно сохраняется явно обозначенная одношаговая оценка с доступной фактической историей. Baseline — последнее значение и среднее трёх; условия горизонта одинаковы. Это одно фиксированное тестовое окно, а не оценка по множеству временных окон.

Расход = прогноз × норматив, затем суммы по услугам/месяцам. Одна закупка покрывает весь горизонт: max(0, потребность + минимальный запас − остаток), округление вверх по purchase_step. Стоимость считается после округления количества, HALF_UP до двух знаков.

Выход: demand_predictions.json/CSV, material_predictions.json, material_model.pth, material_metrics.json и demand_test_predictions.csv. Метрики нового спроса — в количестве записей; неприменимые wMAPE/R² — null. Отчёт отдельно показывает границы выборок и качество для lead=1/2/3.

Checkpoint хранит агрегированную историю. Для прогноза после перезапуска достаточно модели и каталога с materials.json/service_materials.json. Если подана новая monthly_demand.csv, она должна вместе с preparation.json дополнять сохранённую историю полными месяцами того же набора услуг; прошлые строки менять нельзя. Статистики при прогнозе не переобучаются. Загружайте только доверенные checkpoint: формат PyTorch использует pickle.

Рекомендация цены остаётся воспроизведением исходного формульного правила; она не прогнозирует наблюдаемую будущую цену.

## Совместимость с ServiceFlow

Для старых экспортов и ratio-весов явно выбирайте legacy-parity:

```bash
python -m predictml prepare --data examples/raw --output artifacts/legacy-data --as-of 2026-12-31 --mode legacy-parity
python -m predictml train-predict materials --data artifacts/legacy-data --output artifacts/legacy-model --mode legacy-parity
python -m predictml predict materials --data artifacts/legacy-data --output artifacts/legacy-prediction --model artifacts/legacy-model/material_model.pth --mode legacy-parity
```

Этот режим намеренно сохраняет утечку среднего, включение месяца as-of и исходную методику оценки; он предназначен для проверки переноса, а не для достоверной оценки будущего спроса. [Контракт v1](docs/contracts.md). Старый checkpoint не применяется к v2 автоматически — необходимо новое обучение.

## Ядро и проверки

Python API: `from predictml.pipeline import run`; параметры соответствуют CLI, включая mode. Модели и статистики принадлежат отдельному вызову. Параллельные прогнозы с отдельными каталогами проверены; воспроизводимость параллельного обучения не гарантируется из-за общего генератора случайных чисел PyTorch.

Общая MLP — models/mlp.py, тренер — training.py, загрузчик — inference.py. Исправленный спрос — demand.py/demand_pipeline.py, закупки — purchasing.py. scenarios/ содержит исходные предметные алгоритмы для совместимости. Четыре неизменённых исходных Python-скрипта сохранены только в tests/reference/python. Контрольные SHA-256 — docs/source-manifest.json. Ядро не зависит от Streamlit.

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

Тесты можно запускать и через `python -m unittest discover -s tests -v`. Для проверки исходных Ruby-вычислений нужен Ruby; при его отсутствии эта проверка явно пропускается. Rails/PostgreSQL для тестов не нужны.

Отчёты: [итерация 1](docs/iteration-1.md), [итерация 2](docs/iteration-2.md), [итерация 3](docs/iteration-3.md), [итерация 4](docs/iteration-4.md). ServiceFlow_back не изменяется, реальные данные и исходные checkpoint не включены.

Следующие этапы: линейная модель и единое сравнение, полноценные версии/SQLite, дообучение, Streamlit. Полное ТЗ ещё не завершено.
