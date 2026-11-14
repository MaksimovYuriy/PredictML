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

train-table сравнивает среднее train (baseline), линейную регрессию и MLP на одних выборках и одном предобработчике. В model.joblib по умолчанию остаётся MLP; --model-kind best выбирает минимум validation MSE. Также сохраняются отдельные model-linear.joblib, model-mlp.joblib и model-baseline.joblib. Сравнение: comparison.html, comparison.csv, comparison_predictions.csv и report.json. Test используется только для итоговой оценки.

Настройки: --hidden-sizes 64 32, --learning-rate 0.001, --batch-size 32, --epochs 200, --seed 42, --patience 20, --min-delta 0, --validation-interval 1. Флаг --disable-early-stopping отключает раннюю остановку. По умолчанию train-table/compare проверяют validation каждую эпоху и восстанавливают лучшие веса. Параметры и фактически выполненные эпохи записываются в отчёт.

## Самостоятельный пример

Все данные examples/raw синтетические. Режим по умолчанию — demand-v2, только полные месяцы до as-of.

```bash
python -m predictml prepare --data examples/raw --output artifacts/data --as-of 2027-01-01
python -m predictml train-predict materials --data artifacts/data --output artifacts/material-v1 --months 3
python -m predictml train-predict price --data artifacts/data --output artifacts/price-v1
python -m predictml predict materials --data artifacts/data --output artifacts/prediction --model artifacts/material-v1/material_model.pth --months 2
```

Для быстрой проверки обучения добавьте `--epochs 50`; это не настройка для достижения заданного качества. По умолчанию сохранены 1500/1000 эпох. Каждый запуск получает отдельный каталог в runs. Повторное использование --output сохраняет предыдущие версии; плоские файлы обновляются как копии последнего результата.

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

## Сравнение моделей спроса и цены

```bash
python -m predictml compare materials --data artifacts/data --output artifacts/compare-demand --epochs 200 --seed 42
python -m predictml compare price --data artifacts/data --output artifacts/compare-price --epochs 200 --seed 42
python -m predictml predict materials --data artifacts/data --output artifacts/linear-demand --model artifacts/compare-demand/material-linear.pth --months 3
python -m predictml predict price --data artifacts/data --output artifacts/linear-price --model artifacts/compare-price/price-linear.pth
```

Получаются четыре обученные модели: material-mlp.pth, material-linear.pth, price-mlp.pth, price-linear.pth. Baseline спроса — последнее значение и среднее трёх, цены — текущая цена. Для цены отдельно показан прямой расчёт исходного правила; он может закономерно оказаться точнее моделей.

Спрос сравнивается в одинаковой рекурсии 1–3 месяцев от одной точки. Цена оценивается на месячных строках в денежных единицах как воспроизведение формулы; это не оценка будущей рыночной цены или прибыли. Подробные условия — [отчёт итерации 5](docs/iteration-5.md).

Откройте comparison.html в выходном каталоге: таблица метрик, график, параметры и границы выборок. comparison.json сохраняет исходные данные отчёта; comparison.csv — метрики. Выбор recommended_model основан на validation MSE, а не на test. Неприменимые wMAPE/R² отображаются как неопределённые.

## Версии и история SQLite

Высокоуровневые команды обучения автоматически создают history.sqlite3 и неизменяемые комплекты в runs/<run-id>/versions/<version-id>. Повторное обучение в тот же --output добавляет версии. Старые плоские файлы остаются копиями последнего результата, архивом являются каталоги runs.

```bash
python -m predictml versions --store artifacts/housing
python -m predictml runs --store artifacts/housing
python -m predictml show-version --store artifacts/housing --id VERSION_ID
python -m predictml predict-version --store artifacts/housing --id VERSION_ID --data examples/tabular/housing_new.csv --output artifacts/housing-prediction --format both
python -m predictml export-metrics --store artifacts/housing --id VERSION_ID --output artifacts/housing-metrics.xlsx
```

VERSION_ID замените на ID из versions. Для спроса/цены выберите их хранилище, а --data указывает каталог подготовленных данных; для спроса достаточно также двух JSON-справочников, если применяется сохранённая история. --months задаёт горизонт спроса. Новый прогноз имеет свой run ID и ссылку на выбранную версию, обучение не выполняется.

Комплект содержит параметры, схему, цель, метрики, окружение и SHA-256. Перед загрузкой проверяются целостность и совместимость. Для сериализуемых библиотек требуется точное совпадение версий, для Python — major/minor. Не редактируйте архивные файлы вручную. Загружать следует только доверенные pickle/joblib-комплекты.

При жёстком прерывании, после проверки остановки процесса, можно пометить конкретный running-запуск прерванным: `python -m predictml recover-run --store artifacts/housing --id RUN_ID`. Завершённые версии эта команда не изменяет. Подробности: [контракт хранилища](docs/storage-contract.md).

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

Отчёты: [итерация 1](docs/iteration-1.md), [итерация 2](docs/iteration-2.md), [итерация 3](docs/iteration-3.md), [итерация 4](docs/iteration-4.md), [итерация 5](docs/iteration-5.md), [итерация 6](docs/iteration-6.md). ServiceFlow_back не изменяется, реальные данные и исходные checkpoint не включены.

Следующие этапы: переобучение/дообучение и сравнение обновлённых версий, Streamlit. Полное ТЗ ещё не завершено.
