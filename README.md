# PredictML

Учебная программа для регрессионного прогнозирования: загрузка CSV/XLSX, подготовка признаков, обучение baseline/линейной регрессии/MLP, сравнение, сохранение моделей и прогноз. Предметные примеры — спрос на услуги, закупки материалов и формульная рекомендация цены.

## Установка и запуск

Python 3.12, локальная работа на CPU.

Linux / WSL:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m predictml ui
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
.venv\Scripts\python.exe -m predictml ui
```

Интерфейс открывается на `http://127.0.0.1:8501`; остановка — Ctrl+C. Он использует встроенный синтетический пример. Каталоги результатов создаются при работе и не входят в учебные исходники. Параметры рабочей папки и хранилища:

```bash
python -m predictml ui --workspace artifacts/work --store artifacts/models --port 8502
```

Далее `python` означает интерпретатор `.venv`. Нативная Windows требует отдельной проверки; Linux/WSL не заменяет её.

## Примеры данных

Единственная исходная копия примеров находится в `predictml/data/examples/` и входит в устанавливаемый пакет. Чтобы выполнять упражнения с отдельными файлами:

```bash
python -m predictml demo --output examples
```

Команда не перезаписывает непустой каталог. Папка `examples/` — создаваемая рабочая копия, она исключена из Git. Данные синтетические; работа с ними проверяет исполнение программы, а не доказывает качество на реальном бизнесе.

## Обучение и прогноз таблицы

```bash
python -m predictml inspect-table --file examples/tabular/housing.xlsx --sheet training --rows 5
python -m predictml train-table --file examples/tabular/housing.xlsx --sheet training --config examples/tabular/housing-config.json --output artifacts/housing --epochs 100 --seed 42 --model-kind best
python -m predictml predict-table --file examples/tabular/housing_new.csv --model artifacts/housing/model.joblib --output artifacts/housing/predictions.xlsx
```

Обучаются baseline, linear и MLP на одном разбиении. `best` выбирает по validation; test используется для итоговой оценки. Baseline возвращает среднее train и потому может быть константным. Число эпох относится к MLP: линейная регрессия решает другую задачу оптимизации и может обучиться очень быстро.

Результаты: `report.json`, `test_predictions.csv`, `comparison.html` и отдельные модели. В прогнозе новых данных без фактической цели качество ещё не измерено. Подробнее — [таблицы и Python API](docs/tabular-contract.md).

## Спрос, материалы и цена

Сначала экспортируйте встроенные примеры командой `demo`, затем:

```bash
python -m predictml prepare --data examples/raw --output artifacts/prepared --as-of 2027-01-01
python -m predictml compare materials --data artifacts/prepared --output artifacts/demand --epochs 200 --seed 42
python -m predictml compare price --data artifacts/prepared --output artifacts/price --epochs 200 --seed 42
```

Получаются четыре обученные модели: MLP и linear для спроса и цены. Спрос прогнозируется на 1–3 месяца с причинными лагами и рекурсией; закупка рассчитывается отдельно по нормативам, остаткам и шагу покупки. Подготовка исключает незавершённый месяц даты `as-of`.

Цена — рекомендация по заданной формуле, не прогноз будущей рыночной цены. Прямой расчёт формулы может оказаться точнее ML-моделей. Условия описаны в [контракте спроса](docs/contracts-v2.md) и [описании переноса](docs/transfer-report.md). Исторический режим `--mode legacy-parity` сохранён для сравнения с ServiceFlow; его [контракт](docs/contracts.md) отличается от основного режима.

## Версии и обновление

```bash
python -m predictml versions --store artifacts/housing
python -m predictml predict-version --store artifacts/housing --id VERSION_ID --data examples/tabular/housing_new.csv --format both
python -m predictml retrain --store artifacts/housing --id VERSION_ID --data examples/tabular/housing_update.csv --epochs 100 --seed 42
python -m predictml fine-tune --store artifacts/housing --id MLP_VERSION_ID --data examples/tabular/housing_update.csv --epochs 100 --seed 42
python -m predictml compare-versions --store artifacts/housing --ids ORIGINAL_ID UPDATED_ID --control examples/tabular/housing_control.csv
```

ID берутся из списка версий. Retrain обучает заново, fine-tune продолжает только MLP с сохранённым предобработчиком. Общий контроль должен быть независимым от ранее использованных наблюдений. Версии добавляются в историю, прежние комплекты не перезаписываются. Подробности — [хранение](docs/storage-contract.md) и [обновление/контроль](docs/updates-contract.md).

## Навигация и проверка

- [Назначение файлов и состав учебных исходников](docs/project-guide.md).
- [Архитектура и путь от данных до модели](docs/architecture.md).
- [Руководство интерфейса](docs/ui-guide.md).
- [Требования к трём модулям курса и тестовые вопросы](docs/course/decisions.md).
- [Тесты, сборка и сквозная проверка](docs/acceptance.md).

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Справка по всем командам: `python -m predictml --help`. Программа не требует Rails/PostgreSQL и не меняет ServiceFlow_back. Модели joblib/pickle должны происходить из доверенного источника. Реальные данные и исходные пользовательские модели в учебную поставку не входят.
