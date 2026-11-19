# Проверка программы и сборки

Проверяемый профиль — Python 3.12, CPU, зависимости из `requirements.txt` и `requirements-dev.txt`. Используйте интерпретатор `.venv`; инструкции установки находятся в README.

## Тесты

```bash
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest -q
```

В набор входят контракты таблиц, train-only преобразования, MLP/linear/baseline, выбор по validation, причинные лаги, рекурсия, расчёт закупок, версии/целостность, обновление, независимый контроль, UI и загрузка старых обычных/сжатых joblib. `tests/reference` нужен для сравнения с исходными Python/Ruby-вычислениями; `tests/fixtures/pre_refactor` — для проверки старых моделей. Они не являются исходниками приложения для пособия.

## Сборка и установленный пакет

```bash
python -m build --wheel --sdist --no-isolation --outdir dist
python -m pip install --no-deps --force-reinstall dist/predictml_serviceflow-0.1.0-py3-none-any.whl
```

Для проверки wheel перейдите в другую рабочую папку без PYTHONPATH репозитория. Запустите оттуда тесты с абсолютным путём и `--import-mode=importlib`, затем сквозную проверку:

```bash
python /PATH/TO/PredictML/scripts/acceptance.py --output /PATH/TO/EMPTY/RESULTS
```

Скрипт экспортирует встроенные примеры, читает XLSX, обучает три табличных алгоритма, прогнозирует по версии в отдельном процессе, выполняет retrain/fine-tune, независимый контроль, экспорт метрик и оба предметных сценария. Короткие 3–5 эпох проверяют работоспособность, а не качество модели. Результаты создаются заново; хранить старые `artifacts`, `build` и `dist` в учебных исходниках не нужно.

Проверка выполняется также в `.github/workflows/acceptance.yml` для Linux и Windows. Наличие workflow не означает, что конкретный запуск CI уже состоялся.

## Браузерная проверка

AppTest интерфейса входит в обычный набор. Для отдельной проверки Chromium:

```bash
python -m pip install -r requirements-browser.txt
python -m playwright install chromium
python -m predictml ui
# В другом терминале, после demo --output examples:
python scripts/browser_acceptance.py --examples examples --output browser-results
```

Проверяются настоящая загрузка XLSX, выбор листа, обучение, скачивание CSV, загрузка новой таблицы и скачивание прогнозов XLSX. `browser-results` можно удалить после просмотра.

## Границы

Программа использует доверенные pickle/joblib, строго проверяет сохранённое окружение и не исправляет manifest скрытно. Не подтверждаются качество на реальном бизнесе, превосходство MLP, оптимальность цены или реальный SQL-экспорт. Нативная Windows проверяется отдельно от Linux/WSL. Исторические результаты приёмки и бинарные отчёты удалены из рабочего дерева; воспроизводимые проверки сохранены.
