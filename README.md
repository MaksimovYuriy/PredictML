# PredictML — выделенный модуль ServiceFlow

Самостоятельный Python-пакет: без Rails, Sidekiq, PostgreSQL и файлов исходного репозитория. Это первый этап отделения по существующим контрактам, а не законченная универсальная программа из полного ТЗ.

## Установка

Python 3.10+; проверка выполнена в текущем Linux-окружении. Windows отдельно не проверен.

```bash
python -m venv .venv
# Linux:
source .venv/bin/activate
# Windows PowerShell вместо предыдущей команды:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

requirements.txt фиксирует версии трёх основных библиотек из проверенного окружения; это не полный lock транзитивных зависимостей.

## Полный самостоятельный пример

Все данные examples/raw синтетические. Они не скопированы из рабочей базы.

```bash
python -m predictml prepare --data examples/raw --output artifacts/data --as-of 2026-12-31
python -m predictml train-predict materials --data artifacts/data --output artifacts/material-v1 --months 3
python -m predictml train-predict price --data artifacts/data --output artifacts/price-v1
python -m predictml predict materials --data artifacts/data --output artifacts/prediction --model artifacts/material-v1/material_model.pth --months 2
python -m unittest discover -s tests -v
```

Для быстрой проверки обучения добавьте `--epochs 50`; это не настройка для достижения заданного качества. По умолчанию сохранены исходные 1500/1000 эпох. Каждому запуску задавайте отдельный выходной каталог: повторное использование каталога перезаписывает его файлы.

Python API: `from predictml.pipeline import run`; параметры соответствуют CLI. Внутри процесса вызовы выполняются последовательно: адаптер настраивает глобальные параметры исходных модулей и не предназначен для параллельных потоков.

## Контракты

`prepare` получает каталог с CSV (UTF-8, запятая, заголовок):

| Файл | Обязательные столбцы |
|---|---|
| notes.csv | service_id, client_id, status, start_at, total_price |
| services.csv | id, price, active |
| materials.csv | id, title, quantity, minimal_quantity, price |
| service_materials.csv | service_id, material_id, required_quantity |

Завершённая запись: status=2 или completed. active: true/false либо 1/0. start_at — дата/время без часового пояса в календаре экспорта БД. Идентификаторы числовые; справочники должны покрывать используемые связи. Цена услуги положительная. Количества и нормы задаются в согласованных единицах исходного ServiceFlow (исходный контракт не содержит отдельного поля единицы).

Выход `prepare`: material_forecast.csv, price_analysis.csv, materials.json и service_materials.json — существующие контракты ServiceFlow; точный состав полей описан в docs/audit.md. Можно пропустить подготовку и передать каталог уже готового экспорта в `train-predict`/`predict`.

Выход обучения: *_model.pth, *_metrics.json; выход прогноза: *_predictions.json. Веса загружаются только из доверенного источника: исходный формат PyTorch использует pickle. Внешние пути задаются аргументами; зависимости от расположения ServiceFlow нет.

## Что сохранено и что ещё предстоит

Четыре исходных скрипта находятся в predictml/legacy без изменения алгоритмов. Контрольные SHA-256 исходников — docs/source-manifest.json. Ruby-сервисы подготовки переведены в predictml/preparation.py. Проверена подготовка на известных значениях, оба сценария обучения/прогноза на синтетике и совпадение прогнозов материалов после загрузки в новом процессе. В итерации 1 добавлено сравнение обеих подготовок с исходными Ruby-сервисами на изолированном синтетическом примере; реальная БД не запускалась. Контракты и результаты: [docs/contracts.md](docs/contracts.md), [docs/iteration-1.md](docs/iteration-1.md).

Режим этого выделения — совместимость с исходной методикой: включает месяц as-of, использует среднее спроса по всей истории и сохраняет исходные ограничения отношения спроса. Известные утечка, несовпадение правил тренда и особенности оценки перечислены в docs/audit.md. Метрики этой версии нельзя считать независимой оценкой будущего многопериодного прогноза. Рекомендация цены воспроизводит формулу, не будущую наблюдаемую цену.

Это промежуточный перенос. Согласованные исключение неполных месяцев, исправление утечки, явные единицы/округление закупок, универсальные модели, UI, XLSX, версии/SQLite и дообучение остаются дальнейшими этапами. ServiceFlow_back не изменяется. Реальные данные и исходные checkpoint в пакет не включены.
