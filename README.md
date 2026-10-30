# MOEX: данные облигаций и анализ Z-спредов

Дневной сбор TQCB, историческая база, анализ спредов и локальное приложение.

## Где что лежит

| Папка | Содержимое |
|---|---|
| [`moex_bonds/`](moex_bonds/) | Весь рабочий код: источники, сбор, хранение, анализ, интерфейс |
| [`config/`](config/) | Настройки расчёта |
| [`data/`](data/) | Актуальные дневные данные, история, опубликованный анализ |
| [`tests/`](tests/) | Тесты |
| [`docs/`](docs/) | Действующие инструкции и методология |
| [`archive/`](archive/) | Старые выгрузки, драфты, ревью, планы и резервные копии |

Начните с [карты архитектуры](docs/architecture.md).
Правила расчёта: [методология](docs/current_methodology.md).
Все параметры команд и форматы файлов: [подробное руководство](docs/operations.md).

## Запуск

Из `D:\z\moex_api_probe`:

```powershell
python -m pip install -r requirements.txt
python main.py
python history.py                 # Полная история: первый запуск или явная пересборка
python analyze_spreads.py
python -m streamlit run app.py --server.address 127.0.0.1
```

Для повседневного обновления достаточно `main.py` → `analyze_spreads.py`.
После первой полной загрузки истории дневной сбор сам догружает пропущенные даты.
Приложение читает готовые результаты; после обновления анализа обновите страницу.

Выходы по умолчанию:

- `data/daily_data/daily/bonds_daily.xlsx` — дневной отбор;
- `data/history_data/bonds_history.csv` — историческая база;
- `data/spread_analysis/` — сводки, таблицы и снимки `snapshots/YYYY-MM-DD`.

Параметры анализа находятся в [`config/spread_config.json`](config/spread_config.json).
Все пути по умолчанию привязаны к папке проекта. Корневые Python-файлы —
короткие команды запуска; реализация находится в `moex_bonds`.

## Проверка

```powershell
python -m unittest discover -s tests -t .
```

Существующее локальное окружение с зависимостями:

```powershell
.\.work\streamlit-env\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
.\.work\streamlit-env\Scripts\python.exe -m unittest discover -s tests -t .
```

Тесты используют временные данные. Контрольные суммы и карта переноса файлов
сохранены в [`archive/reorganization/`](archive/reorganization/).

`static/` содержит локальный шрифт и лицензию; `.streamlit/` — настройки интерфейса.
`.work/` — служебные окружения и временные материалы разработки.


## GitHub publication

`python main.py --publish` runs collection, history synchronization, analysis,
bundle validation and Git push. `python main.py --publish-retry` retries a saved
attempt without collection. Plain `main.py` keeps its original behavior.

The remote is https://github.com/PypsenishAL/Bonds-screener.git, configured in
`config/github_publish.json`. First configure the current directory as a Git
checkout tracking remote main, set the commit author, and sign into GitHub with
Git Credential Manager through your browser. Scheduled runs must use that same
Windows account. No token belongs in this project. The first code/data commit
must be reviewed separately; daily runs only stage `data/published`.

Full history, audit and prior local analysis snapshots remain local. Only the
current UI bundle is placed in `data/published`. Previous Git commits still
occupy repository storage. Dirty tracked code, unrelated staged changes or a
diverged remote stop delivery. Do not use force push to resolve this.

For Community Cloud select this repository, actual branch and `app.py`, and add
`BONDS_DATA_MODE = "published"` in app secrets. Locally the environment variable
of the same name takes precedence; default is `local`. Published mode validates
all bundle bytes and never falls back to raw local data. Dependencies are in
`requirements.txt`; collection/XLSX generation never run on the cloud server.

Pending state and diagnostics: `.work/publication/state.json` and
`.work/publication/last_result.json`. Exit codes: 20 setup, 21 collection/history,
22 analysis/bundle, 23 Git delivery, 24 lock contention. On lost push response
retry checks remote commit identity. An abandoned lock requires manual PID
inspection before removing it. Never delete source data to repair a delivery.

Tests: `python -m unittest discover -s tests -t .`. Offline tests do not prove
real write access, Linux peak RAM, or Community Cloud availability.
