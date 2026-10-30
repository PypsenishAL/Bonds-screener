# Рабочие данные

| Папка | Пишет | Читает |
|---|---|---|
| `daily_data/` | `main.py` | История, анализ, дополнительные поля интерфейса |
| `history_data/` | `history.py` и догрузка дневного сборщика | Анализ |
| `spread_analysis/` | `analyze_spreads.py` | `app.py` |

`daily_data` содержит последний дневной сбор и перезаписывается по прежним
правилам. `history_data` — постоянная база с опубликованными торговыми датами.
`spread_analysis/snapshots` хранит публикации анализа по датам.
`run_manifest.json`, `raw_responses` и `_support` относятся к своей папке:
их нельзя отделять от соответствующих таблиц.

Главная дневная книга — `daily_data/daily/bonds_daily.xlsx`.
Главная история — `history_data/bonds_history.csv`.
Главная сводка анализа — `spread_analysis/summary.csv`.
Их схемы и статусы описаны в [руководстве](../docs/operations.md).

Проверочные пересчёты запускайте с отдельным `--output`, например
`archive/validation/manual-check`, чтобы не заменять рабочую публикацию.


## GitHub publication

`moex_bonds/publication` prepares a validated current bundle in `data/published`
and sends it through HTTPS Git. Full history and audit remain local.
Commands: `python main.py --publish` and `python main.py --publish-retry`.
See [operations](../docs/operations.md) for setup and error recovery.
