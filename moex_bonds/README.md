# Исходный код

Все рабочие модули находятся в этом пакете. Разделы организованы по назначению,
импорты используют полные имена `moex_bonds.<раздел>.<модуль>`.

| Раздел | Ответственность |
|---|---|
| `sources` | Внешние источники и их ответы |
| `collection` | Сценарии сбора, отбор, история и экспорт |
| `storage` | Файловые контракты, блокировки, безопасная публикация истории |
| `analysis` | Анализ сохранённых входов и публикация его результатов |
| `ui` | Чтение завершённых снимков и интерфейс |
| `paths.py` | Единственные определения стандартных папок и конфигурации |

Основные сценарии запускаются корневыми `main.py`, `history.py`,
`analyze_spreads.py`, `app.py`. Вспомогательные CLI:

```powershell
python -m moex_bonds.collection.daily --help
python -m moex_bonds.sources.cbr_zcyc --help
python -m moex_bonds.sources.cbr_ruonia --help
```

JavaScript-экспортёр `collection/export_daily.mjs` вызывается из `daily.py`;
он лежит рядом с потребителем. Прямое исполнение вложенных `.py` по файловому
пути не используется: запускайте пакетные модули через `python -m`.

Подробная [карта модулей и потока данных](../docs/architecture.md).


## GitHub publication

`moex_bonds/publication` prepares a validated current bundle in `data/published`
and sends it through HTTPS Git. Full history and audit remain local.
Commands: `python main.py --publish` and `python main.py --publish-retry`.
See [operations](../docs/operations.md) for setup and error recovery.
