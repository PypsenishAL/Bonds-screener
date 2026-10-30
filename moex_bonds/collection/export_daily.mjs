// Public Artifact Tool API only. Called by daily.py with the bundled runtime.
import fs from 'node:fs/promises';
import path from 'node:path';
const { Workbook, SpreadsheetFile } = await import(process.env.BONDS_ARTIFACT_MODULE);
const outputDir = path.resolve(process.argv[2]);
const supportDir = path.join(outputDir, '_support');
const data = JSON.parse(await fs.readFile(path.join(supportDir, 'daily_dataset.json'), 'utf8'));
const wb = Workbook.create();
const m = data.metadata;

function col(n) {
  let s = '';
  while (n) { n--; s = String.fromCharCode(65 + n % 26) + s; n = Math.floor(n / 26); }
  return s;
}
function tableSheet(name, title, notes, columns, rows, tableName) {
  const sh = wb.worksheets.add(name);
  const end = col(columns.length);
  sh.showGridLines = false;
  sh.getRange(`A1:${end}${Math.max(rows.length + 8, 10)}`).format.font = { name: 'Arial', size: 10, color: '#202B38' };
  sh.getRange('A2').values = [[title]];
  sh.getRange('A2').format.font = { size: 14, bold: true };
  for (let i = 0; i < notes.length; i++) {
    sh.getCell(i + 2, 0).values = [[notes[i]]];
    sh.getCell(i + 2, 0).format.font = { size: 10, color: '#536174' };
  }
  const headerRow = notes.length + 5;
  sh.getRange(`A${headerRow}:${end}${headerRow}`).values = [columns.map(c => c[1])];
  if (rows.length) {
    sh.getRange(`A${headerRow + 1}:${end}${headerRow + rows.length}`).values = rows.map(row => columns.map(([key, , type]) => {
      const v = row[key] ?? null;
      if (key === 'SMARTLAB_AT' && v) {
        // ISO datetime strings are auto-coerced to UTC by this runtime even
        // in text-formatted cells. Keep an explicit local-time text label.
        return `${v.slice(8, 10)}.${v.slice(5, 7)}.${v.slice(0, 4)} ${v.slice(11, 19)} МСК`;
      }
      // Numeric Excel serials survive export as typed dates; Date objects in
      // this runtime's block setter are otherwise serialised to ISO text.
      return type === 'date' && v ? (Date.parse(`${v}T00:00:00Z`) - Date.UTC(1899, 11, 30)) / 86400000 : v;
    }));
    const t = sh.tables.add(`A${headerRow}:${end}${headerRow + rows.length}`, true, tableName);
    t.showFilterButton = true;
  } else {
    sh.getRange(`A${headerRow + 1}`).values = [['Нет выпусков, прошедших фильтры. Причины — на листе Исключены.']];
  }
  sh.getRange(`A${headerRow}:${end}${headerRow}`).format = {
    fill: '#253D59', font: { name: 'Arial', size: 10, bold: true, color: '#FFFFFF' },
    wrapText: true, horizontalAlignment: 'center', verticalAlignment: 'center', rowHeight: 46,
  };
  if (rows.length) sh.getRange(`A${headerRow + 1}:${end}${headerRow + rows.length}`).format.rowHeight = 28;
  columns.forEach(([key, , type], i) => {
    const c = col(i + 1);
    const rng = sh.getRange(`${c}${headerRow + 1}:${c}${headerRow + Math.max(rows.length, 1)}`);
    sh.getRange(`${c}1:${c}${headerRow + Math.max(rows.length, 1)}`).format.columnWidth =
      key === 'SECID' ? 18 : ['NAME', 'ISSUER', 'SECTOR'].includes(key) ? 25 :
      key === 'DATA_GAPS' || key === 'reasons' ? 80 : key.endsWith('URL') ? 62 : key.endsWith('_AT') ? 36 : 20;
    rng.format.verticalAlignment = 'center';
    if (type === 'date') rng.setNumberFormat('dd/mm/yyyy');
    else if (type === 'number') rng.setNumberFormat('#,##0.00');
    else if (type === 'integer') rng.setNumberFormat('#,##0');
    else rng.setNumberFormat('@');
    rng.format.horizontalAlignment = ['number', 'integer'].includes(type) ? 'right' : 'left';
    if (key === 'DATA_GAPS' || key === 'reasons') {
      rng.format.wrapText = true;
      if (rows.length) rng.format.autofitRows();
    }
  });
  sh.freezePanes.freezeRows(headerRow);
  sh.freezePanes.freezeColumns(2);
  return { sh, headerRow };
}

const summary = tableSheet('Итог', `Корпоративные облигации • ${m.trade_date}`, [
  `${m.scope}. Отобрано ${m.final_count} из ${m.input_count}; исключено ${m.excluded_count}.`,
  'Цена и YTM — по WAPRICE; Z-spread — п.п.; G-spread — б.п. по цене расчёта MOEX yields.',
  'Рейтинг, сектор и эмитент — текущая справка Smart-Lab; дата получения указана в строке.',
], data.columns, data.rows, 'BondsDaily');
summary.sh.tabColor = '#253D59';
const exColumns = [['SECID', 'SECID', 'text'], ['NAME', 'Название бумаги', 'text'],
  ['RATING', 'Рейтинг Smart-Lab', 'text'], ['reasons', 'Причины исключения', 'text'],
  ['SMARTLAB_AT', 'Справка получена (МСК)', 'text'], ['SMARTLAB_URL', 'Карточка Smart-Lab', 'text']];
const excluded = tableSheet('Исключены', `Исключённые выпуски • ${m.trade_date}`, [
  'Все причины сохранены по SECID. Пустая справка не считается выполнением условия отбора.',
], exColumns, data.excluded, 'ExcludedBonds');

const dictionary = {
  SECID: ['MOEX securities.SECID', 'Ключ выпуска; одна строка на SECID, выбранный режим TQCB.'],
  NAME: ['MOEX securities.SHORTNAME', 'Название бумаги из текущего справочника исходного запуска.'],
  ISSUER: ['Smart-Lab h1', 'Текст после «Облигация», до распознанного кода серии; юридическое наименование отдельно не подставляется.'],
  RATING: ['Smart-Lab «Кредитный рейтинг»', 'Текущее значение. BB включительно, BB- исключён. При нескольких рейтингах — минимальный. Национальные ru/(RU)/.ru нормализуются; исходный текст сохранён.'],
  SECTOR: ['Smart-Lab «Сектор»', 'Текущая справка, не восстановленная на историческую дату.'],
  WAPRICE: ['MOEX history.WAPRICE', 'Средневзвешенная чистая цена в процентах текущего номинала; 100 означает 100%.'],
  ACCINT: ['MOEX history.ACCINT', 'Дневной НКД в рублях на одну бумагу. Не подменяется текущим ACCRUEDINT или yields.ACCINT на дату расчётов.'],
  YTM_WAPRICE: ['MOEX history_yields.EFFECTIVEYIELDWAPRICE', 'Только YIELDDATETYPE=MATDATE, YIELDDATE=погашение, свежая TRADEMOMENT и совпадающая WAPRICE. Проценты годовых, 15 означает 15%.'],
  ZSPREADATWAPRICE: ['MOEX history.ZSPREADATWAPRICE', 'Z-spread к RUONIA по средневзвешенной цене, процентные пункты; 0.76 п.п. = 76 б.п. Не ZSPREADBP.'],
  ZSPREAD: ['MOEX history.ZSPREAD', 'Z-spread к RUONIA по последней цене, процентные пункты. Отдельно от ZSPREADATWAPRICE.'],
  DURATION_DAYS: ['history_yields.DURATIONWAPRICE; иначе history.DURATION', 'Биржевая дюрация в днях. Не подписывается как модифицированная.'],
  DURATION_YEARS: ['DURATION_DAYS / 365', 'Перевод дней в годы; не расчёт модифицированной дюрации.'],
  MATDATE: ['MOEX history.MATDATE', 'Погашение не ранее даты данных + 3 календарных месяца, граница включается.'],
  VALUE_RUB: ['MOEX history.VALUE', 'Дневной оборот режима TQCB; RUB подтверждается FACEUNIT и CURRENCYID.'],
  GSPREADBP: ['MOEX history_yields.GSPREADBP', 'Опубликованный G-spread, б.п., по PRICE строки yields, не обязательно по WAPRICE. Только свежая строка к погашению.'],
  NEXTCOUPON: ['MOEX bondization.coupons.coupondate', 'Ближайшая дата строго после даты данных из полностью выгруженного календаря. Это расписание, полученное на дату запуска.'],
  COUPONPERCENT: ['MOEX history.COUPONPERCENT', 'Ставка, % годовых, за дату данных.'],
  COUPONVALUE: ['MOEX history.COUPONVALUE', 'Размер купона на одну бумагу, руб., за дату данных.'],
  FACEVALUE: ['MOEX history.FACEVALUE', 'Текущий непогашенный номинал на дату данных, руб.'],
  NUMTRADES: ['MOEX history.NUMTRADES', 'Число сделок выбранного режима за дату данных. Ноль не заменяется пропуском.'],
  VOLUME: ['MOEX history.VOLUME', 'Объём в бумагах за дату данных.'],
  ADV20_RUB: ['Среднее history.VALUE за 20 дат', 'В новом дневном режиме не загружается и остаётся пустым. В прежних сохранённых аудитах: среднее за 20 разных опубликованных дат, включая нули. Пропуски не заменяются нулями.'],
  CLOSE: ['MOEX history.CLOSE', 'Чистая цена закрытия, % номинала.'],
  DAYS_TO_MATURITY: ['MATDATE минус TRADEDATE', 'Календарные дни.'],
  TRADEDATE: ['run_manifest.trade_date; MOEX dates.till или --date', 'Единая опубликованная дата полных дневных итогов TQCB. Для прежних аудитов используется PREVDATE. Текущие котировки не подставляются.'],
  BOARDID: ['MOEX securities.BOARDID', 'Один выбранный режим, без суммирования разных режимов или сессий. История TRADINGSESSION=3.'],
  TRADE_STATUS: ['MOEX history.NUMTRADES и дата сделки', 'traded: сделки подтверждены; no_trades: явно ноль; no_history_record: записи нет; unknown_trade_count: число неизвестно; stale_trade_date: другая дата сделки. Без подтверждённых сделок цена, доходность, спреды и дюрация пусты.'],
  DATA_GAPS: ['Проверки полей и дат', 'Причина пропусков; пропуски остаются пустыми ячейками.'],
  SMARTLAB_AT: ['Локальное время получения HTTP-ответа', 'Московское время. Исторический рейтинг на дату торгов не подтверждается.'],
  SMARTLAB_URL: ['Штатный поиск Smart-Lab по SECID', 'Точное совпадение SECID результата поиска и заголовка карточки. HTML сохранён в _support/smartlab_raw.'],
};
const rules = [
  ['Отбор', 'RU ISIN + corporate_bond/exchange_bond; номинал и расчёты RUB/SUR; известный фиксированный BONDTYPE; рейтинг >= BB; срок >= 3 календарных месяцев.'],
  ['Оферты', 'Исключаются любые обнаруженные даты оферт, put/call, события offers, признаки оферты в BONDSUBTYPE и карточке Smart-Lab, включая прошлые. Календарь offers должен быть полным.'],
  ['Субординация', m.subordination_basis], ['Резидентство', m.country_basis],
  ['Область охвата', `${m.scope}. Источник: ${m.source_run}`],
  ['Справка MOEX', 'https://iss.moex.com/iss/reference/495'],
  ['Определение Z-spread', 'MOEX, методика с 22.06.2026, раздел 3.2: https://www.moex.com/files/4pmhxzvda2tkqs08jcb19jmedh'],
];
const fields = tableSheet('Поля', 'Поля, источники и правила отбора', [
  'Данные рынка относятся к одной дате; справка Smart-Lab получена позднее и имеет отдельную метку времени.',
], [['key', 'Поле', 'text'], ['source', 'Источник', 'text'], ['meaning', 'Значение и ограничения', 'text']],
  [...data.columns.map(([key, label]) => ({ key: `${label} [${key}]`, source: dictionary[key]?.[0] ?? '', meaning: dictionary[key]?.[1] ?? '' })),
    ...rules.map(([key, meaning]) => ({ key, source: '', meaning }))], 'FieldDictionary');
fields.sh.getRange('A1:A48').format.columnWidth = 50;
fields.sh.getRange('B1:B48').format.columnWidth = 59;
fields.sh.getRange('C1:C48').format.columnWidth = 110;
fields.sh.getRange('A7:C48').format.wrapText = true;
fields.sh.getRange('A7:C48').format.autofitRows();
wb.recalculate();
console.log((await wb.inspect({kind: 'table', range: `Итог!A${summary.headerRow}:J${summary.headerRow + Math.min(data.rows.length, 4)}`, tableMaxRows: 5, tableMaxCols: 10, maxChars: 3000})).ndjson);
console.log((await wb.inspect({kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!|#CALC!', options: {useRegex: true, maxResults: 20}, maxChars: 1000})).ndjson);
for (const [name, range, file] of [
  ['Итог', `A1:J${summary.headerRow + Math.max(data.rows.length, 1)}`, 'preview_main.png'],
  ['Итог', `K${summary.headerRow}:V${summary.headerRow + Math.max(data.rows.length, 1)}`, 'preview_metrics.png'],
  ['Итог', `W${summary.headerRow}:${col(data.columns.length)}${summary.headerRow + Math.max(data.rows.length, 1)}`, 'preview_quality.png'],
  ['Исключены', 'A1:D11', 'preview_excluded.png'],
  ['Поля', 'A1:C12', 'preview_fields.png'],
]) {
  const preview = await wb.render({ sheetName: name, range, scale: 1, format: 'png' });
  await fs.writeFile(path.join(supportDir, file), new Uint8Array(await preview.arrayBuffer()));
}
const staged = path.join(supportDir, 'pending_bonds_daily.xlsx');
await (await SpreadsheetFile.exportXlsx(wb)).save(staged);
await fs.rename(staged, path.join(outputDir, 'bonds_daily.xlsx'));
console.log(`Saved ${path.join(outputDir, 'bonds_daily.xlsx')}`);
