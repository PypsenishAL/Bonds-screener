"""Publish and retain only analysis results, isolated from their source files."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd

from moex_bonds.storage.daily_storage import checked_child, checked_output, output_lock
from moex_bonds.analysis.spread_model import DIRECTIONS
from moex_bonds.analysis.spread_classification import CANDIDATE_KEYS, CANDIDATE_FIELDS
from moex_bonds.analysis.spread_recovery import EPISODE_KEYS, HISTORY_KEYS


BASE_KEYS = {
    'summary': ['analysis_date', 'SECID'],
    'basket_members': ['analysis_date', 'SECID', 'comparison', 'member_SECID'],
    'comparison_history': ['analysis_date', 'observation_date', 'SECID', 'comparison'],
    'metrics': ['analysis_date', 'SECID', 'comparison', 'window'],
}
KEYS = dict(BASE_KEYS, recovery_episodes=EPISODE_KEYS, recovery_history=HISTORY_KEYS)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def validate_output(output, sources):
    output = checked_output(output)
    for source in sources:
        source = Path(source).resolve()
        if output == source or output in source.parents:
            raise ValueError('Результаты должны находиться отдельно от папок входных источников')
    return output


def build_summary(daily, metrics, candidates, config):
    fields = ['SECID', 'NAME', 'ISSUER', 'RATING', 'SECTOR', 'BOARDID', 'TRADEDATE',
              'TRADE_STATUS', 'DATA_GAPS', 'SMARTLAB_AT', 'ZSPREADATWAPRICE', 'DURATION_YEARS',
              'VALUE_RUB', 'adv20', 'adv_source', 'adv_n_values', 'adv_last_value_date', 'structured']
    summary = daily[[f for f in fields if f in daily]].copy()
    summary.insert(0, 'analysis_date', daily.TRADEDATE.iloc[0])
    index = metrics.set_index(['analysis_date', 'SECID', 'comparison', 'window'], verify_integrity=True)
    common = ['current_metric_bps', 'benchmark_bps', 'gap_bps', 'n_members', 'n_issuers', 'n_eff',
              'change_1d_bps', 'change_1d_reason', 'current_reason', 'evidence_status']
    historical = ['hist_mean_bps', 'hist_median_bps', 'excess_bps', 'percentile', 'n_obs',
                  'robust_score', 'mad_bps', 'hist_reason', 'scale_reason', 'quality_flags',
                  'assessment_status', 'upper_tail_anomaly', 'robust_anomaly', 'anomaly_basis',
                  'candidate_qualified', 'calendar_confirmed']
    additions = {}
    windows = config['windows']
    primary = config['classification']['primary_window']
    keys = list(zip(summary.analysis_date, summary.SECID))
    def comparison_values(kind, window, field):
        return index.reindex([(day, secid, kind, window) for day, secid in keys])[field].tolist()
    for kind in DIRECTIONS:
        for field in common:
            additions[f'{kind}__{field}'] = comparison_values(kind, primary, field)
        for window in windows:
            for field in historical:
                additions[f'{kind}__{window}__{field}'] = comparison_values(kind, window, field)
    candidate_index = candidates.set_index(CANDIDATE_KEYS, verify_integrity=True)
    for window in windows:
        selected = candidate_index.reindex([(day, secid, window) for day, secid in keys])
        for field in CANDIDATE_FIELDS:
            additions[f'candidate__{window}__{field.removeprefix("candidate_")}'] = selected[field].tolist()
    for field in CANDIDATE_FIELDS:
        additions[field] = additions[f'candidate__{primary}__{field.removeprefix("candidate_")}']
    return pd.concat([summary.reset_index(drop=True), pd.DataFrame(additions)], axis=1)


def write_csv(frame, path):
    """CSV booleans are lowercase, including legacy boolean cells after merging."""
    frame = frame.copy()
    boolean_fields = {'calendar_confirmed', 'upper_tail_anomaly', 'robust_anomaly', 'candidate_qualified',
                      'structured', 'liquidity_omitted', 'recovered_within_horizon'}
    for column in frame.columns:
        if column in boolean_fields or any(column.endswith('__' + field) for field in boolean_fields):
            def boolean_cell(value):
                if pd.isna(value):
                    return None
                if isinstance(value, (bool, np.bool_)):
                    return 'true' if value else 'false'
                if isinstance(value, str) and value.lower() in ('true', 'false'):
                    return value.lower()
                raise ValueError('Неверное логическое значение: ' + column)
            frame[column] = frame[column].map(boolean_cell)
    if 'result_schema_version' in frame:
        frame['result_schema_version'] = frame.result_schema_version.astype('Int64')
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def publish(output, tables, exclusions, metadata, report, sources):
    """Replace a whole analysis-date partition, including removed basket members.

    CSV renames are atomic individually; manifest.status is 'publishing' until
    every table and dated snapshot is committed. Readers must check the marker.
    A rerun repairs an interrupted publication from the same source inputs.
    """
    output = validate_output(output, sources)
    day = metadata['analysis_date']
    with output_lock(output):
        snapshots = checked_child(output, 'snapshots')
        snapshots.mkdir(exist_ok=True)
        snapshot = checked_child(snapshots, day)
        stage = checked_child(output, '.staging-' + uuid.uuid4().hex)
        stage.mkdir()
        staged_snapshot = stage / 'snapshot'
        staged_snapshot.mkdir()
        try:
            for name, keys in KEYS.items():
                frame = tables[name]
                if frame[keys].isna().any().any() or frame.duplicated(keys).any():
                    raise ValueError('Повторный или пустой ключ результата: ' + name)
                write_csv(frame, staged_snapshot / (name + '.csv'))
                target = checked_child(output, name + '.csv')
                if target.exists() and name != 'summary':
                    old = pd.read_csv(target, dtype={'SECID': str, 'member_SECID': str})
                    if old[keys].isna().any().any() or old.duplicated(keys).any():
                        raise ValueError('Повторный ключ существующих результатов: ' + name)
                    old = old[old.analysis_date.ne(day)]
                    merged = pd.concat([old.astype(object), frame.astype(object)], ignore_index=True) if len(old) else frame
                else:
                    merged = frame
                if merged.duplicated(keys).any():
                    raise ValueError('Повторный ключ при сохранении: ' + name)
                write_csv(merged, stage / (name + '.csv'))
            write_csv(exclusions, staged_snapshot / 'candidate_exclusions.csv')
            (staged_snapshot / 'report.md').write_text(report, encoding='utf-8')
            serialized = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)
            (staged_snapshot / 'run_manifest.json').write_text(serialized, encoding='utf-8')
            pending = dict(metadata, status='publishing')
            marker = checked_child(output, 'run_manifest.json')
            marker.write_text(json.dumps(pending, ensure_ascii=False, indent=2), encoding='utf-8')
            snapshot.mkdir(exist_ok=True)
            for path in staged_snapshot.iterdir():
                os.replace(path, checked_child(snapshot, path.name))
            for name in KEYS:
                os.replace(stage / (name + '.csv'), checked_child(output, name + '.csv'))
            (stage / 'run_manifest.json').write_text(serialized, encoding='utf-8')
            os.replace(stage / 'run_manifest.json', marker)
        finally:
            # The staging path was created above inside the checked result root.
            # Revalidate its absolute target before recursively removing it.
            safe_stage = checked_child(output, stage.name)
            if safe_stage.resolve().parent != output:
                raise ValueError('Небезопасный путь временных результатов')
            shutil.rmtree(safe_stage)
