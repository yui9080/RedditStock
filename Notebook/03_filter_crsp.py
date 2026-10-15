"""Retain CRSP rows whose PERMNO occurs in completed 02_02 retained mentions."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time

import pandas as pd
import pyarrow.parquet as pq


COLUMNS = ['Ticker', 'permno', 'DlyCalDt', 'DlyPrc', 'DlyCap', 'DlyRet', 'DlyRetx', 'DlyVol']

def permno_keys(values):
    """Validate identifiers; never infer PERMNO from ticker labels."""
    keys = values.astype('string').str.strip()
    if not keys.str.fullmatch(r'[1-9][0-9]*').fillna(False).all():
        raise ValueError('Missing or invalid PERMNO')
    return keys.map(int)


def filtered_manifest(root, manifest=None):
    if manifest is not None:
        path = Path(manifest)
        return path if path.is_absolute() else root / path
    candidates = []
    for path in (root / 'Data/process/02_02_filtered_comment_ticker').glob('*/run_summary.json'):
        info = json.loads(path.read_text(encoding='utf-8'))
        if info.get('status') == 'complete' and info.get('review_unit') == 'permno' and (path.parent / 'comment_ticker_cleaned.parquet').is_file():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError('Run PERMNO-based 02_02 first')
    return max(candidates, key=lambda path: path.parent.name)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run(root=None, chunk_size=100_000, manifest=None):
    root = Path(root) if root is not None else next(
        p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'Data/process').is_dir()
    )
    source = root / 'Data/raw/stock/RAW_crsp.csv'
    manifest_path = filtered_manifest(root, manifest)
    manifest_bytes = manifest_path.read_bytes()
    upstream = json.loads(manifest_bytes)
    if upstream.get('status') != 'complete' or upstream.get('review_unit') != 'permno':
        raise ValueError('Require completed PERMNO-based 02_02')
    retained_source = manifest_path.parent / 'comment_ticker_cleaned.parquet'
    retained_stat = (retained_source.stat().st_size, retained_source.stat().st_mtime_ns)
    retained_hash = sha256_file(retained_source)
    review_path = Path(upstream['review_file'])
    if not review_path.is_absolute():
        review_path = root / review_path
    review_bytes = review_path.read_bytes()
    review_hash = hashlib.sha256(review_bytes).hexdigest()
    review = pd.read_csv(review_path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    if review.columns.tolist() != ['permno', 'ticker', 'times', 'del']:
        raise ValueError('Unexpected review columns')
    review_ids = permno_keys(review.permno)
    if not review_ids.is_unique:
        raise ValueError('Review PERMNOs must be unique')
    if not review['del'].isin(['0', '1']).all():
        raise ValueError('Review del must be 0 or 1')
    excluded = set(review_ids.loc[review['del'].eq('1')])
    if (upstream.get('status') != 'complete'
            or upstream.get('review_sha256') != review_hash
            or set(upstream.get('excluded_permnos', [])) != excluded):
        raise ValueError('Review differs from completed 02_02; rerun 02_02 first')
    retained = set()
    for batch in pq.ParquetFile(retained_source).iter_batches(columns=['permno'], batch_size=chunk_size):
        retained.update(permno_keys(batch.to_pandas().permno))
    if not retained or retained & excluded:
        raise ValueError('02_02 retained PERMNOs are empty or contain excluded PERMNOs')
    header = pd.read_csv(source, nrows=0).columns.tolist()
    if not set(COLUMNS) <= set(header):
        raise ValueError(f'Missing CRSP columns: {set(COLUMNS) - set(header)}')
    source_stat = (source.stat().st_size, source.stat().st_mtime_ns)
    source_hash = sha256_file(source)
    out = root / 'Data/process/02_03_filtered_crsp' / datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S_UTC')
    out.mkdir(parents=True, exist_ok=False)
    dest = out / 'crsp_cleaned.csv'
    temp = out / 'crsp_cleaned.csv.partial'
    started = time.perf_counter()
    counts = {'before_rows': 0, 'after_rows': 0, 'removed_rows': 0, 'blank_ticker_rows': 0}
    before_tickers, after_tickers, permnos = set(), set(), set()
    date_min = date_max = None
    # Read every field as text so prices, returns, missing markers, and IDs stay unchanged.
    with temp.open('w', encoding='utf-8', newline='') as stream:
        pd.DataFrame(columns=COLUMNS).to_csv(stream, index=False)
        for batch_no, frame in enumerate(pd.read_csv(
            source, usecols=COLUMNS, dtype=str, keep_default_na=False,
            encoding='utf-8-sig', chunksize=chunk_size
        ), 1):
            frame = frame[COLUMNS]
            # Normalize only the comparison key; keep original Ticker values in output.
            keys = frame.Ticker.str.strip().str.upper()
            ids = permno_keys(frame.permno)
            remove = ~ids.isin(retained)
            cleaned = frame.loc[~remove]
            counts['before_rows'] += len(frame)
            counts['after_rows'] += len(cleaned)
            counts['removed_rows'] += int(remove.sum())
            counts['blank_ticker_rows'] += int(keys.eq('').sum())
            before_tickers.update(keys[keys.ne('')])
            kept_keys = keys.loc[~remove]
            after_tickers.update(kept_keys[kept_keys.ne('')])
            permnos.update(ids.loc[~remove])
            dates = cleaned.loc[cleaned.DlyCalDt.ne(''), 'DlyCalDt']
            if len(dates):
                date_min = min(date_min, dates.min()) if date_min else dates.min()
                date_max = max(date_max, dates.max()) if date_max else dates.max()
            cleaned.to_csv(stream, index=False, header=False)
            if batch_no % 10 == 0:
                print(f'Processed {counts["before_rows"]:,} rows; retained {counts["after_rows"]:,}', flush=True)
    # Independently reread the complete output before publishing a successful run.
    checked_rows = 0
    for frame in pd.read_csv(temp, dtype=str, keep_default_na=False, chunksize=chunk_size):
        if frame.columns.tolist() != COLUMNS or not permno_keys(frame.permno).isin(retained).all():
            raise ValueError('Output validation failed')
        checked_rows += len(frame)
    if checked_rows != counts['after_rows'] or counts['before_rows'] != counts['after_rows'] + counts['removed_rows']:
        raise ValueError('Row counts do not reconcile')
    if review_path.read_bytes() != review_bytes or source_stat != (source.stat().st_size, source.stat().st_mtime_ns):
        raise ValueError('Inputs changed during processing')
    if (manifest_path.read_bytes() != manifest_bytes
            or retained_stat != (retained_source.stat().st_size, retained_source.stat().st_mtime_ns)
            or sha256_file(retained_source) != retained_hash):
        raise ValueError('02_02 inputs changed during processing')
    temp.replace(dest)
    summary = {
        'status': 'complete', 'source': str(source), 'source_sha256': source_hash,
        'review_file': str(review_path), 'review_sha256': review_hash,
        'upstream_manifest': str(manifest_path), 'columns': COLUMNS,
        'filter_rule': 'Retain only PERMNOs present in 02_02 retained mentions; preserve all CRSP dates, source values and row order.',
        'filter_unit': 'permno', 'retained_source': str(retained_source),
        'retained_source_sha256': retained_hash, 'retained_permnos': sorted(retained),
        'retained_permno_count': len(retained),
        'excluded_permnos': sorted(excluded), 'excluded_permno_count': len(excluded),
        **counts, 'before_unique_tickers': len(before_tickers),
        'after_unique_tickers': len(after_tickers), 'after_unique_permnos': len(permnos),
        'retained_permnos_not_found_in_crsp': sorted(retained - permnos),
        'after_date_min': date_min, 'after_date_max': date_max,
        'output': str(dest), 'output_sha256': sha256_file(dest),
        'validated_output_rows': checked_rows, 'processing_seconds': time.perf_counter() - started,
    }
    (out / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ['before_rows', 'after_rows', 'removed_rows', 'after_unique_tickers', 'output']}, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    run()
