"""Filter CRSP by the manually reviewed ticker exclusions (02_03)."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time

import pandas as pd


COLUMNS = ['Ticker', 'permno', 'DlyCalDt', 'DlyPrc', 'DlyCap', 'DlyRet', 'DlyRetx', 'DlyVol']
REVIEW_PATH = 'Data/process/02_01_ticker_frequency_review/2026-10-05_08-24-30_UTC/top300_ticker_review.csv'
FILTER_MANIFEST = 'Data/process/02_02_filtered_comment_ticker/2026-10-05_14-08-44_UTC/run_summary.json'


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run(root=None, chunk_size=100_000):
    root = Path(root) if root is not None else next(
        p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'Data/process').is_dir()
    )
    source = root / 'Data/raw/stock/RAW_crsp.csv'
    review_path = root / REVIEW_PATH
    manifest_path = root / FILTER_MANIFEST
    review_bytes = review_path.read_bytes()
    review_hash = hashlib.sha256(review_bytes).hexdigest()
    review = pd.read_csv(review_path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    if review.columns.tolist() != ['ticker', 'mention_count', 'del']:
        raise ValueError('Unexpected review columns')
    if not review.ticker.is_unique or not review.ticker.ne('').all():
        raise ValueError('Review tickers must be unique and nonempty')
    if not review.ticker.eq(review.ticker.str.strip().str.upper()).all():
        raise ValueError('Review tickers must be uppercase without surrounding whitespace')
    if not review['del'].isin(['0', '1']).all():
        raise ValueError('Review del must be 0 or 1')
    excluded = set(review.loc[review['del'].eq('1'), 'ticker'])
    upstream = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (upstream.get('status') != 'complete'
            or upstream.get('review_sha256') != review_hash
            or set(upstream.get('excluded_tickers', [])) != excluded):
        raise ValueError('Review differs from completed 02_02; rerun 02_02 first')
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
            remove = keys.isin(excluded)
            cleaned = frame.loc[~remove]
            counts['before_rows'] += len(frame)
            counts['after_rows'] += len(cleaned)
            counts['removed_rows'] += int(remove.sum())
            counts['blank_ticker_rows'] += int(keys.eq('').sum())
            before_tickers.update(keys[keys.ne('')])
            kept_keys = keys.loc[~remove]
            after_tickers.update(kept_keys[kept_keys.ne('')])
            permnos.update(cleaned.loc[cleaned.permno.ne(''), 'permno'])
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
        if frame.columns.tolist() != COLUMNS or frame.Ticker.str.strip().str.upper().isin(excluded).any():
            raise ValueError('Output validation failed')
        checked_rows += len(frame)
    if checked_rows != counts['after_rows'] or counts['before_rows'] != counts['after_rows'] + counts['removed_rows']:
        raise ValueError('Row counts do not reconcile')
    if review_path.read_bytes() != review_bytes or source_stat != (source.stat().st_size, source.stat().st_mtime_ns):
        raise ValueError('Inputs changed during processing')
    temp.replace(dest)
    summary = {
        'status': 'complete', 'source': str(source), 'source_sha256': source_hash,
        'review_file': str(review_path), 'review_sha256': review_hash,
        'upstream_manifest': str(manifest_path), 'columns': COLUMNS,
        'filter_rule': 'Exclude only del=1 tickers; retain del=0 and unreviewed tickers. Compare trimmed uppercase keys; preserve source values and row order.',
        'excluded_tickers': sorted(excluded), 'excluded_ticker_count': len(excluded),
        **counts, 'before_unique_tickers': len(before_tickers),
        'after_unique_tickers': len(after_tickers), 'after_unique_permnos': len(permnos),
        'excluded_tickers_found': sorted(before_tickers & excluded),
        'excluded_tickers_not_found': sorted(excluded - before_tickers),
        'after_date_min': date_min, 'after_date_max': date_max,
        'output': str(dest), 'output_sha256': sha256_file(dest),
        'validated_output_rows': checked_rows, 'processing_seconds': time.perf_counter() - started,
    }
    (out / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ['before_rows', 'after_rows', 'removed_rows', 'after_unique_tickers', 'output']}, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    run()
