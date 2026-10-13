"""Monthly PERMNO network diagnostics and SNA.

Run: python Notebook/04_01_network_sna.py
Optional: --network-run PATH --crsp PATH --on-anomaly stop

Ticker edges use ONLY a unique ticker->PERMNO mapping observed in the preceding
six calendar months of CRSP. Invalid/unresolved records are warned and excluded
by default, and results are marked partial; --on-anomaly stop rejects the month.
Duplicate mapped pairs are all excluded, never summed: ticker-level counts do
not contain the thread IDs needed to deduplicate common threads across aliases.
Strength uses log1p(raw weight); clustering is the ordinary unweighted local
coefficient (zero for degree < 2), averaged over all nodes including isolates.
The percentile plot is a generic distribution, not an unidentified Figure 6.
Dependencies: pandas, numpy, pyarrow, scipy, matplotlib, threadpoolctl.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import argparse
import hashlib
import json
import math
import os
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from threadpoolctl import threadpool_limits
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'redditstock-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


NODE_SCHEMA = pa.schema([
    ('network_month', pa.string()), ('permno', pa.int64()),
    ('degree', pa.int64()), ('strength', pa.float64()), ('clustering', pa.float64()),
])
METRICS = [
    'number_of_nodes', 'number_of_edges', 'network_density', 'average_degree',
    'median_degree', 'maximum_degree', 'average_raw_edge_weight',
    'median_raw_edge_weight', 'raw_edge_weight_p90', 'raw_edge_weight_p99',
    'average_clustering_coefficient', 'largest_connected_component_share',
    'top_1pct_degree_share', 'top_5pct_degree_share',
]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def latest_run(parent, artifact=None):
    runs = []
    for manifest in parent.glob('*/run_summary.json'):
        info = json.loads(manifest.read_text(encoding='utf-8'))
        if info.get('status') == 'complete' and (artifact is None or (manifest.parent / artifact).exists()):
            runs.append(manifest.parent)
    if not runs:
        raise FileNotFoundError(f'No completed input run in {parent}')
    return max(runs, key=lambda p: p.name)


def crsp_monthly_mapping(path):
    """Observed monthly identities, without using future CRSP observations."""
    monthly = defaultdict(lambda: defaultdict(set))
    for chunk in pd.read_csv(path, usecols=['Ticker', 'permno', 'DlyCalDt'],
                             dtype=str, keep_default_na=False, chunksize=250_000):
        frame = chunk.drop_duplicates().copy()
        frame['Ticker'] = frame.Ticker.str.strip().str.upper()
        dates = pd.to_datetime(frame.DlyCalDt, format='%Y-%m-%d', errors='coerce')
        ids = pd.to_numeric(frame.permno, errors='coerce')
        valid = dates.notna() & ids.notna() & ids.gt(0) & ids.mod(1).eq(0) & frame.Ticker.ne('')
        if not valid.all():
            warnings.warn(f'CRSP mapping: {int((~valid).sum())} blank/invalid mapping rows omitted', stacklevel=2)
        pairs = pd.DataFrame({'month': dates[valid].dt.strftime('%Y-%m'),
                              'ticker': frame.loc[valid, 'Ticker'],
                              'permno': ids[valid].astype('int64')}).drop_duplicates()
        for month, ticker, permno in pairs.itertuples(index=False, name=None):
            monthly[month][ticker].add(permno)
    return monthly


def window_mapping(month, monthly):
    target = pd.Period(month, freq='M')
    result = defaultdict(set)
    for offset in range(6, 0, -1):
        for ticker, permnos in monthly.get(str(target - offset), {}).items():
            result[ticker].update(permnos)
    return result


def concentration(degree, fraction):
    total = degree.sum()
    if not len(degree) or total == 0:
        return 0.0
    k = max(1, math.ceil(fraction * len(degree)))
    return float(np.sort(degree)[-k:].sum() / total)


def graph_statistics(nodes, edges):
    nodes = np.asarray(sorted(nodes), dtype=np.int64)
    n, m = len(nodes), len(edges)
    i = np.searchsorted(nodes, edges.permno_i.to_numpy(dtype=np.int64))
    j = np.searchsorted(nodes, edges.permno_j.to_numpy(dtype=np.int64))
    raw = edges.n_common_threads.to_numpy(dtype=np.float64)
    transformed = np.log1p(raw)
    degree = np.bincount(np.concatenate([i, j]), minlength=n).astype(np.int64)
    strength = np.bincount(np.concatenate([i, j]), weights=np.concatenate([transformed, transformed]), minlength=n)
    adjacency = sparse.csr_matrix((np.ones(2 * m), (np.concatenate([i, j]), np.concatenate([j, i]))), shape=(n, n))
    clustering = np.zeros(n)
    if m:
        # Each adjacent-node intersection contributes twice the node's triangle count.
        # BLAS is faster for dense later networks; sparse multiplication for early ones.
        with threadpool_limits(limits=4, user_api='blas'):
            if n and 2 * m / (n * n) > 0.10:
                dense = adjacency.toarray()
                squares = dense @ dense
                closed = np.einsum('ij,ij->i', squares, dense)
            else:
                closed = np.asarray((adjacency @ adjacency).multiply(adjacency).sum(axis=1)).ravel()
        np.divide(closed, degree * (degree - 1), out=clustering, where=degree >= 2)
    if n:
        _, labels = connected_components(adjacency, directed=False)
        lcc = float(np.bincount(labels).max() / n)
    else:
        lcc = 0.0
    dist = dict.fromkeys(['mean', 'median', 'p90', 'p99', 'max'], None)
    if m:
        q = np.percentile(raw, [50, 90, 99], method='linear')
        dist = {'mean': float(raw.mean()), 'median': float(q[0]), 'p90': float(q[1]),
                'p99': float(q[2]), 'max': float(raw.max())}
    summary = {
        'number_of_nodes': n, 'number_of_edges': m,
        'network_density': 2 * m / (n * (n - 1)) if n > 1 else 0.0,
        'average_degree': float(degree.mean()) if n else 0.0,
        'median_degree': float(np.median(degree)) if n else 0.0,
        'maximum_degree': int(degree.max()) if n else 0,
        'average_raw_edge_weight': dist['mean'], 'median_raw_edge_weight': dist['median'],
        'raw_edge_weight_p90': dist['p90'], 'raw_edge_weight_p99': dist['p99'],
        'average_clustering_coefficient': float(clustering.mean()) if n else 0.0,
        'largest_connected_component_share': lcc,
        'top_1pct_degree_share': concentration(degree, .01),
        'top_5pct_degree_share': concentration(degree, .05),
    }
    if degree.sum() != 2 * m or not np.isclose(strength.sum(), 2 * transformed.sum()):
        raise ValueError('Graph degree/strength reconciliation failed')
    if not ((clustering >= -1e-10) & (clustering <= 1 + 1e-10)).all():
        raise ValueError('Clustering outside [0, 1]')
    node_frame = pd.DataFrame({'permno': nodes, 'degree': degree, 'strength': strength, 'clustering': clustering})
    return summary, node_frame, dist, raw


def validate_month(folder, mappings, warn):
    edges = pd.read_parquet(folder / 'edges.parquet')
    input_edges = len(edges)
    # Prefer point-in-time PERMNO identities when supplied by the rebuilt 04.
    ticker_mode = not {'permno_i', 'permno_j'} <= set(edges.columns) and {'ticker_a', 'ticker_b'} <= set(edges.columns)
    if 'n_common_threads' not in edges:
        if 'weight' not in edges:
            raise ValueError('Edges need n_common_threads (or upstream raw weight)')
        edges = edges.rename(columns={'weight': 'n_common_threads'})
    if ticker_mode:
        lookup = {ticker: next(iter(ids)) for ticker, ids in mappings.items() if len(ids) == 1}
        for col in ['ticker_a', 'ticker_b']:
            keys = edges[col].astype('string').str.strip().str.upper()
            unresolved = ~keys.isin(lookup)
            if unresolved.any():
                examples = keys.loc[unresolved].drop_duplicates().head(20).fillna('<missing>').tolist()
                warn('unresolved_edge_endpoint_' + col, int(unresolved.sum()), examples)
            edges['permno_i' if col == 'ticker_a' else 'permno_j'] = keys.map(lookup)
    elif not {'permno_i', 'permno_j'} <= set(edges.columns):
        raise ValueError('Unknown edge endpoint columns')
    for col in ['permno_i', 'permno_j', 'n_common_threads']:
        edges[col] = pd.to_numeric(edges[col], errors='coerce')
    finite = pd.Series(np.isfinite(edges[['permno_i', 'permno_j', 'n_common_threads']]).all(axis=1), index=edges.index)
    valid_ids = edges.permno_i.gt(0) & edges.permno_j.gt(0) & edges.permno_i.mod(1).eq(0) & edges.permno_j.mod(1).eq(0)
    valid_weights = edges.n_common_threads.gt(0) & edges.n_common_threads.mod(1).eq(0)
    invalid = ~(finite & valid_ids & valid_weights)
    if invalid.any(): warn('invalid_id_or_weight', int(invalid.sum()), edges.loc[invalid].head(5).fillna('').to_dict('records'))
    edges = edges.loc[~invalid, ['permno_i', 'permno_j', 'n_common_threads']].astype('int64')
    reverse = edges.permno_i.gt(edges.permno_j)
    if reverse.any() and not ticker_mode:
        warn('noncanonical_pair_order', int(reverse.sum()), 'Endpoints explicitly reordered; counts unchanged')
    values = np.sort(edges[['permno_i', 'permno_j']].to_numpy(), axis=1)
    edges[['permno_i', 'permno_j']] = values
    loops = edges.permno_i.eq(edges.permno_j)
    if loops.any(): warn('self_loop', int(loops.sum()), edges.loc[loops].head(5).to_dict('records'))
    edges = edges.loc[~loops]
    duplicates = edges.duplicated(['permno_i', 'permno_j'], keep=False)
    if duplicates.any():
        warn('duplicate_stock_pair', int(duplicates.sum()), edges.loc[duplicates].head(10).to_dict('records'))
    edges = edges.loc[~duplicates].copy()
    nodes = set(edges.permno_i) | set(edges.permno_j)
    node_path = folder / 'nodes.parquet'
    if node_path.exists():
        node_frame = pd.read_parquet(node_path)
        if 'permno' in node_frame:
            ids = pd.to_numeric(node_frame.permno, errors='coerce')
            valid = np.isfinite(ids) & ids.gt(0) & ids.mod(1).eq(0)
            if (~valid).any(): warn('invalid_node_permno', int((~valid).sum()), None)
            mapped = ids[valid].astype('int64')
            if mapped.duplicated().any(): warn('duplicate_node_permno', int(mapped.duplicated().sum()), None)
            nodes.update(mapped)
        elif 'ticker' in node_frame and ticker_mode:
            keys = node_frame.ticker.astype('string').str.strip().str.upper()
            mapped = keys.map(lookup)
            if mapped.isna().any():
                examples = [{'ticker': str(t), 'candidate_permnos': sorted(mappings.get(t, set()))}
                            for t in keys[mapped.isna()].drop_duplicates().head(20)]
                warn('unresolved_node_ticker', int(mapped.isna().sum()), examples)
            nodes.update(mapped.dropna().astype('int64'))
        else:
            warn('unknown_node_schema', len(node_frame), node_frame.columns.tolist())
    else:
        warn('missing_node_file', 1, 'Isolated nodes cannot be recovered from edges')
    if edges.duplicated(['permno_i', 'permno_j']).any() or not edges.permno_i.lt(edges.permno_j).all():
        raise ValueError('Final pair validation failed')
    return nodes, edges, input_edges


def plot_results(summary, percentiles, out):
    figures = out / 'figures'
    figures.mkdir()
    plots = [
        ('number_of_nodes', 'Number of nodes'), ('number_of_edges', 'Number of edges'),
        ('average_degree', 'Average degree'), ('network_density', 'Network density'),
        ('average_clustering_coefficient', 'Average clustering coefficient'),
        ('top_5pct_degree_share', 'Top 5% nodes share of total degree'),
    ]
    dates = pd.to_datetime(summary.network_month)
    for index, (column, label) in enumerate(plots, 1):
        fig, ax = plt.subplots(figsize=(10, 4))
        values = pd.to_numeric(summary[column], errors='coerce')
        ax.plot(dates, values, linewidth=1.4)
        partial = summary.status.eq('partial') & values.notna()
        ax.scatter(dates[partial], values[partial], s=9, color='darkorange', label='Partial: records excluded')
        if partial.any(): ax.legend(fontsize=8)
        ax.set(xlabel='Network month', ylabel=label, title=label)
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(figures / f'{index:02d}_{column}.png', dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    for month, curve in percentiles:
        ax.plot(np.arange(101), curve, color='steelblue', alpha=.15, linewidth=.7)
    if percentiles:
        median_curve = np.median(np.stack([curve for _, curve in percentiles]), axis=0)
        ax.plot(np.arange(101), median_curve, color='black', linewidth=1.8, label='Median across monthly percentile curves')
        ax.set_yscale('log')
        ax.legend(fontsize=8)
    ax.set(xlabel='Percentile of within-month stock-pair distribution', ylabel='Common-thread count (log axis)',
           title='Common-thread count percentile distribution (generic Figure 6 style)')
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(figures / '07_common_thread_percentile_distribution.png', dpi=180)
    plt.close(fig)


def self_check():
    edges = pd.DataFrame({'permno_i': [1, 1, 2], 'permno_j': [2, 3, 3], 'n_common_threads': [1, 2, 3]})
    summary, nodes, _, _ = graph_statistics({1, 2, 3, 4}, edges)
    assert summary['number_of_edges'] == 3 and summary['network_density'] == .5
    assert np.allclose(nodes.clustering, [1, 1, 1, 0])
    assert summary['largest_connected_component_share'] == .75
    assert np.isclose(nodes.loc[nodes.permno.eq(1), 'strength'].iloc[0], np.log1p(1) + np.log1p(2))
    assert summary['top_5pct_degree_share'] == 1 / 3
    empty = pd.DataFrame(columns=['permno_i', 'permno_j', 'n_common_threads'])
    assert graph_statistics({1, 2}, empty)[0]['largest_connected_component_share'] == .5
    mappings = {'2020-03': {'OLD': {1}}, '2020-04': {'A': {2}},
                '2020-09': {'A': {3}}, '2020-10': {'FUTURE': {4}}}
    assert dict(window_mapping('2020-10', mappings)) == {'A': {2, 3}}
    # Invalid weights, reverse duplicates, self loops and missing mappings must
    # produce explicit warnings and never leak into the resulting stock graph.
    with tempfile.TemporaryDirectory(prefix='redditstock-sna-check-') as directory:
        folder = Path(directory)
        pd.DataFrame({'permno_i': [1, 2, 3, 1, 2], 'permno_j': [2, 1, 3, 4, 4],
                      'n_common_threads': [1, 2, 1, 0, 3]}).to_parquet(folder / 'edges.parquet', index=False)
        pd.DataFrame({'permno': [1, 2, 3, 4, 5]}).to_parquet(folder / 'nodes.parquet', index=False)
        problems = []
        nodes, checked, count = validate_month(folder, {}, lambda kind, n, details: problems.append((kind, n)))
        assert count == 5 and nodes == {1, 2, 3, 4, 5}
        assert checked[['permno_i', 'permno_j']].values.tolist() == [[2, 4]]
        assert dict(problems) == {'invalid_id_or_weight': 1, 'noncanonical_pair_order': 1,
                                  'self_loop': 1, 'duplicate_stock_pair': 2}
        pd.DataFrame({'ticker_a': ['A', 'B', 'UNKNOWN', None], 'ticker_b': ['C', 'C', 'A', 'A'],
                      'weight': [1, 2, 3, 1]}).to_parquet(folder / 'edges.parquet', index=False)
        pd.DataFrame({'ticker': ['A', 'B', 'C', 'UNKNOWN']}).to_parquet(folder / 'nodes.parquet', index=False)
        problems = []
        nodes, checked, _ = validate_month(folder, {'A': {1}, 'B': {1}, 'C': {2}},
                                           lambda kind, n, details: problems.append((kind, n)))
        assert nodes == {1, 2} and checked.empty
        assert dict(problems)['duplicate_stock_pair'] == 2
        assert dict(problems)['unresolved_node_ticker'] == 1
        captured = []
        validate_month(folder, {'A': {1}, 'B': {1}, 'C': {2}},
                       lambda kind, n, details: captured.append({'kind': kind, 'examples': details}))
        json.dumps(captured)  # Null ticker diagnostics must remain serializable.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network-run', type=Path)
    parser.add_argument('--crsp', type=Path)
    parser.add_argument('--on-anomaly', choices=['exclude', 'stop'], default='exclude')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    self_check()
    if args.self_check:
        print('SNA metric and rolling mapping checks passed')
        return
    root = Path(__file__).resolve().parents[1]
    network_run = args.network_run or latest_run(root / 'Data/process/04_monthly_ticker_network')
    folders = sorted(p.parent for p in network_run.glob('????-??/edges.parquet'))
    if not folders: raise FileNotFoundError('No monthly edge files')
    manifest = json.loads((network_run / 'run_summary.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete': raise ValueError('04 run is not complete')
    if manifest.get('network_count') != len(folders): raise ValueError('04 network count differs from monthly files')
    out = root / 'Data/process/04_01_network_sna' / datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S_UTC')
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    needs_mapping = any('ticker_a' in pq.read_schema(folder / 'edges.parquet').names for folder in folders)
    crsp = args.crsp or root / 'Data/raw/stock/RAW_crsp.csv'
    mapping_stat = (crsp.stat().st_size, crsp.stat().st_mtime_ns) if needs_mapping else None
    monthly_mapping = crsp_monthly_mapping(crsp) if needs_mapping else {}
    crsp_hash = sha256_file(crsp) if needs_mapping else None
    summaries, distributions, curves, month_logs, events = [], [], [], [], []
    temp = out / 'network_node_monthly.parquet.partial'
    print(f'Input: {network_run}\nOutput: {out}', flush=True)
    with pq.ParquetWriter(temp, NODE_SCHEMA, compression='zstd') as writer:
        for folder in folders:
            month = str(pd.Period(folder.name, freq='M'))
            tick = time.perf_counter()
            anomaly_counts = {}
            def warn(kind, count, details):
                anomaly_counts[kind] = anomaly_counts.get(kind, 0) + count
                event = {'network_month': month, 'kind': kind, 'count': count, 'examples': details}
                events.append(event)
                warnings.warn(f'{month}: {kind}={count}; see validation_warnings.jsonl', stacklevel=2)
            edge_path = folder / 'edges.parquet'
            node_path = folder / 'nodes.parquet'
            stats_before = [(p, p.stat().st_size, p.stat().st_mtime_ns) for p in [edge_path, node_path] if p.exists()]
            mappings = window_mapping(month, monthly_mapping)
            nodes, edges, input_edges = validate_month(folder, mappings, warn)
            anomaly_count = sum(anomaly_counts.values())  # condition occurrences; may overlap records
            status = 'invalid' if anomaly_count and args.on_anomaly == 'stop' else ('partial' if anomaly_count else 'complete')
            if status == 'invalid':
                metrics = dict.fromkeys(METRICS, None)
                dist = dict.fromkeys(['mean', 'median', 'p90', 'p99', 'max'], None)
            else:
                metrics, node_frame, dist, raw = graph_statistics(nodes, edges)
                node_frame.insert(0, 'network_month', month)
                writer.write_table(pa.Table.from_pandas(node_frame, schema=NODE_SCHEMA, preserve_index=False))
                if len(raw): curves.append((month, np.percentile(raw, np.arange(101), method='linear')))
            for path, size, modified in stats_before:
                if (path.stat().st_size, path.stat().st_mtime_ns) != (size, modified):
                    raise ValueError(f'Input changed: {path}')
            summaries.append({'network_month': month, 'status': status, **metrics,
                              'input_edges': input_edges, 'excluded_edges': input_edges - len(edges),
                              'anomaly_count': anomaly_count})
            distributions.append({'network_month': month, 'status': status, **dist})
            month_logs.append({'network_month': month, 'status': status, 'nodes': metrics['number_of_nodes'],
                               'edges': metrics['number_of_edges'], 'input_edges': input_edges,
                               'excluded_edges': input_edges - len(edges), 'anomaly_count': anomaly_count,
                               'anomalies': anomaly_counts, 'runtime_seconds': time.perf_counter() - tick})
            print(f'{month}: {status}, nodes={metrics["number_of_nodes"]}, edges={metrics["number_of_edges"]}, anomalies={anomaly_count}', flush=True)
    expected_rows = sum(s['number_of_nodes'] or 0 for s in summaries)
    if pq.read_metadata(temp).num_rows != expected_rows: raise ValueError('Node output row count mismatch')
    # Check uniqueness of node-month keys in the written file.
    for batch in pq.ParquetFile(temp).iter_batches(batch_size=100_000):
        frame = batch.to_pandas()
        if frame.duplicated(['network_month', 'permno']).any(): raise ValueError('Duplicate node-month key')
    if needs_mapping and mapping_stat != (crsp.stat().st_size, crsp.stat().st_mtime_ns):
        raise ValueError('CRSP input changed during run')
    temp.replace(out / 'network_node_monthly.parquet')
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(out / 'network_monthly_summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(distributions).to_csv(out / 'network_weight_distribution.csv', index=False, encoding='utf-8-sig')
    with (out / 'validation_warnings.jsonl').open('w', encoding='utf-8') as stream:
        for event in events: stream.write(json.dumps(event, ensure_ascii=False) + '\n')
    plot_results(summary_frame, curves, out)
    report = {
        'status': 'complete_with_warnings' if events else 'complete',
        'network_run': str(network_run.resolve()), 'crsp_mapping_source': str(crsp.resolve()) if needs_mapping else None,
        'crsp_mapping_sha256': crsp_hash, 'network_count': len(folders), 'node_month_rows': expected_rows,
        'mapping_policy': 'Unique observed ticker-PERMNO identity in [M-6, M-1]; no future/global fallback.',
        'anomaly_policy': args.on_anomaly, 'duplicate_pair_policy': 'Exclude all members; never sum raw common-thread counts.',
        'strength_weight': 'log1p(n_common_threads)', 'clustering_definition': 'Unweighted local clustering; degree<2 -> 0; mean over all nodes.',
        'concentration_definition': 'Largest ceil(p*n) degrees / sum(degree); zero if total degree is zero.',
        'percentile_method': 'Linear within-month edge quantiles; each edge counts once, not weighted by thread count.',
        'figure_6_note': 'Original paper not specified. Generic monthly percentile curves, log y-axis; black curve is median of monthly quantile curves, not pooled edge quantiles.',
        'anomaly_count_definition': 'Sum of condition occurrences; counts can overlap the same record.',
        'months': month_logs, 'warning_events': len(events), 'runtime_seconds': time.perf_counter() - started,
        'output_directory': str(out),
    }
    (out / 'run_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Finished: {out}', flush=True)


if __name__ == '__main__':
    main()
