"""Shared output naming and successful-run registry for stages 00–03."""
from datetime import datetime, timezone
import json
from pathlib import Path

STAGES = {
    'universe': '00_approx_crsp_universe',
    'cleaned': '01_initial_csv_clean',
    'matching': '02_ticker_matching',
    'thread': '03_thread_construction',
}

def new_run(root, stage):
    path = Path(root)/'Data/process'/STAGES[stage]/datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S_UTC')
    path.mkdir(parents=True, exist_ok=False)
    return path

def registry(root):
    path = Path(root)/'Data/process/pipeline_runs.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}

def input_run(root, stage, legacy=None):
    value = registry(root).get(stage+'_manifest')
    if value:
        path = Path(value)
        if not path.is_absolute(): path = Path(root)/path
        info = json.loads(path.read_text(encoding='utf-8'))
        if info.get('status') != 'complete': raise ValueError(f'Incomplete upstream: {path}')
        return path.parent
    if legacy is not None:
        print(f'Using explicitly supported legacy {stage} input: {legacy}')
        return Path(legacy)
    raise FileNotFoundError(f'Run upstream stage {stage} first')

def publish(root, stage, run, summary):
    root, run = Path(root), Path(run)
    path = run/'run_summary.json'
    content = dict(summary, status='complete', run_directory=str(run))
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)
    state = registry(root)
    state[stage+'_manifest'] = str(path)
    downstream = {'universe':['matching','thread'], 'cleaned':['matching','thread'], 'matching':['thread'], 'thread':[]}
    for name in downstream[stage]: state[name+'_manifest'] = None
    pointer = root/'Data/process/pipeline_runs.json'
    temporary = pointer.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
    temporary.replace(pointer)

def remove_work_database(path):
    """Remove only this run's closed SQLite scratch files; no recursive deletion."""
    path = Path(path)
    if path.parent.name != '_work': raise ValueError('Expected a run-local _work directory')
    for suffix in ['', '-journal', '-wal', '-shm']:
        Path(str(path)+suffix).unlink(missing_ok=True)
    path.parent.rmdir()
