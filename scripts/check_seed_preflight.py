#!/usr/bin/env python3
"""Lightweight pre-submission validation. Never imports the training module."""
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path('/global/homes/a/anis/shallowconvnet')
WORK = Path('/pscratch/sd/a/anis/shallowconvnet')
INSTALL = json.loads((REPO / 'docs/seed-update-installation.json').read_text())
BACKUP = Path(INSTALL['backup_directory'])
SEEDS = [100042, 200042, 300042]
report = {'status': 'PASS', 'reference_job': 61441374,
          'utc': datetime.now(timezone.utc).isoformat(),
          'project_root': str(REPO), 'backup_directory': str(BACKUP),
          'no_jobs_submitted': True, 'training_executed': False,
          'cuda_execution_performed': False}
new = ast.parse((REPO / 'run_shallow_all.py').read_text())
old = ast.parse((BACKUP / 'run_shallow_all.py.original').read_text())
for name in ('run_shallow_all.py', 'aggregate_seeds.py'):
    compile((REPO / name).read_text(), str(REPO / name), 'exec')
launcher = (REPO / 'submit_seeds.sh').read_text()
inline = launcher.split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
compile(inline, 'submit_seeds.sh:inline-preflight', 'exec')
subprocess.run(['bash', '-n', str(REPO / 'submit_seeds.sh')], check=True)
report['syntax'] = 'PASS: both Python files, launcher, and inline Python'

def without_seed_changes(tree):
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in ('seed', 'RESULTS_ROOT')
                for t in node.targets):
            continue
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == 'print'
                and node.value.args and isinstance(node.value.args[0], ast.JoinedStr)
                and any(isinstance(v, ast.Constant) and isinstance(v.value, str)
                        and v.value.startswith('Seed : ') for v in node.value.args[0].values)):
            continue
        body.append(node)
    return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)

assert without_seed_changes(new) == without_seed_changes(old)
assert hashlib.sha256((REPO / 'run_shallow_all.py').read_bytes()).hexdigest() == INSTALL['download_sha256']['run_shallow_all.py']
assert (WORK / 'code/run_shallow_all.py').read_bytes() == (BACKUP / 'run_shallow_all.py.original').read_bytes()
report['training_source'] = 'Exact download; only seed assignment, output assignment, and seed print differ from job 61441374'
report['historical_scratch_source_unchanged'] = True
config_nodes = [n for n in new.body if isinstance(n, ast.Assign) and
                any(isinstance(t, ast.Name) and t.id in
                    ('seed', 'DRIVE', 'DATASET_ROOT', 'RESULTS_ROOT') for t in n.targets)]
config_code = compile(ast.Module(body=config_nodes, type_ignores=[]), '<configuration-only>', 'exec')
outputs = {}
for seed in SEEDS:
    os.environ.update(MI_SEED=str(seed), MI_DRIVE_ROOT=str(WORK / 'data'))
    scope = {'os': os}
    exec(config_code, scope)
    assert scope['seed'] == seed
    assert scope['DATASET_ROOT'] == str(WORK / 'data/datasetpaper_4_40hz_MIonly')
    p = Path(scope['RESULTS_ROOT'])
    assert p == WORK / f'data/mi_cmp_baseline_seed{seed}'
    assert p.resolve() == p and not os.path.lexists(p)
    outputs[str(seed)] = str(p)
assert len(set(outputs.values())) == 3
assert re.search(r'^SEEDS=\(100042 200042 300042\)$', launcher, re.M)
assert '#SBATCH --array=0-2\n' in launcher
report['seeds'] = SEEDS
report['separate_fresh_outputs'] = outputs
report['original_output_not_selected'] = True
expected = {
    'account': 'm4431_g', 'qos': 'shared', 'constraint': 'gpu&hbm40g',
    'nodes': '1', 'ntasks': '1', 'gpus-per-task': '1',
    'cpus-per-task': '32', 'mem': '55G', 'time': '08:00:00',
    'chdir': str(REPO)}
for key, value in expected.items():
    assert f'#SBATCH --{key}={value}\n' in launcher
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    assert f'export {name}=16\n' in launcher
assert '--cpu-bind=cores' in launcher
assert 'source /global/homes/a/anis/shallowconvnet/scripts/job_common.sh' not in launcher
assert '--mode ready' not in launcher
assert (WORK / 'logs').is_dir() and (WORK / 'validation').is_dir()
assert os.access(WORK / 'data', os.W_OK)
assert os.access(WORK / 'logs', os.W_OK)
guard = subprocess.run(['bash', str(REPO / 'submit_seeds.sh')], capture_output=True, text=True)
assert guard.returncode == 1 and 'Slurm array on a Perlmutter compute node' in guard.stderr
report['login_node_guard'] = 'PASS: exit 1 before module load, file writes, or srun'
report['slurm_request'] = expected
report['shellcheck'] = 'Unavailable; bash -n and targeted launcher checks passed'

reference = json.loads((WORK / 'logs/scn-baseline-61441374/readiness.json').read_text())
identity = reference['identity']
current = {
    'python': platform.python_version(),
    'packages': {p: importlib.metadata.version(p) for p in identity['packages']},
    'torch_cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version()}
for key, value in current.items():
    assert identity[key] == value, (key, identity[key], value)
report['environment'] = current
report['environment_matches_job_61441374'] = True
modules = subprocess.run(['bash', '-lc', 'module load pytorch/2.8.0\nmodule -t list'], capture_output=True, text=True, check=True)
report['current_modules'] = modules.stderr.strip().splitlines()
spec = importlib.util.spec_from_file_location('original_validator', REPO / 'scripts/validate_perlmutter.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)
validator.SOURCE = REPO / 'run_shallow_all.py'
scope = validator.definitions()
records, fingerprint = validator.inspect_dataset(scope, full=False)
assert records == reference['subjects']
assert fingerprint == identity['dataset_fingerprint']
report['dataset'] = {'path': str(validator.DATASET), 'subjects': 18,
    'data_shape_per_subject': [320, 62, 1000], 'data_dtype': 'float32',
    'labels_dtype': 'int64', 'class_counts_per_subject': [40] * 8,
    'fingerprint': fingerprint, 'matches_previous_job': True,
    'eeg_payload_loaded': False}
old_split_node = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == 'subject_splits')
old_scope = dict(np=np, StratifiedKFold=scope['StratifiedKFold'], train_test_split=scope['train_test_split'])
exec(compile(ast.Module(body=[old_split_node], type_ignores=[]), '<original-splits>', 'exec'), old_scope)
for record in records:
    with np.load(validator.DATASET / record['name'], allow_pickle=False) as z:
        labels = z['labels']
    baseline = old_scope['subject_splits'](labels, record['subject'], 5)
    for seed in SEEDS:
        scope['seed'] = seed
        splits = scope['subject_splits'](labels, record['subject'], 5)
        assert all(np.array_equal(a, b) for first, second in zip(baseline, splits) for a, b in zip(first, second))
report['folds'] = 'PASS: all 18 subjects, all 28 pair restrictions, all 5 folds; all three seeds match original train/validation/test indices'

fixture = WORK / 'validation' / ('seed-preflight-synthetic-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
fixture.mkdir(parents=True, exist_ok=False)
base = pd.DataFrame({'pair': ['HOC/WFE', 'HOC/WFE'], 'subject': [1, 1],
                     'window_ms': [800, 1500], 'accuracy': [50., 60.], 'n_test': [80, 80]})
def save(name, frame):
    directory = fixture / name
    directory.mkdir()
    frame.to_csv(directory / 'all_pairs_per_subject.csv', index=False)
    return directory
a = save('seed-a-SYNTHETIC', base)
bframe = base.copy()
bframe['accuracy'] += 2
b = save('seed-b-SYNTHETIC', bframe.iloc[::-1])
cframe = base.copy()
cframe['accuracy'] += 4
c = save('seed-c-SYNTHETIC', cframe)
checks = {}
def aggregate(name, inputs, success, script=None):
    cwd = fixture / name
    cwd.mkdir()
    command = ['python', str(script or REPO / 'aggregate_seeds.py')] + [str(x) for x in inputs]
    proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    (cwd / 'stdout.txt').write_text(proc.stdout)
    (cwd / 'stderr.txt').write_text(proc.stderr)
    assert (proc.returncode == 0) == success, (name, proc.stdout, proc.stderr)
    if not success:
        assert not (cwd / 'baseline_seed_mean.csv').exists(), name
    checks[name] = 'PASS'
    return cwd
cwd = aggregate('valid-three-seeds-reordered', [a, b, c], True)
out = pd.read_csv(cwd / 'baseline_seed_mean.csv')
np.testing.assert_allclose(out['accuracy'], [52., 62.])
assert out['n_seeds'].tolist() == [3, 3]
extra = pd.concat([bframe, pd.DataFrame({'pair': ['WFE/WAA'], 'subject': [1], 'window_ms': [800], 'accuracy': [70.], 'n_test': [80]})], ignore_index=True)
extra_dir = save('extra-cell-SYNTHETIC', extra)
aggregate('original-extra-cell-bug-reproduced', [a, extra_dir], True, BACKUP / 'downloads/aggregate_seeds.py')
aggregate('extra-cell-rejected', [a, extra_dir], False)
aggregate('missing-cell-rejected', [a, save('missing-SYNTHETIC', base.iloc[:1])], False)
aggregate('duplicate-key-rejected', [a, save('duplicate-SYNTHETIC', pd.concat([base, base.iloc[:1]]))], False)
aggregate('duplicate-directory-rejected', [a, a], False)
missing = base.copy()
missing.loc[0, 'accuracy'] = np.nan
aggregate('nan-rejected', [a, save('nan-SYNTHETIC', missing)], False)
counts = bframe.copy()
counts.loc[0, 'n_test'] = 79
aggregate('test-count-mismatch-rejected', [a, save('count-mismatch-SYNTHETIC', counts)], False)
aggregate('too-few-inputs-rejected', [a], False)
report['aggregation_tests'] = checks
report['synthetic_fixture_directory'] = str(fixture)
report['aggregation_output_location'] = 'baseline_seed_mean.csv in the chosen working directory; use a new scratch aggregation directory'

before = json.loads((BACKUP / 'baseline-before.json').read_text())
root = WORK / 'data/mi_cmp_baseline'
after = {}
for parent, dirs, names in os.walk(root, followlinks=False):
    for name in names:
        p = Path(parent) / name
        assert not p.is_symlink()
        st = p.stat()
        after[str(p.relative_to(root))] = {'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns,
                                         'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
assert before == after, 'Original results changed'
report['original_baseline_integrity'] = {'status': 'PASS', 'files': len(after),
    'bytes': sum(x['bytes'] for x in after.values()), 'all_sha256_and_mtimes_unchanged': True}
report['script_sha256'] = {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
                         for name in ('run_shallow_all.py', 'submit_seeds.sh', 'aggregate_seeds.py')}
report['recommended_time_limit'] = '08:00:00 per seed'
report['previous_elapsed'] = '05:46:27'
report['headroom'] = '02:13:33 (38.55%)'
report['pending'] = ['Actual CUDA/GPU checks for these jobs occur on compute nodes after explicit submission approval',
    'The changed seeds have not been trained; scientific results and exact runtime are unmeasured',
    'No sbatch test-only request was made']
(REPO / 'docs/seed-preflight.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
