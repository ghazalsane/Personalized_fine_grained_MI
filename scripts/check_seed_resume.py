"""Exercise interruption and checkpoint reuse with synthetic results, never training."""
import copy
import hashlib
import importlib.util
import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path('/global/homes/a/anis/shallowconvnet')
WORK = Path('/pscratch/sd/a/anis/shallowconvnet')
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

resume = load('resume_helper', REPO / 'scripts/seed_resume.py')
validator = load('validator', REPO / 'scripts/validate_perlmutter.py')
validator.SOURCE = REPO / 'run_shallow_all.py'
scope = validator.definitions()
fixture = WORK / 'validation' / ('resume-SYNTHETIC-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
fixture.mkdir(exist_ok=False)
root = fixture / 'mi_cmp_baseline_seed100042'
identity = dict(schema_version=1, seed=100042, source_sha256=hashlib.sha256(validator.SOURCE.read_bytes()).hexdigest(),
                dataset_fingerprint='SYNTHETIC', environment={'fixture': True})
assert resume.prepare_results(root, identity)['mode'] == 'fresh'
scope.update(RESULTS_ROOT=str(root), SUBJ_ALL=np.arange(1, 19), TEST_TIME_WINDOWS_MS_FINAL=resume.WINDOWS,
             JOINT_NAMES=dict(enumerate(resume.JOINTS)), DATASET_ROOT='SYNTHETIC',
             seed=100042, GPU='SYNTHETIC-no-training')
def checkpoint(directory, sid):
    directory.mkdir(parents=True, exist_ok=True)
    folds = [dict(fold=k, test_accuracy=50., val_accuracy=50., n_test_samples=16,
                  test_preds=np.zeros(16, dtype=int), test_labels=np.tile([0, 1], 8)) for k in range(1, 6)]
    windows = {w: dict(test_accuracy=50., n_test_samples=80, fold_results=copy.deepcopy(folds),
                      test_preds=np.zeros(80, dtype=int), test_labels=np.tile([0, 1], 40),
                      confusion_matrix=np.array([[40, 0], [40, 0]])) for w in resume.WINDOWS}
    with (directory / f'S{sid}.pkl').open('xb') as stream:
        pickle.dump(dict(subject_id=sid, n_folds=5, time_window_results=windows), stream)
for sid in range(1, 19):
    checkpoint(root / 'HOC_WFE/_partial', sid)
scope['run_pair'](0, 1)  # Produce a completed synthetic pair entirely from saved subjects.
checkpoint(root / 'HOC_WAA/_partial', 1)
state = resume.prepare_results(root, identity, resume=True)
assert state['completed_pairs'] == 1 and state['checkpoints'] == 19
scope['run_pair'](0, 1)  # Must skip, leaving every completed-pair file unchanged.
resume.verify_protected(root, state['protected'])

class SyntheticInterruption(Exception):
    pass
class InterruptedData:
    def __getitem__(self, _):
        raise SyntheticInterruption('Synthetic interruption before any model or optimizer is constructed')
scope.update(X_ALL=InterruptedData(), Y_ALL=np.zeros(18, dtype=int))
try:
    scope['run_pair'](0, 2)  # Load S1, then interrupt at unfinished S2 without a model fit.
    raise AssertionError('Synthetic interruption was not exercised')
except SyntheticInterruption:
    pass
resume.verify_protected(root, state['protected'])
for sid in range(2, 19):
    checkpoint(root / 'HOC_WAA/_partial', sid)
second_state = resume.prepare_results(root, identity, resume=True)
scope['run_pair'](0, 2)  # All checkpoints load; original data sentinel cannot be touched.
resume.verify_protected(root, second_state['protected'])
assert resume.prepare_results(root, identity, resume=True)['completed_pairs'] == 2
tests = {'existing_completed_pair_skipped_without_changes': 'PASS',
         'saved_subject_reloaded_before_synthetic_interruption': 'PASS',
         'interrupted_pair_rebuilt_from_saved_subjects': 'PASS',
         'completed_files_and_checkpoints_hashes_and_mtimes_unchanged': 'PASS',
         'no_model_fitting_or_cuda_execution': 'PASS'}
def rejected(name, func):
    try:
        func()
    except (ValueError, AssertionError, KeyError):
        tests[name] = 'PASS'
    else:
        raise AssertionError(name + ' should have been rejected')
rejected('fresh_mode_rejects_existing_directory', lambda: resume.prepare_results(root, identity))
bad = copy.deepcopy(identity)
bad['source_sha256'] = 'changed-source'
rejected('resume_rejects_changed_identity', lambda: resume.prepare_results(root, bad, resume=True))
rejected('original_baseline_path_forbidden', lambda: resume.prepare_results(fixture / 'mi_cmp_baseline', identity, resume=True))
missing = fixture / 'no-marker' / root.name
missing.mkdir(parents=True)
rejected('resume_requires_identity_marker', lambda: resume.prepare_results(missing, identity, resume=True))
truncated = fixture / 'truncated' / root.name
shutil.copytree(root, truncated)
(truncated / 'HOC_WFE/wide_subject_x_window.csv').write_text('subject,800\n1,50\n')
rejected('resume_rejects_truncated_completion_marker', lambda: resume.prepare_results(truncated, identity, resume=True))
finalized = fixture / 'finalized' / root.name
shutil.copytree(root, finalized)
(finalized / 'all_pairs_per_subject.csv').write_text('SYNTHETIC-existing-summary\n')
rejected('resume_refuses_existing_run_summary', lambda: resume.prepare_results(finalized, identity, resume=True))

launcher = (REPO / 'submit_seeds.sh').read_text()
compile(launcher.split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0], 'inline-preflight', 'exec')
compile((REPO / 'scripts/seed_resume.py').read_text(), 'seed_resume.py', 'exec')
record = dict(status='PASS', fixture_directory=str(fixture), tests=tests,
              testing_method='Synthetic checkpoint-only execution of the unchanged run_pair; interruption before constructing a model',
              resume_granularity='Completed subject per pair; an unfinished subject repeats its 4 windows x 5 folds',
              source_sha256=identity['source_sha256'],
              file_sha256={name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in ['submit_seeds.sh', 'scripts/seed_resume.py', 'run_shallow_all.py']})
(REPO / 'docs/seed-resume-validation.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
