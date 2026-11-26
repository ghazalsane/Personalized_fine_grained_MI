#!/usr/bin/env python3
"""Dataset inspection and isolated GPU checks, never full training.

The original script starts training at import time. Compile only its original
function/class definitions for checks, without running its top-level statements.
"""
import argparse
import ast
import gc
import glob
import hashlib
import importlib.metadata
import itertools
import json
import os
import pickle
import platform
import re
import socket
import time
import zipfile
from pathlib import Path

SOURCE = Path('/pscratch/sd/a/anis/shallowconvnet/code/run_shallow_all.py')
DATASET = Path('/pscratch/sd/a/anis/shallowconvnet/data/datasetpaper_4_40hz_MIonly')
WINDOWS = [800, 1500, 3000, 4000]
PACKAGES = ['torch', 'numpy', 'pandas', 'scikit-learn', 'scipy', 'joblib', 'threadpoolctl']


def require_compute_step():
    if (not os.environ.get('SLURM_JOB_ID')
            or not socket.gethostname().split('.')[0].startswith('nid')
            or os.environ.get('SLURM_STEP_ID', '') in ('', 'batch', 'extern')):
        raise RuntimeError('GPU validation requires an srun step on a compute node.')


def definitions():
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    selected = ast.Module(body=[n for n in tree.body
                               if isinstance(n, (ast.FunctionDef, ast.ClassDef))],
                          type_ignores=[])
    scope = dict(np=np, pd=pd, torch=torch, nn=nn, Dataset=Dataset,
                 DataLoader=DataLoader, StratifiedKFold=StratifiedKFold,
                 train_test_split=train_test_split, accuracy_score=accuracy_score,
                 confusion_matrix=confusion_matrix, os=os, pickle=pickle,
                 time=time, json=json, gc=gc, re=re)
    exec(compile(selected, str(SOURCE), 'exec'), scope)
    return scope


def inspect_dataset(scope, full=False):
    np = scope['np']
    paths = glob.glob(str(DATASET / 'subject*_eeg_epochs_*.npz'))
    ids = sorted({scope['get_subject_number'](p) for p in paths})
    assert ids == list(range(1, 19)), f'Expected subjects 1..18, found {ids}'
    records = []
    for sid in ids:
        candidates = glob.glob(str(DATASET / f'subject{sid}_eeg_epochs_*.npz'))
        assert len(candidates) == 1, (sid, candidates)
        path = Path(candidates[0])
        with zipfile.ZipFile(path) as z:
            headers = {}
            for key in ('data', 'labels'):
                with z.open(key + '.npy') as f:
                    version = np.lib.format.read_magic(f)
                    reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                              else np.lib.format.read_array_header_2_0)
                    shape, fortran, dtype = reader(f)
                headers[key] = dict(shape=list(shape), dtype=str(dtype), fortran=fortran)
        assert headers['data']['shape'] == [320, 62, 1000], (sid, headers)
        assert headers['data']['dtype'] == 'float32', (sid, headers)
        assert headers['labels']['shape'] == [320], (sid, headers)
        assert headers['labels']['dtype'] == 'int64', (sid, headers)
        with np.load(path, allow_pickle=False) as z:
            labels = z['labels'].astype(np.int64)
            classes, counts = np.unique(labels, return_counts=True)
            assert np.array_equal(classes, np.arange(8)), (sid, classes)
            assert np.array_equal(counts, np.full(8, 40)), (sid, counts)
            if full:
                data = z['data']
                assert np.isfinite(data).all(), f'Nonfinite EEG data: subject {sid}'
                del data
        splits = scope['subject_splits'](labels, sid, 5)
        for tr, va, te in splits:
            assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
            assert len(set(tr) | set(va) | set(te)) == 320
            for a, b in itertools.combinations(range(8), 2):
                for idx, count in ((tr, 24), (va, 8), (te, 8)):
                    assert (labels[idx] == a).sum() == count
                    assert (labels[idx] == b).sum() == count
        assert np.array_equal(np.sort(np.concatenate([te for _, _, te in splits])),
                              np.arange(320))
        stat = path.stat()
        records.append(dict(subject=sid, name=path.name, size=stat.st_size,
                            mtime_ns=stat.st_mtime_ns, headers=headers,
                            class_counts=counts.tolist()))
        print(f'S{sid:02}: (320,62,1000), classes 0..7 x 40; all 28 pair splits OK')
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return records, digest


def verify_resume(scope, directory):
    """Exercise original completed-subject/pair branches with no model fitting."""
    np, pd = scope['np'], scope['pd']
    root = directory / 'synthetic_resume_fixture'
    partial = root / 'HOC_WFE' / '_partial'
    partial.mkdir(parents=True, exist_ok=False)
    fold = dict(test_accuracy=100., val_accuracy=100., n_test_samples=2)
    tw = {w: dict(test_accuracy=100., n_test_samples=10,
                  fold_results=[dict(fold, fold=k) for k in range(1, 6)])
          for w in WINDOWS}
    checkpoint = partial / 'S1.pkl'
    with checkpoint.open('wb') as f:
        pickle.dump(dict(subject_id=1, n_folds=5, time_window_results=tw), f)
    original_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    scope.update(RESULTS_ROOT=str(root), SUBJ_ALL=np.array([1]),
                 TEST_TIME_WINDOWS_MS_FINAL=WINDOWS, JOINT_NAMES={0: 'HOC', 1: 'WFE'},
                 DATASET_ROOT=str(DATASET), seed=42, GPU='synthetic-resume-check')
    first = scope['run_pair'](0, 1)
    second = scope['run_pair'](0, 1)
    pd.testing.assert_frame_equal(first, second)
    assert first.shape == (4, 5)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == original_digest
    print('Original resume branches passed: subject checkpoint load and completed-pair skip.')


def gpu_checks(scope, backward):
    np, torch = scope['np'], scope['torch']
    assert os.environ.get('PYTHONHASHSEED') == '42'
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    assert torch.cuda.is_available(), 'CUDA unavailable; CPU fallback is forbidden.'
    assert torch.cuda.device_count() == 1, 'Exactly one visible GPU is required.'
    name = torch.cuda.get_device_name(0)
    assert 'A100' in name, name
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    if backward:
        path = glob.glob(str(DATASET / 'subject1_eeg_epochs_*.npz'))[0]
        with np.load(path, allow_pickle=False) as z:
            x, y = z['data'].astype(np.float32), z['labels'].astype(np.int64)
        tr, va, te = scope['subject_splits'](y, 1, 5)[0]
        for window, samples in zip(WINDOWS, (200, 375, 750, 1000)):
            xn, _, _ = scope['z_score_normalize'](x[tr, :, :samples],
                                                 x[va, :, :samples], x[te, :, :samples])
            model = scope['ShallowConvNet'](62, samples, n_classes=2).to('cuda')
            model.apply(scope['init_weights'])
            inputs = torch.as_tensor(xn[:2], dtype=torch.float32, device='cuda')
            outputs = model(inputs)
            loss = torch.nn.CrossEntropyLoss()(outputs, torch.tensor([0, 1], device='cuda'))
            loss.backward()  # Test kernels only: no optimizer or experiment training.
            torch.cuda.synchronize()
            assert outputs.shape == (2, 2) and torch.isfinite(outputs).all()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters()
                       if p.grad is not None)
            print(f'{window}ms: original model CUDA forward/backward passed on {name}')
            del model, inputs, outputs, loss
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['metadata', 'gpu', 'ready'], required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--validated', type=Path)
    args = parser.parse_args()
    if args.mode != 'metadata':
        require_compute_step()
    scope = definitions()
    records, fingerprint = inspect_dataset(scope, full=args.mode == 'gpu')
    identity = dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    dataset_fingerprint=fingerprint, python=platform.python_version(),
                    packages={p: importlib.metadata.version(p) for p in PACKAGES},
                    torch_cuda=scope['torch'].version.cuda,
                    cudnn=scope['torch'].backends.cudnn.version())
    if args.mode != 'metadata':
        identity['gpu'] = gpu_checks(scope, backward=args.mode == 'gpu')
    if args.mode == 'ready':
        assert args.validated, '--validated GPU report is required'
        previous = json.loads(args.validated.read_text())
        assert previous['mode'] == 'gpu' and previous['passed']
        assert previous['identity'] == identity, 'Source, dataset, environment, or GPU changed; revalidate.'
        result_root = DATASET.parent / 'mi_cmp_baseline'
        result_root.mkdir(exist_ok=True)
        manifest = result_root / '_perlmutter_identity.json'
        if manifest.exists():
            assert json.loads(manifest.read_text()) == identity, 'Existing results use a different run identity.'
        else:
            assert not any(result_root.iterdir()), 'Existing results lack run identity; review before continuing.'
            with manifest.open('x') as f:
                json.dump(identity, f, indent=2)
    else:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        verify_resume(scope, args.report.parent)
    report = dict(passed=True, mode=args.mode, identity=identity, subjects=records,
                  job_id=os.environ.get('SLURM_JOB_ID'),
                  utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  full_eeg_finite_scan=args.mode == 'gpu')
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS ({args.mode}): {args.report}')


if __name__ == '__main__':
    main()
