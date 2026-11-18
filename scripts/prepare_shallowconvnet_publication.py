#!/usr/bin/env python3
"""Prepare or verify the completed ShallowConvNet result tables and figures.

Reads only accuracy CSVs and small result JSON files. Does not import training
code, load EEG/checkpoint files, or perform training. See --help for usage.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / 'results/shallowconvnet/job-61441374'
WINDOWS = [800, 1500, 3000, 4000]
SUBJECTS = list(range(1, 19))
NAMES = ['HOC', 'WFE', 'WAA', 'EPS', 'EFE', 'SPS', 'SAA', 'SFE']
PAIRS = ['/'.join(p) for p in itertools.combinations(NAMES, 2)]
SOURCE_CSVS = ['all_pairs_per_subject.csv', 'pair_x_window_mean.csv',
               'subject_x_pair_4000ms.csv']
KEYS = ['pair', 'subject', 'window_ms']
SOURCE_SHA = 'e2ea42c2db7f0861caaaece7042bd81c69fd54b9555b3b6dbdb90db7463fec2e'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, **kwargs):
    return pd.read_csv(path, float_precision='round_trip', **kwargs)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12, err_msg=message)


def load_tables():
    tables = OUTPUT / 'tables'
    data = read(tables / SOURCE_CSVS[0])
    expected = set(itertools.product(PAIRS, SUBJECTS, WINDOWS))
    keys = list(data[KEYS].itertuples(index=False, name=None))
    check(len(data) == 2016 and len(set(keys)) == 2016 and set(keys) == expected,
          'Subject/pair/window cells are missing or duplicated.')
    check(np.isfinite(data.accuracy).all() and data.accuracy.between(0, 100).all(),
          'Invalid accuracy.')
    check(data.n_test.eq(80).all(), 'Each subject/pair/window must have 80 tests.')
    folds = read(tables / 'all_pairs_per_fold.csv')
    fold_keys = list(folds[KEYS + ['fold']].itertuples(index=False, name=None))
    check(len(folds) == 10080 and len(set(fold_keys)) == 10080
          and set(fold_keys) == {k + (f,) for k in expected for f in range(1, 6)},
          'Fold cells are missing or duplicated.')
    check(folds.n_test.eq(16).all(), 'Each fold must have 16 tests.')
    check(np.isfinite(folds[['accuracy', 'val_accuracy']]).all().all()
          and folds.accuracy.between(0, 100).all()
          and folds.val_accuracy.between(0, 100).all(), 'Invalid fold accuracy.')
    pooled = folds.groupby(KEYS).accuracy.mean().sort_index()
    close(pooled.to_numpy(), data.set_index(KEYS).accuracy.sort_index().to_numpy(),
          'Pooled test accuracy does not match five equally sized test folds.')
    pair = data.groupby(['pair', 'window_ms']).accuracy.mean().unstack().reindex(
        index=PAIRS, columns=WINDOWS)
    subject = data.groupby(['subject', 'window_ms']).accuracy.mean().unstack().reindex(
        index=SUBJECTS, columns=WINDOWS)
    original_pair = read(tables / 'pair_x_window_mean.csv', index_col=0)
    original_pair.columns = original_pair.columns.astype(int)
    close(pair.to_numpy(), original_pair.reindex(index=PAIRS, columns=WINDOWS).to_numpy(),
          'Pair summary differs from original output.')
    full = data[data.window_ms.eq(4000)].pivot(index='subject', columns='pair', values='accuracy')
    original_full = read(tables / 'subject_x_pair_4000ms.csv', index_col=0)
    close(full.reindex(index=SUBJECTS, columns=PAIRS).to_numpy(),
          original_full.reindex(index=SUBJECTS, columns=PAIRS).to_numpy(),
          'Full-window subject/pair matrix differs from original output.')
    return data, folds, pair, subject


def check_originals(source, data, folds):
    manifest_path = OUTPUT / 'original_output_sha256.json'
    files = {name: sha(source / name) for name in SOURCE_CSVS}
    check(sha(REPO / 'run_shallow_all.py') == SOURCE_SHA, 'Training source changed.')
    for name in SOURCE_CSVS:
        check(sha(source / name) == sha(OUTPUT / 'tables' / name),
              'Copied summary is not byte-identical: ' + name)
    for pair in PAIRS:
        directory = source / pair.replace('/', '_')
        for name in ['per_subject.csv', 'per_fold.csv', 'wide_subject_x_window.csv', 'results.json']:
            path = directory / name
            files[str(path.relative_to(source))] = sha(path)
        original = read(directory / 'per_subject.csv').sort_values(KEYS).reset_index(drop=True)
        published = data[data.pair.eq(pair)].sort_values(KEYS).reset_index(drop=True)
        pd.testing.assert_frame_equal(original, published, check_exact=True)
        original_folds = read(directory / 'per_fold.csv').sort_values(KEYS + ['fold']).reset_index(drop=True)
        published_folds = folds[folds.pair.eq(pair)].sort_values(KEYS + ['fold']).reset_index(drop=True)
        pd.testing.assert_frame_equal(original_folds, published_folds, check_exact=True)
        wide = original.pivot(index='subject', columns='window_ms', values='accuracy').reindex(
            index=SUBJECTS, columns=WINDOWS)
        original_wide = read(directory / 'wide_subject_x_window.csv', index_col=0)
        original_wide.columns = original_wide.columns.astype(int)
        close(wide.to_numpy(), original_wide.reindex(index=SUBJECTS, columns=WINDOWS).to_numpy(),
              'Per-pair wide matrix mismatch: ' + pair)
        payload = json.loads((directory / 'results.json').read_text())
        check(payload['subjects'] == SUBJECTS and payload['windows'] == WINDOWS,
              'Per-pair JSON coverage mismatch: ' + pair)
        close(wide.to_numpy(), np.array(payload['matrix']), 'Per-pair JSON matrix mismatch: ' + pair)
    if manifest_path.exists():
        check(json.loads(manifest_path.read_text()) == files, 'Original output hashes changed.')
    else:
        manifest_path.write_text(json.dumps(files, indent=2, sort_keys=True) + '\n')
    return len(files)


def derived_tables(subject):
    subject.index.name = 'subject'
    subject.to_csv(OUTPUT / 'tables/subject_x_window_mean.csv', float_format='%.17g')
    summary = pd.DataFrame(dict(window_ms=WINDOWS,
                                mean_accuracy_pct=subject.mean().to_numpy(),
                                between_subject_sd_pp=subject.std(ddof=1).to_numpy(),
                                n_subjects=18, n_pairs=28))
    summary.to_csv(OUTPUT / 'tables/window_summary.csv', index=False, float_format='%.17g')


def check_derived(subject):
    saved = read(OUTPUT / 'tables/subject_x_window_mean.csv', index_col=0)
    saved.columns = saved.columns.astype(int)
    close(saved.reindex(index=SUBJECTS, columns=WINDOWS).to_numpy(), subject.to_numpy(),
          'Subject/window means mismatch.')
    summary = read(OUTPUT / 'tables/window_summary.csv')
    check(summary.window_ms.tolist() == WINDOWS, 'Incorrect window order.')
    check(summary.n_subjects.eq(18).all() and summary.n_pairs.eq(28).all(), 'Incorrect summary counts.')
    close(summary.mean_accuracy_pct, subject.mean(), 'Window means mismatch.')
    close(summary.between_subject_sd_pp, subject.std(ddof=1), 'Between-subject SD mismatch.')


def figures(data, pair, subject):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'shallowconvnet-61441374'})
    colors = ['#0072B2', '#009E73', '#D55E00', '#CC79A7']

    def save(fig, name):
        for suffix in ['png', 'pdf', 'svg']:
            metadata = {'Creator': 'ShallowConvNet results job 61441374'}
            if suffix == 'pdf':
                metadata.update(CreationDate=None, ModDate=None)
            if suffix == 'svg':
                metadata['Date'] = None
            if suffix == 'png':
                metadata = {'Software': 'matplotlib'}
            target = OUTPUT / 'figures' / (name + '.' + suffix)
            fig.savefig(target, dpi=240,
                        bbox_inches='tight', metadata=metadata)
            if suffix == 'svg':
                target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines()) + '\n')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), layout='constrained')
    x = np.array(WINDOWS)
    for values in subject.to_numpy():
        ax.plot(x, values, color='#738395', alpha=.28, linewidth=.9)
    means, sd = subject.mean().to_numpy(), subject.std(ddof=1).to_numpy()
    ax.errorbar(x, means, yerr=sd, color='#152F4F', linewidth=2.2, marker='o',
                markersize=6, capsize=5, label='Mean ± between-subject SD (n=18)')
    for i, mean in enumerate(means):
        ax.annotate(f'{mean:.2f}%', (i, mean + sd[i]), xytext=(0, 6),
                    textcoords='offset points', ha='center', fontsize=9)
    ax.axhline(50, color='#666666', linestyle='--', linewidth=1, label='50% nominal chance')
    ax.set(xticks=x, xticklabels=[f'{w:,}' for w in WINDOWS], ylim=(0, 100),
           xlabel='EEG window (ms)', ylabel='Test accuracy (%)',
           title='ShallowConvNet accuracy by EEG window')
    ax.grid(axis='y', alpha=.16)
    ax.legend(loc='lower left', frameon=False, fontsize=8)
    save(fig, 'accuracy_by_window')

    fig, ax = plt.subplots(figsize=(12, 4.8), layout='constrained')
    x = np.arange(18)
    for i, w in enumerate(WINDOWS):
        ax.bar(x + (i - 1.5) * .2, subject[w], width=.19, color=colors[i],
               label=f'{w:,} ms')
    ax.axhline(50, color='#666666', linestyle='--', linewidth=1)
    ax.set(xticks=x, xticklabels=[f'S{s:02}' for s in SUBJECTS], ylim=(0, 100),
           ylabel='Mean test accuracy (%)', xlabel='Subject',
           title='All 18 subjects: mean over 28 joint pairs')
    ax.legend(frameon=False, ncol=4, loc='upper left')
    ax.grid(axis='y', alpha=.14)
    save(fig, 'accuracy_by_subject')

    fig, ax = plt.subplots(figsize=(6.7, 10), layout='constrained')
    im = ax.imshow(pair.to_numpy(), cmap='cividis', vmin=0, vmax=100, aspect='auto')
    ax.set(xticks=np.arange(4), xticklabels=[f'{w:,}' for w in WINDOWS],
           yticks=np.arange(28), yticklabels=PAIRS, xlabel='EEG window (ms)',
           title='All 28 pairs: mean over 18 subjects')
    for row in range(28):
        for col in range(4):
            ax.text(col, row, f'{pair.iloc[row, col]:.2f}', ha='center', va='center',
                    color='black', fontsize=8)
    fig.colorbar(im, ax=ax, label='Mean test accuracy (%)', shrink=.7)
    save(fig, 'accuracy_by_pair')

    fig, axes = plt.subplots(2, 2, figsize=(16, 10.5), layout='constrained')
    for ax, w in zip(axes.flat, WINDOWS):
        matrix = data[data.window_ms.eq(w)].pivot(index='subject', columns='pair', values='accuracy')
        matrix = matrix.reindex(index=SUBJECTS, columns=PAIRS)
        im = ax.imshow(matrix.to_numpy(), cmap='cividis', vmin=0, vmax=100, aspect='auto')
        ax.set(xticks=np.arange(28), xticklabels=PAIRS, yticks=np.arange(18),
               yticklabels=[f'S{s:02}' for s in SUBJECTS], title=f'{w:,} ms')
        ax.tick_params(axis='x', labelrotation=90, labelsize=7)
        ax.tick_params(axis='y', labelsize=8)
    fig.colorbar(im, ax=list(axes.flat), label='Test accuracy (%)', shrink=.72)
    fig.suptitle('Every subject and joint pair at all four EEG windows', fontsize=14)
    save(fig, 'subject_pair_all_windows')


def manifest(verify=False):
    path = OUTPUT / 'SHA256SUMS'
    if verify:
        recorded = dict(line.split('  ', 1)[::-1] for line in path.read_text().splitlines())
        actual = {str(p.relative_to(OUTPUT)): sha(p) for p in OUTPUT.rglob('*')
                  if p.is_file() and p != path}
        check(recorded == actual, 'Published artifact hashes or file list changed.')
    else:
        path.write_text(''.join(sha(p) + '  ' + str(p.relative_to(OUTPUT)) + '\n'
                                for p in sorted(OUTPUT.rglob('*')) if p.is_file() and p != path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-results', type=Path,
                        help='Original mi_cmp_baseline directory; enables byte/value comparisons.')
    parser.add_argument('--verify-only', action='store_true', help='Verify without writing any files.')
    args = parser.parse_args()
    source = json.loads((OUTPUT / 'provenance.json').read_text())['source']
    for name, digest in source['training_scripts_sha256'].items():
        check(sha(REPO / name) == digest, 'Original training script changed: ' + name)
    if args.source_results and not args.verify_only:
        for name in SOURCE_CSVS:
            shutil.copyfile(args.source_results / name, OUTPUT / 'tables' / name)
        combined = pd.concat([read(args.source_results / pair.replace('/', '_') / 'per_fold.csv')
                              for pair in PAIRS], ignore_index=True)
        combined.to_csv(OUTPUT / 'tables/all_pairs_per_fold.csv', index=False, float_format='%.17g')
    data, folds, pair, subject = load_tables()
    source_count = None
    if args.source_results:
        source_count = check_originals(args.source_results, data, folds)
    if args.verify_only:
        check_derived(subject)
        check(sha(REPO / 'run_shallow_all.py') == SOURCE_SHA, 'Training source changed.')
        manifest(verify=True)
        print('PASS: table coverage, fold pooling, original summaries, derived summaries and artifact hashes.')
    else:
        derived_tables(subject)
        check_derived(subject)
        figures(data, pair, subject)
        report = dict(status='PASS', job_id=61441374, subjects=18, pairs=28,
                      windows_ms=WINDOWS, subject_pair_window_rows=2016, fold_rows=10080,
                      original_metric_files_checked=source_count or 115,
                      copied_summary_csvs_byte_identical=bool(args.source_results),
                      numerical_comparison_atol=1e-12, numerical_comparison_rtol=0,
                      per_pair_subject_and_fold_values_compared_exactly=bool(args.source_results),
                      source_sha256=SOURCE_SHA,
                      notes='Checks against original outputs do not establish independent scientific replication.')
        if not args.source_results:
            previous = OUTPUT / 'verification.json'
            if previous.exists():
                report = json.loads(previous.read_text())
        (OUTPUT / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
        manifest()
        print('Prepared 6 tables and 4 figures in PNG/PDF/SVG; verification PASS.')
    print(subject.mean().round(6).to_string())


if __name__ == '__main__':
    main()
