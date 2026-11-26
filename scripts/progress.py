#!/usr/bin/env python3
"""Inspect baseline checkpoints/logs with the Python standard library only."""
import itertools
import re
import statistics
from pathlib import Path

root = Path('/pscratch/sd/a/anis/shallowconvnet')
names = ['HOC', 'WFE', 'WAA', 'EPS', 'EFE', 'SPS', 'SAA', 'SFE']
pairs = [f'{a}_{b}' for a, b in itertools.combinations(names, 2)]
results = root / 'data' / 'mi_cmp_baseline'
complete_pairs = 0
complete_units = 0
for pair in pairs:
    directory = results / pair
    complete_pairs += (directory / 'wide_subject_x_window.csv').is_file()
    complete_units += sum((directory / '_partial' / f'S{s}.pkl').is_file()
                          for s in range(1, 19))
print(f'Completed pairs: {complete_pairs}/28')
print(f'Completed subject/pair checkpoints: {complete_units}/504')
print('Each subject/pair checkpoint covers 4 windows x 5 folds x 50 epochs.')
pattern = re.compile(r'^S\s*\d+:.*\[\s*([0-9.]+)s\]\s*$')
logs = sorted((root / 'logs').glob('scn-baseline-*.out'), key=lambda p: p.stat().st_mtime)
durations = []
for path in logs[-5:]:
    with path.open(errors='replace') as f:
        for line in f:
            match = pattern.match(line)
            if match:
                durations.append(float(match.group(1)))
if durations:
    recent = durations[-10:]
    seconds = statistics.median(recent)
    remaining = max(0, 504 - complete_units) * seconds / 3600
    print(f'Median of latest {len(recent)} completed units: {seconds:.1f} seconds')
    print(f'Estimated remaining compute time: {remaining:.2f} hours')
    print('Rough estimate; excludes queue waits, startup, and interrupted-subject recomputation.')
    if len(durations) < 3:
        print('Wait for several completed subjects before relying on the estimate.')
else:
    print('No completed-subject timing lines yet; runtime cannot be estimated.')
