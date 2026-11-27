#!/usr/bin/env python3
"""Build the job list for the FEDformer reproduction from the official scripts.

Source of truth: `official/scripts/run_M.sh` (multivariate). The script is a bash
double loop over `for model in ...` and `for preLen in 96 192 336 720`; this script
expands both loops textually, keeps only `model = FEDformer`, drops the illness
dataset (not part of this repository's benchmark), and turns every remaining
`python -u run.py ...` call into one row per seed.

Everything a call does not set falls back to the `run.py` default, recorded in the
`*_src` columns as 'run.py default'. `--root_path` is dropped here because the
scheduler supplies it (one data root for all job work dirs), and `--itr` is forced
to 1 (the repository convention is three explicit seeds, not the script's itr=3).

Outputs (configs/):
  official_script_args.json  every call, verbatim args, per (model, preLen, dataset)
  jobs.csv                   FEDformer only: 8 datasets x 4 horizons x 3 seeds
"""
import csv
import json
import os
import re
import shlex

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OFF = os.path.join(ROOT, 'official')
CFG = os.path.join(ROOT, 'configs')
SEEDS = [2026, 2027, 2028]

# dataset csv -> (name used in results, subdirectory under the data root, rows)
DATASETS = {
    'ETTh1.csv': ('ETTh1', 'ETT-small', 17420),
    'ETTh2.csv': ('ETTh2', 'ETT-small', 17420),
    'ETTm1.csv': ('ETTm1', 'ETT-small', 69680),
    'ETTm2.csv': ('ETTm2', 'ETT-small', 69680),
    'electricity.csv': ('ECL', 'electricity', 26304),
    'traffic.csv': ('Traffic', 'traffic', 17544),
    'weather.csv': ('Weather', 'weather', 52696),
    'exchange_rate.csv': ('Exchange', 'exchange_rate', 7587),
}
DROP = {'national_illness.csv'}
ORDER = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Exchange', 'Weather', 'ECL', 'Traffic']
HP = ['seq_len', 'label_len', 'pred_len', 'e_layers', 'd_layers', 'd_model', 'd_ff',
      'n_heads', 'batch_size', 'learning_rate', 'train_epochs', 'patience', 'lradj',
      'dropout', 'factor', 'enc_in', 'dec_in', 'c_out', 'embed', 'freq']


def run_py_defaults():
    src = open(os.path.join(OFF, 'run.py')).read()
    d = {}
    for line in src.split('\n'):
        m = re.search(r"add_argument\('--(\w+)'.*?default=([^,]+),", line)
        if m:
            d[m.group(1)] = m.group(2).strip().strip("'\"")
    return d


def parse_calls(script):
    """Expand the two loops and return one dict per `python -u run.py` call."""
    txt = open(script).read().replace('\\\n', ' ')
    models = re.search(r'for model in ([^\n]+)', txt).group(1).split()
    prelens = re.search(r'for preLen in ([^\n]+)', txt).group(1).split()
    calls = []
    for model in models:
        for pl in prelens:
            t = txt.replace('$model', model).replace('$preLen', str(pl))
            for line in t.split('\n'):
                line = line.strip()
                if not line.startswith('python'):
                    continue
                toks = shlex.split(line)
                args, i = {}, 0
                while i < len(toks):
                    if toks[i].startswith('--'):
                        k = toks[i][2:]
                        if i + 1 < len(toks) and not toks[i + 1].startswith('--'):
                            args[k] = toks[i + 1]
                            i += 2
                        else:
                            args[k] = 'True'      # flag argument
                            i += 1
                    else:
                        i += 1
                calls.append((model, int(pl), args))
    return calls


def main():
    os.makedirs(CFG, exist_ok=True)
    defaults = run_py_defaults()
    calls = parse_calls(os.path.join(OFF, 'scripts', 'run_M.sh'))

    # keep FEDformer, drop illness, one entry per (dataset, pred_len)
    kept, seen = [], set()
    for model, pl, args in calls:
        if model != 'FEDformer':
            continue
        dp = args.get('data_path', '')
        if dp in DROP:
            continue
        if dp not in DATASETS:
            continue
        key = (dp, pl)
        if key in seen:
            continue
        seen.add(key)
        kept.append((model, pl, args))

    # verbatim args for the record
    verbatim = [{'model': m, 'pred_len': pl,
                 'data_path': a.get('data_path'),
                 'argv': [x for kv in a.items() for x in (f'--{kv[0]}', kv[1])]}
                for m, pl, a in kept]
    json.dump(verbatim, open(os.path.join(CFG, 'official_script_args.json'), 'w'), indent=1)

    fields = ['job_id', 'dataset', 'pred_len', 'seed', 'script', 'data_subdir',
              'n_test_windows', 'ram_est_gb']
    for hp in HP:
        fields += [hp, hp + '_src']
    fields += ['official_args']

    rows = []
    for model, pl, args in sorted(kept, key=lambda x: (ORDER.index(DATASETS[x[2]['data_path']][0]), x[1])):
        dp = args['data_path']
        name, subdir, nrows = DATASETS[dp]
        # official args: everything the script sets, minus root_path (scheduler) and itr
        clean = {k: v for k, v in args.items() if k not in ('root_path', 'itr')}
        clean['itr'] = '1'
        argv = ' '.join(f'--{k} {v}' for k, v in clean.items())
        # estimate: the official test() keeps all predictions in RAM
        test_span = nrows - int(args.get('seq_len', defaults['seq_len'])) - pl + 1
        ram = max(2.0, round(test_span * int(args.get('enc_in', 7)) * pl * 8 / 1e9 + 2.0, 1))
        for seed in SEEDS:
            row = {'job_id': f'{name}_H{pl}_s{seed}', 'dataset': name, 'pred_len': pl,
                   'seed': seed, 'script': 'scripts/run_M.sh', 'data_subdir': subdir,
                   'n_test_windows': test_span, 'ram_est_gb': ram, 'official_args': argv}
            for hp in HP:
                row[hp] = args.get(hp, defaults.get(hp, ''))
                row[hp + '_src'] = 'script' if hp in args else 'run.py default'
            rows.append(row)

    with open(os.path.join(CFG, 'jobs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f'{len(kept)} (dataset, horizon) cells -> {len(rows)} jobs')
    for r in rows[:3]:
        print(' ', r['job_id'], '|', r['official_args'][:130])


if __name__ == '__main__':
    main()
