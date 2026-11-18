#!/usr/bin/env python3
"""Build the job list for the iTransformer reproduction directly from the
official scripts (official/scripts/multivariate_forecasting/*/*.sh) and the
argparse defaults in official/run.py. Nothing is hand-typed: every
hyperparameter comes either from the script line or from the run.py default.

Outputs (configs/):
  official_script_args.json  every `python -u run.py ...` call, verbatim args
  jobs.csv                   long-term benchmark (Table 10): 9 datasets x 4 horizons x seeds
  jobs_pems.csv              PEMS (Table 9, horizons 12/24/48/96) -- optional, not in default run
"""
import csv, glob, json, os, re, shlex

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OFF = os.path.join(ROOT, 'official')
CFG = os.path.join(ROOT, 'configs')
SEEDS = [2026, 2027, 2028]

# rows in each data file (excluding header) -> used only for the RAM estimate
N_ROWS = {'electricity.csv': 26304, 'traffic.csv': 17544, 'weather.csv': 52696,
          'exchange_rate.csv': 7587, 'solar_AL.txt': 52560, 'ETTh1.csv': 17420,
          'ETTh2.csv': 17420, 'ETTm1.csv': 69680, 'ETTm2.csv': 69680,
          'PEMS03.npz': 26208, 'PEMS04.npz': 16992, 'PEMS07.npz': 28224, 'PEMS08.npz': 17856}
DATASET_NAME = {'electricity.csv': 'ECL', 'traffic.csv': 'Traffic', 'weather.csv': 'Weather',
                'exchange_rate.csv': 'Exchange', 'solar_AL.txt': 'Solar-Energy',
                'ETTh1.csv': 'ETTh1', 'ETTh2.csv': 'ETTh2', 'ETTm1.csv': 'ETTm1', 'ETTm2.csv': 'ETTm2',
                'PEMS03.npz': 'PEMS03', 'PEMS04.npz': 'PEMS04', 'PEMS07.npz': 'PEMS07', 'PEMS08.npz': 'PEMS08'}
ORDER = ['ETTh1', 'ETTh2', 'Exchange', 'ETTm1', 'ETTm2', 'Weather', 'ECL', 'Solar-Energy', 'Traffic',
         'PEMS08', 'PEMS04', 'PEMS03', 'PEMS07']
HP = ['seq_len', 'pred_len', 'e_layers', 'd_model', 'd_ff', 'n_heads', 'batch_size', 'learning_rate',
      'train_epochs', 'patience', 'lradj', 'dropout', 'use_norm', 'enc_in']


def run_py_defaults():
    src = open(os.path.join(OFF, 'run.py')).read()
    d = {}
    for line in src.split('\n'):
        m = re.search(r"add_argument\('--(\w+)'.*?default=([^,]+),", line)
        if m:
            d[m.group(1)] = m.group(2).strip().strip("'\"")
    return d


def parse_scripts():
    calls = []
    for f in sorted(glob.glob(os.path.join(OFF, 'scripts/multivariate_forecasting/*/*.sh'))):
        txt = open(f).read().replace('\\\n', ' ')
        env = {}
        for line in txt.split('\n'):
            line = line.strip()
            m = re.match(r'^(\w+)=(\S+)$', line)
            if m:
                env[m.group(1)] = m.group(2)
                continue
            if not line.startswith('python'):
                continue
            for k, v in env.items():
                line = line.replace('$' + k, v)
            toks = shlex.split(line)
            toks = toks[toks.index('run.py') + 1:]
            args, i = {}, 0
            while i < len(toks):
                k = toks[i][2:]
                if i + 1 < len(toks) and not toks[i + 1].startswith('--'):
                    args[k] = toks[i + 1]; i += 2
                else:
                    args[k] = True; i += 1
            calls.append({'script': os.path.relpath(f, OFF), 'args': args})
    return calls


def n_test_windows(data, fname, seq_len, pred_len):
    n = N_ROWS[fname]
    if data in ('ETTh1', 'ETTh2'):
        return 4 * 30 * 24 + seq_len - seq_len - pred_len + 1
    if data in ('ETTm1', 'ETTm2'):
        return 4 * 30 * 24 * 4 - pred_len + 1
    if data == 'PEMS':
        return n - int(0.8 * n) - seq_len - pred_len + 1
    return int(n * 0.2) - pred_len + 1  # custom / Solar


def main():
    defaults = run_py_defaults()
    calls = parse_scripts()
    json.dump({'defaults_from_run_py': defaults, 'calls': calls},
              open(os.path.join(CFG, 'official_script_args.json'), 'w'), indent=1)
    rows = {'jobs.csv': [], 'jobs_pems.csv': []}
    for c in calls:
        a = c['args']
        eff = dict(defaults); eff.update(a)
        fname = a['data_path']
        ds = DATASET_NAME[fname]
        H, C = int(eff['pred_len']), int(eff['enc_in'])
        nt = n_test_windows(eff['data'], fname, int(eff['seq_len']), H)
        # official test(): preds+trues kept as python lists, then np.array copy -> ~4 float32 copies
        ram_gb = round(4 * nt * H * C * 4 / 1e9 + 3.0, 1)
        sub = os.path.relpath(os.path.normpath(a['root_path']), 'dataset')
        argv = []
        for k, v in a.items():
            if k == 'root_path':
                continue
            argv += ['--' + k] if v is True else ['--' + k, str(v)]
        for seed in SEEDS:
            r = {'job_id': f"{ds}_H{H}_s{seed}", 'dataset': ds, 'pred_len': H, 'seed': seed,
                 'script': c['script'], 'data_subdir': sub, 'n_test_windows': nt, 'ram_est_gb': ram_gb}
            for k in HP:
                if k == 'pred_len':
                    continue
                r[k] = eff[k]
                r[k + '_src'] = 'script' if k in a else 'run.py default'
            r['official_args'] = ' '.join(shlex.quote(x) for x in argv)
            rows['jobs_pems.csv' if ds.startswith('PEMS') else 'jobs.csv'].append(r)
    for name, rs in rows.items():
        rs.sort(key=lambda r: (ORDER.index(r['dataset']), r['pred_len'], r['seed']))
        with open(os.path.join(CFG, name), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0].keys()))
            w.writeheader(); w.writerows(rs)
        print(name, len(rs), 'jobs')


if __name__ == '__main__':
    main()
