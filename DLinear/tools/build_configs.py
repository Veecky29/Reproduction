#!/usr/bin/env python3
"""Build the job list for the DLinear reproduction from the official
scripts (official/scripts/EXP-LongForecasting/Linear/*.sh) and the
argparse defaults in official/run_longExp.py.

The official long-forecasting scripts fix seq_len=336. PAC-Former and
iTransformer Table 10 (the numbers PAC-Former copies) use lookback 96.
This builder therefore substitutes seq_len=96 for every official call
(including the model_id string). That is the only intentional deviation
from the official scripts. Set LOOKBACK=336 to emit a verbatim official
job list instead. No hyperparameter search.

Outputs (configs/):
  official_script_args.json  every `python -u run_longExp.py ...` call, plus defaults
  jobs.csv                   8 datasets x 4 horizons x 3 seeds (no Solar: no official script or loader)
"""
import csv, glob, json, os, re, shlex

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OFF = os.path.join(ROOT, 'official')
CFG = os.path.join(ROOT, 'configs')
SEEDS = [2026, 2027, 2028]
LOOKBACK = int(os.environ.get('LOOKBACK', '96'))

N_ROWS = {'electricity.csv': 26304, 'traffic.csv': 17544, 'weather.csv': 52696,
          'exchange_rate.csv': 7587, 'ETTh1.csv': 17420, 'ETTh2.csv': 17420,
          'ETTm1.csv': 69680, 'ETTm2.csv': 69680}
DATASET_NAME = {'electricity.csv': 'ECL', 'traffic.csv': 'Traffic', 'weather.csv': 'Weather',
                'exchange_rate.csv': 'Exchange', 'ETTh1.csv': 'ETTh1', 'ETTh2.csv': 'ETTh2',
                'ETTm1.csv': 'ETTm1', 'ETTm2.csv': 'ETTm2'}
DATA_SUBDIR = {'electricity.csv': 'electricity', 'traffic.csv': 'traffic', 'weather.csv': 'weather',
               'exchange_rate.csv': 'exchange_rate', 'ETTh1.csv': 'ETT-small', 'ETTh2.csv': 'ETT-small',
               'ETTm1.csv': 'ETT-small', 'ETTm2.csv': 'ETT-small'}
ORDER = ['ETTh1', 'ETTh2', 'Exchange', 'ETTm1', 'ETTm2', 'Weather', 'ECL', 'Traffic']
# ili.sh and univariate/ are excluded; there is no official Solar script
SKIP_SCRIPTS = {'ili.sh'}
HP = ['seq_len', 'pred_len', 'enc_in', 'batch_size', 'learning_rate', 'train_epochs',
      'patience', 'lradj', 'individual', 'moving_avg', 'features', 'itr']


def run_py_defaults():
    src = open(os.path.join(OFF, 'run_longExp.py')).read()
    d = {}
    for line in src.split('\n'):
        m = re.search(r"add_argument\('--(\w+)'.*?default=([^,]+),", line)
        if m:
            val = m.group(2).strip().strip("'\"")
            if m.group(1) == 'individual' or 'store_true' in line:
                val = 'False' if 'store_true' in line and 'default=False' in line else val
            d[m.group(1)] = val
    return d


def parse_scripts():
    calls = []
    pattern = os.path.join(OFF, 'scripts/EXP-LongForecasting/Linear/*.sh')
    for f in sorted(glob.glob(pattern)):
        if os.path.basename(f) in SKIP_SCRIPTS:
            continue
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
            env_use = dict(env)
            # official scripts set seq_len=336; override for the PAC-Former / Table 10 protocol
            env_use['seq_len'] = str(LOOKBACK)
            for k, v in env_use.items():
                line = line.replace('$' + k, v)
            toks = shlex.split(line)
            toks = toks[toks.index('run_longExp.py') + 1:]
            # drop shell redirections (`>logs/...` or `> path`) so they are not parsed as flags
            cleaned, i = [], 0
            while i < len(toks):
                tok = toks[i]
                if tok in ('>', '>>', '2>', '2>>', '&>'):
                    i += 2
                    continue
                if tok.startswith('>') or tok.startswith('2>'):
                    i += 1
                    continue
                cleaned.append(tok); i += 1
            toks = cleaned
            args, i = {}, 0
            while i < len(toks):
                k = toks[i][2:]
                if i + 1 < len(toks) and not toks[i + 1].startswith('--'):
                    args[k] = toks[i + 1]; i += 2
                else:
                    args[k] = True; i += 1
            calls.append({'script': os.path.relpath(f, OFF), 'args': args,
                          'seq_len_overridden': LOOKBACK != int(env.get('seq_len', LOOKBACK))})
    return calls


def n_test_windows(data, fname, seq_len, pred_len):
    n = N_ROWS[fname]
    if data in ('ETTh1', 'ETTh2'):
        return 4 * 30 * 24 - pred_len + 1
    if data in ('ETTm1', 'ETTm2'):
        return 4 * 30 * 24 * 4 - pred_len + 1
    return int(n * 0.2) - pred_len + 1  # Dataset_Custom: 70/10/20


def main():
    defaults = run_py_defaults()
    calls = parse_scripts()
    os.makedirs(CFG, exist_ok=True)
    json.dump({'lookback': LOOKBACK, 'defaults_from_run_longExp_py': defaults, 'calls': calls},
              open(os.path.join(CFG, 'official_script_args.json'), 'w'), indent=1)
    rows = []
    for c in calls:
        a = c['args']
        eff = dict(defaults); eff.update(a)
        # individual is a store_true flag
        if 'individual' in a:
            eff['individual'] = '1'
        else:
            eff['individual'] = '0'
        fname = a['data_path']
        ds = DATASET_NAME[fname]
        H, C = int(eff['pred_len']), int(eff['enc_in'])
        L = int(eff['seq_len'])
        nt = n_test_windows(eff['data'], fname, L, H)
        # test() keeps preds, trues and inputx as lists then concatenates (~6 float32 copies of preds/trues, plus inputx)
        ram_gb = round(6 * nt * H * C * 4 / 1e9 + 2 * nt * L * C * 4 / 1e9 + 3.0, 1)
        argv = []
        for k, v in a.items():
            if k == 'root_path':
                continue
            argv += ['--' + k] if v is True else ['--' + k, str(v)]
        for seed in SEEDS:
            r = {'job_id': f"{ds}_H{H}_s{seed}", 'dataset': ds, 'pred_len': H, 'seed': seed,
                 'script': c['script'], 'data_subdir': DATA_SUBDIR[fname],
                 'n_test_windows': nt, 'ram_est_gb': ram_gb}
            for k in HP:
                if k == 'pred_len':
                    continue
                r[k] = eff[k]
                if k == 'seq_len' and c['seq_len_overridden']:
                    r[k + '_src'] = f'script had {336}; overridden to {LOOKBACK} (PAC-Former / iTransformer Table 10 protocol)'
                elif k in a:
                    r[k + '_src'] = 'script'
                else:
                    r[k + '_src'] = 'run_longExp.py default'
            r['official_args'] = ' '.join(shlex.quote(x) for x in argv)
            rows.append(r)
    rows.sort(key=lambda r: (ORDER.index(r['dataset']), r['pred_len'], r['seed']))
    out = os.path.join(CFG, 'jobs.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print('jobs.csv', len(rows), 'jobs; lookback', LOOKBACK)


if __name__ == '__main__':
    main()
