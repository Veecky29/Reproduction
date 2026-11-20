#!/usr/bin/env python3
"""Build the job list for the RLinear reproduction from official/script.md
and the argparse defaults in official/run.py.

Official script.md fixes seq_len=336 and only lists pred_len=96, and it
passes --model Linear (the code at this commit only registers RLinear; an
alias is added in patches/official_changes.diff). PAC-Former and
iTransformer Table 10 (the numbers PAC-Former copies) use lookback 96 and
horizons 96/192/336/720. This builder therefore:

  * substitutes seq_len=96 for every official call
  * expands each official pred_len=96 example to 96/192/336/720 with the
    same hyperparameters
  * emits --model RLinear (the Linear alias remains so a verbatim
    script.md command would also run)

Set LOOKBACK=336 to keep the official lookback (horizons are still
expanded). There is no official script for Exchange or Traffic; those
jobs use a documented proposal (Exchange follows ETTh1, Traffic follows
ECL) and are marked proposed=1. No hyperparameter search.

Outputs (configs/):
  official_script_args.json  parsed script.md calls, defaults, proposals
  jobs.csv                   8 datasets x 4 horizons x 3 seeds = 96 jobs
"""
import csv, json, os, re, shlex

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OFF = os.path.join(ROOT, 'official')
CFG = os.path.join(ROOT, 'configs')
SEEDS = [2026, 2027, 2028]
LOOKBACK = int(os.environ.get('LOOKBACK', '96'))
HORIZONS = [96, 192, 336, 720]

N_ROWS = {'electricity.csv': 26304, 'traffic.csv': 17544, 'weather.csv': 52696,
          'exchange_rate.csv': 7587, 'ETTh1.csv': 17420, 'ETTh2.csv': 17420,
          'ETTm1.csv': 69680, 'ETTm2.csv': 69680}
DATASET_NAME = {'electricity.csv': 'ECL', 'traffic.csv': 'Traffic', 'weather.csv': 'Weather',
                'exchange_rate.csv': 'Exchange', 'ETTh1.csv': 'ETTh1', 'ETTh2.csv': 'ETTh2',
                'ETTm1.csv': 'ETTm1', 'ETTm2.csv': 'ETTm2'}
DATA_SUBDIR = {'electricity.csv': 'electricity', 'traffic.csv': 'traffic', 'weather.csv': 'weather',
               'exchange_rate.csv': 'exchange_rate', 'ETTh1.csv': 'ETT-small', 'ETTh2.csv': 'ETT-small',
               'ETTm1.csv': 'ETT-small', 'ETTm2.csv': 'ETT-small'}
# Official script.md uses Weather.csv / ECL.csv; the shared dataset bundle uses these names.
DATA_PATH_MAP = {'Weather.csv': 'weather.csv', 'ECL.csv': 'electricity.csv'}
ORDER = ['ETTh1', 'ETTh2', 'Exchange', 'ETTm1', 'ETTm2', 'Weather', 'ECL', 'Traffic']
# Official_args flag order (deterministic). seed, root_path, gpu and num_workers are left to the scheduler.
ARG_ORDER = ['is_training', 'model', 'data', 'data_path', 'features', 'channel', 'itr',
             'batch_size', 'learning_rate', 'drop', 'seq_len', 'pred_len']
FLAG_ORDER = ['rev', 'individual']
HP = ['seq_len', 'channel', 'batch_size', 'learning_rate', 'drop', 'train_epochs',
      'patience', 'lradj', 'rev', 'individual', 'features', 'itr']


def run_py_defaults():
    src = open(os.path.join(OFF, 'run.py'), encoding='utf-8').read()
    d = {}
    for line in src.split('\n'):
        m = re.search(r"add_argument\('--(\w+)'.*?default=([^,]+),", line)
        if m:
            val = m.group(2).strip().strip("'\"")
            if 'store_true' in line:
                val = 'False'
            d[m.group(1)] = val
    return d


def parse_flag_toks(toks):
    args = {}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith('--'):
            k = t[2:]
            if i + 1 < len(toks) and not toks[i + 1].startswith('--'):
                args[k] = toks[i + 1]
                i += 2
            else:
                args[k] = True
                i += 1
        else:
            i += 1
    return args


def parse_scripts():
    calls = []
    txt = open(os.path.join(OFF, 'script.md'), encoding='utf-8').read()
    for line in txt.splitlines():
        line = line.strip()
        if not line.startswith('python'):
            continue
        toks = shlex.split(line)
        toks = toks[toks.index('run.py') + 1:]
        args = parse_flag_toks(toks)
        calls.append({'script': 'script.md', 'args': args, 'raw': line})
    return calls


def n_test_windows(data, fname, seq_len, pred_len):
    n = N_ROWS[fname]
    if data in ('ETTh1', 'ETTh2'):
        return 4 * 30 * 24 - pred_len + 1
    if data in ('ETTm1', 'ETTm2'):
        return 4 * 30 * 24 * 4 - pred_len + 1
    return int(n * 0.2) - pred_len + 1  # Dataset_Custom: 70/10/20


def argv_from(job_args):
    argv = []
    for k in ARG_ORDER:
        if k not in job_args:
            continue
        v = job_args[k]
        argv += ['--' + k, str(v)]
    for k in FLAG_ORDER:
        if job_args.get(k) is True or job_args.get(k) in ('1', 'True', True):
            argv += ['--' + k]
    return argv


def source_note(k, job_args, official_seq, proposed, proposal_from):
    if k == 'seq_len':
        if proposed:
            return (f'no official script; proposed from {proposal_from} '
                    f'(script had {official_seq}); overridden to {LOOKBACK} '
                    f'(PAC-Former / iTransformer Table 10 protocol); non-official choice')
        return (f'script had {official_seq}; overridden to {LOOKBACK} '
                f'(PAC-Former / iTransformer Table 10 protocol)')
    if proposed:
        if k in job_args or k == 'model':
            return f'proposed from {proposal_from} (no official script; non-official choice)'
        return 'run.py default; proposed job (non-official choice)'
    if k == 'model':
        return 'script had Linear; emitted as RLinear (see patches/official_changes.diff)'
    if k in job_args:
        return 'script.md'
    return 'run.py default'


def add_jobs(rows, defaults, base_args, script, official_seq, proposed=False, proposal_from=''):
    fname_official = base_args['data_path']
    fname = DATA_PATH_MAP.get(fname_official, fname_official)
    ds = DATASET_NAME[fname]
    for H in HORIZONS:
        job_args = dict(base_args)
        job_args['data_path'] = fname
        job_args['seq_len'] = str(LOOKBACK)
        job_args['pred_len'] = str(H)
        job_args['model'] = 'RLinear'
        job_args.pop('gpu', None)
        job_args.pop('root_path', None)
        eff = dict(defaults)
        eff.update({k: v for k, v in job_args.items()})
        eff['rev'] = '1' if job_args.get('rev') else '0'
        eff['individual'] = '1' if job_args.get('individual') else '0'
        C = int(eff['channel'])
        L = int(eff['seq_len'])
        nt = n_test_windows(eff['data'], fname, L, H)
        ram_gb = round(6 * nt * H * C * 4 / 1e9 + 2 * nt * L * C * 4 / 1e9 + 3.0, 1)
        argv = argv_from(job_args)
        for seed in SEEDS:
            r = {'job_id': f'{ds}_H{H}_s{seed}', 'dataset': ds, 'pred_len': H, 'seed': seed,
                 'script': script, 'data_subdir': DATA_SUBDIR[fname],
                 'n_test_windows': nt, 'ram_est_gb': ram_gb,
                 'proposed': '1' if proposed else '0'}
            for k in HP:
                r[k] = eff[k]
                r[k + '_src'] = source_note(k, job_args, official_seq, proposed, proposal_from)
            r['official_args'] = ' '.join(shlex.quote(x) for x in argv)
            rows.append(r)


def main():
    defaults = run_py_defaults()
    calls = parse_scripts()
    # Exchange: no official script; follow ETTh1 (8 vs 7 channels, small data).
    etth1 = next(c['args'] for c in calls if c['args'].get('data_path') == 'ETTh1.csv')
    exchange_args = dict(etth1)
    exchange_args.update({'data': 'custom', 'data_path': 'exchange_rate.csv', 'channel': '8'})
    # Traffic: no official script; follow ECL (large multivariate).
    ecl = next(c['args'] for c in calls if c['args'].get('data_path') in ('ECL.csv', 'electricity.csv'))
    traffic_args = dict(ecl)
    traffic_args.update({'data': 'custom', 'data_path': 'traffic.csv', 'channel': '862'})

    os.makedirs(CFG, exist_ok=True)
    json.dump({
        'lookback': LOOKBACK,
        'horizons': HORIZONS,
        'defaults_from_run_py': defaults,
        'official_calls': calls,
        'notes': {
            'official_seq_len': 336,
            'official_pred_len_only': 96,
            'official_model_flag': 'Linear',
            'emitted_model_flag': 'RLinear',
            'data_path_map': DATA_PATH_MAP,
            'gpu': 'script.md sets --gpu 3 (authors machine); jobs leave the run.py default (0)',
            'num_workers': 'official default 10; scheduler --num-workers can override (use 2 when sharing the GPU)',
            'license': 'plumprc/RTSF declares no license at this commit',
        },
        'proposed_exchange': {
            'reason': 'no official script.md entry for Exchange; follows the ETTh1 official recipe (batch 128, lr 0.005, drop 0.1, --rev, shared). Non-official choice.',
            'from': 'ETTh1 / script.md',
            'args': exchange_args,
        },
        'proposed_traffic': {
            'reason': 'no official script.md entry for Traffic; follows the ECL official recipe (batch default 16, lr 0.001, drop 0, --rev, shared). Non-official choice.',
            'from': 'ECL / script.md',
            'args': traffic_args,
        },
    }, open(os.path.join(CFG, 'official_script_args.json'), 'w', encoding='utf-8'), indent=1)

    rows = []
    for c in calls:
        official_seq = c['args'].get('seq_len', '336')
        add_jobs(rows, defaults, c['args'], 'script.md', official_seq, proposed=False)
    add_jobs(rows, defaults, exchange_args, 'proposed from ETTh1 (no official Exchange script)',
             etth1.get('seq_len', '336'), proposed=True, proposal_from='ETTh1 / script.md')
    add_jobs(rows, defaults, traffic_args, 'proposed from ECL (no official Traffic script)',
             ecl.get('seq_len', '336'), proposed=True, proposal_from='ECL / script.md')
    rows.sort(key=lambda r: (ORDER.index(r['dataset']), int(r['pred_len']), int(r['seed'])))
    out = os.path.join(CFG, 'jobs.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator='\n')
        w.writeheader(); w.writerows(rows)
    print('jobs.csv', len(rows), 'jobs; lookback', LOOKBACK,
          '; datasets', ' '.join(sorted({r['dataset'] for r in rows}, key=ORDER.index)))


if __name__ == '__main__':
    main()
