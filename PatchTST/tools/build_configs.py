#!/usr/bin/env python3
"""Build the job list for the PatchTST reproduction from the official
scripts (official/scripts/PatchTST/*.sh) and the argparse defaults in
official/run_longExp.py.

The official supervised scripts fix seq_len=336 (PatchTST/42). PAC-Former and
iTransformer Table 10 (the numbers PAC-Former copies) use lookback 96.
This builder therefore substitutes seq_len=96 for every official call
(including the model_id string). That is the only intentional deviation
from the official scripts for datasets that have a script.

Set LOOKBACK=336 to emit a verbatim official-lookback job list instead.
Set INCLUDE_TRAFFIC=1 to also emit Traffic jobs (excluded by default).
Set INCLUDE_EXCHANGE=1 to also emit Exchange jobs (excluded by default;
there is no official PatchTST script — see proposed_exchange in
official_script_args.json for the documented ETTh-style recipe).

Outputs (configs/):
  official_script_args.json  every official `python -u run_longExp.py ...` call,
                             plus defaults and the unscheduled Exchange proposal
  jobs.csv                   6 datasets x 4 horizons x 3 seeds = 72 jobs
                             (ETTh1, ETTh2, ETTm1, ETTm2, Weather, ECL)
"""
import csv, glob, json, os, re, shlex

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OFF = os.path.join(ROOT, 'official')
CFG = os.path.join(ROOT, 'configs')
SEEDS = [2026, 2027, 2028]
LOOKBACK = int(os.environ.get('LOOKBACK', '96'))
INCLUDE_TRAFFIC = os.environ.get('INCLUDE_TRAFFIC', '0') == '1'
INCLUDE_EXCHANGE = os.environ.get('INCLUDE_EXCHANGE', '0') == '1'

N_ROWS = {'electricity.csv': 26304, 'traffic.csv': 17544, 'weather.csv': 52696,
          'exchange_rate.csv': 7588, 'ETTh1.csv': 17420, 'ETTh2.csv': 17420,
          'ETTm1.csv': 69680, 'ETTm2.csv': 69680}
# exchange_rate.csv is 7588 lines including the header in the shared bundle
# (7587 data rows). Dataset_Custom uses len(df_raw), i.e. 7588? Need to be careful.
# DLinear used 7587. I'll keep 7588 or 7587 consistently.
# DLinear N_ROWS exchange_rate.csv: 7587. The SHA256 file is the same bundle.
# int(7587 * 0.2) = 1517, test windows = 1517 - pred_len + 1. I'll use 7587 like DLinear.
N_ROWS['exchange_rate.csv'] = 7587
DATASET_NAME = {'electricity.csv': 'ECL', 'traffic.csv': 'Traffic', 'weather.csv': 'Weather',
                'exchange_rate.csv': 'Exchange', 'ETTh1.csv': 'ETTh1', 'ETTh2.csv': 'ETTh2',
                'ETTm1.csv': 'ETTm1', 'ETTm2.csv': 'ETTm2'}
DATA_SUBDIR = {'electricity.csv': 'electricity', 'traffic.csv': 'traffic', 'weather.csv': 'weather',
               'exchange_rate.csv': 'exchange_rate', 'ETTh1.csv': 'ETT-small', 'ETTh2.csv': 'ETT-small',
               'ETTm1.csv': 'ETT-small', 'ETTm2.csv': 'ETT-small'}
# Scheduled job order: small datasets first, ECL last. Traffic and Exchange are not scheduled by default.
ORDER = ['ETTh1', 'ETTh2', 'Exchange', 'ETTm1', 'ETTm2', 'Weather', 'ECL', 'Traffic']
SKIP_SCRIPTS = {'illness.sh'}
# Hyperparameters recorded in jobs.csv (pred_len is a top-level column).
HP = ['seq_len', 'enc_in', 'e_layers', 'n_heads', 'd_model', 'd_ff',
      'dropout', 'fc_dropout', 'head_dropout', 'patch_len', 'stride',
      'padding_patch', 'revin', 'individual', 'batch_size', 'learning_rate',
      'train_epochs', 'patience', 'lradj', 'pct_start', 'features', 'itr']


def run_py_defaults():
    src = open(os.path.join(OFF, 'run_longExp.py')).read()
    d = {}
    for line in src.split('\n'):
        m = re.search(r"add_argument\('--(\w+)'.*?default=([^,)]+)", line)
        if m:
            val = m.group(2).strip().strip("'\"")
            d[m.group(1)] = val
    return d


def _subst_env(line, env):
    # Replace $var / ${var}. Longest names first so $model_id_name is not
    # eaten by a shorter prefix. This also yields the *intended* model_id
    # (ETTh1_96_96). Official bash does not: $model_id_name_$seq_len is
    # parsed as ${model_id_name_}${seq_len} (unset prefix), so the running
    # official script actually sets --model_id 336_96. model_id only names
    # checkpoints / the setting string; metrics are unaffected. We use the
    # intended name after the lookback override.
    names = sorted(env, key=len, reverse=True)
    for k in names:
        line = line.replace('${' + k + '}', env[k])
        line = line.replace('$' + k, env[k])
    return line


def parse_scripts():
    calls = []
    pattern = os.path.join(OFF, 'scripts/PatchTST/*.sh')
    for f in sorted(glob.glob(pattern)):
        if os.path.basename(f) in SKIP_SCRIPTS:
            continue
        raw = open(f).read()
        official_seq = None
        mseq = re.search(r'^seq_len=(\d+)', raw, re.M)
        if mseq:
            official_seq = int(mseq.group(1))
        # Official PatchTST scripts write `python ... \` continuations and
        # wrap the four horizons in `for pred_len in 96 192 336 720`.
        txt = raw.replace('\\\n', ' ')
        env = {}
        loop_name, loop_vals = None, [None]
        for line in txt.split('\n'):
            line = line.strip()
            mfor = re.match(r'^for (\w+) in ([\d\s]+)$', line)
            if mfor:
                loop_name = mfor.group(1)
                loop_vals = mfor.group(2).split()
                continue
            if line in ('do', 'done'):
                if line == 'done':
                    loop_name, loop_vals = None, [None]
                continue
            m = re.match(r'^(\w+)=(\S+)$', line)
            if m:
                env[m.group(1)] = m.group(2)
                continue
            if not line.startswith('python'):
                continue
            values = loop_vals if loop_name else [None]
            for lv in values:
                env_use = dict(env)
                env_use['seq_len'] = str(LOOKBACK)
                if loop_name and lv is not None:
                    env_use[loop_name] = lv
                pline = _subst_env(line, env_use)
                toks = shlex.split(pline)
                toks = toks[toks.index('run_longExp.py') + 1:]
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
                # scheduler supplies --random_seed; drop the hard-coded 2021
                args.pop('random_seed', None)
                if '$' in str(args):
                    raise ValueError(f'unexpanded shell variable in {f}: {args}')
                calls.append({'script': os.path.relpath(f, OFF), 'args': args,
                              'seq_len_overridden': official_seq is not None and LOOKBACK != official_seq,
                              'official_script_seq_len': official_seq})
    return calls


def exchange_proposal(defaults):
    """ETTh-style small model; no official PatchTST Exchange script exists.
    The official PatchTST paper (arXiv:2211.14730) explicitly omits Exchange.
    Closest official recipe is etth1.sh / etth2.sh (7 channels, small d_model).
    Exchange has 8 channels and fewer timesteps than ETTh.
    """
    calls = []
    for pred in (96, 192, 336, 720):
        args = {
            'is_training': '1',
            'root_path': './dataset/',
            'data_path': 'exchange_rate.csv',
            'model_id': f'Exchange_{LOOKBACK}_{pred}',
            'model': 'PatchTST',
            'data': 'custom',
            'features': 'M',
            'seq_len': str(LOOKBACK),
            'pred_len': str(pred),
            'enc_in': '8',
            'e_layers': '3',
            'n_heads': '4',
            'd_model': '16',
            'd_ff': '128',
            'dropout': '0.3',
            'fc_dropout': '0.3',
            'head_dropout': '0',
            'patch_len': '16',
            'stride': '8',
            'des': 'Exp',
            'train_epochs': '100',
            'itr': '1',
            'batch_size': '128',
            'learning_rate': '0.0001',
        }
        # patience / lradj left at run_longExp.py defaults (100 / type3), as in etth1.sh
        calls.append({
            'script': 'proposed_from:scripts/PatchTST/etth1.sh',
            'args': args,
            'seq_len_overridden': True,
            'official_script_seq_len': 336,
            'proposed': True,
            'proposal': 'ETTh-style small model (d_model=16, n_heads=4, d_ff=128, dropout=0.3, '
                        'batch=128, lr=0.0001, epochs=100, patience=100 default, lradj=type3 default). '
                        'Needs user confirmation. Alternative considered: ETTm-style (d_model=128, '
                        'n_heads=16, d_ff=256, dropout=0.2, patience=20, lradj=TST, pct_start=0.4).',
        })
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
    exch = exchange_proposal(defaults)
    os.makedirs(CFG, exist_ok=True)
    json.dump({
        'lookback': LOOKBACK,
        'include_traffic': INCLUDE_TRAFFIC,
        'include_exchange': INCLUDE_EXCHANGE,
        'defaults_from_run_longExp_py': defaults,
        'calls': calls,
        'proposed_exchange': {
            'status': 'not_scheduled' if not INCLUDE_EXCHANGE else 'included',
            'recipe': 'ETTh-style small model, copied from scripts/PatchTST/etth1.sh',
            'reason': 'No official scripts/PatchTST script for Exchange. Official PatchTST paper '
                      'omits Exchange-rate. Exchange has 8 channels (ETT has 7) and is smaller '
                      'than ETTh, so the ETTh-style d_model=16 recipe is the closest official match. '
                      'ETTm-style (d_model=128, lradj=TST, patience=20) is the main alternative. '
                      'User confirmed Exchange is not scheduled (INCLUDE_EXCHANGE=0 by default). '
                      'Set INCLUDE_EXCHANGE=1 to emit the 12 Exchange jobs.',
            'calls': exch,
        },
        'not_scheduled': {
            'Exchange': 'No official script. ETTh-style proposal is recorded in proposed_exchange, '
                        'but Exchange jobs are omitted from the default jobs.csv. '
                        'Set INCLUDE_EXCHANGE=1 to emit them.',
            'Traffic': 'Official script scripts/PatchTST/traffic.sh is recorded in calls, but '
                       'Traffic jobs are omitted from the default jobs.csv (VM rental ends around '
                       '2026-11-22). Set INCLUDE_TRAFFIC=1 to emit them.',
        },
    }, open(os.path.join(CFG, 'official_script_args.json'), 'w'), indent=1)

    scheduled_calls = list(calls)
    if INCLUDE_EXCHANGE:
        scheduled_calls.extend(exch)
    rows = []
    for c in scheduled_calls:
        a = c['args']
        fname = a['data_path']
        ds = DATASET_NAME[fname]
        if ds == 'Traffic' and not INCLUDE_TRAFFIC:
            continue
        eff = dict(defaults); eff.update(a)
        H, C = int(eff['pred_len']), int(eff['enc_in'])
        L = int(eff['seq_len'])
        B = int(eff['batch_size'])
        nt = n_test_windows(eff['data'], fname, L, H)
        # Official data_factory: test uses training batch_size and drop_last=True,
        # so the trailing incomplete batch is not evaluated.
        n_eval = (nt // B) * B
        n_dropped = nt - n_eval
        # test() keeps preds, trues and inputx as lists then concatenates
        # (~6 float32 copies of preds/trues, plus inputx). +4 GB process overhead
        # so two ECL H720 jobs (~30 GB each) will not be co-scheduled under a 60 GB budget.
        ram_gb = round(6 * nt * H * C * 4 / 1e9 + 2 * nt * L * C * 4 / 1e9 + 4.0, 1)
        argv = []
        for k, v in a.items():
            if k in ('root_path', 'random_seed'):
                continue
            argv += ['--' + k] if v is True else ['--' + k, str(v)]
        official_seq = c.get('official_script_seq_len', 336)
        for seed in SEEDS:
            r = {'job_id': f"{ds}_H{H}_s{seed}", 'dataset': ds, 'pred_len': H, 'seed': seed,
                 'script': c['script'], 'data_subdir': DATA_SUBDIR[fname],
                 'n_test_windows': nt, 'n_test_eval': n_eval, 'n_test_dropped': n_dropped,
                 'ram_est_gb': ram_gb, 'proposed': '1' if c.get('proposed') else '0'}
            for k in HP:
                r[k] = eff[k]
                if k == 'seq_len' and c.get('seq_len_overridden'):
                    if c.get('proposed'):
                        r[k + '_src'] = (f'no official script; proposed from etth1.sh '
                                         f'(script had {official_seq}); overridden to {LOOKBACK} '
                                         f'(PAC-Former / iTransformer Table 10 protocol); needs confirmation')
                    else:
                        r[k + '_src'] = (f'script had {official_seq}; overridden to {LOOKBACK} '
                                         f'(PAC-Former / iTransformer Table 10 protocol)')
                elif c.get('proposed') and k in a:
                    r[k + '_src'] = 'proposed from etth1.sh (no official Exchange script; needs confirmation)'
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
    print('jobs.csv', len(rows), 'jobs; lookback', LOOKBACK,
          '; traffic' , 'included' if INCLUDE_TRAFFIC else 'excluded',
          '; exchange', 'included (proposed)' if INCLUDE_EXCHANGE else 'excluded')


if __name__ == '__main__':
    main()
