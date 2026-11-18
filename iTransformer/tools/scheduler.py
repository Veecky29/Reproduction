#!/usr/bin/env python3
"""Job scheduler for the iTransformer reproduction (single GPU).

* Runs official/run.py unchanged except for --seed (see patches/official_changes.diff).
* Every job gets its own working directory (runs/<job_id>/) because the official
  code writes ./checkpoints, ./results, ./test_results and result_long_term_forecast.txt
  relative to the CWD; separate CWDs keep concurrent seeds from overwriting each other.
* Up to --max-parallel jobs at once, and only while the summed RAM estimate
  (configs/jobs.csv: ram_est_gb; the official test() keeps all predictions in RAM)
  stays under --ram-budget-gb.
* Skip-completed: a job whose job_id already has status=ok in results.csv is not rerun.
* After a job finishes, metrics are read from the official metrics.npy
  ([mae, mse, rmse, mape, mspe]) and appended to results/results.csv; the large
  pred.npy/true.npy arrays and the checkpoint are deleted unless --keep-arrays.
* No automatic retries; failures are recorded with status=failed.
"""
import argparse, csv, datetime as dt, glob, json, os, re, shlex, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUN_PY = os.path.join(ROOT, 'official', 'run.py')
FIELDS = ['job_id', 'dataset', 'pred_len', 'seed', 'status', 'test_mse', 'test_mae', 'test_rmse',
          'epochs_ran', 'best_epoch', 'best_val_loss', 'early_stopped', 'elapsed_s', 'returncode',
          'start_local', 'end_local', 'gpu_name', 'setting']


def now():
    return dt.datetime.now().astimezone().isoformat(timespec='seconds')


def gpu_snapshot():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10).stdout
        n, u, m, t = [x.strip() for x in out.strip().split('\n')[0].split(',')]
        return {'name': n, 'util_pct': int(u), 'mem_used_mib': int(m), 'mem_total_mib': int(t)}
    except Exception as e:  # noqa
        return {'error': str(e)}


def load_done(path):
    if not os.path.exists(path):
        return set()
    return {r['job_id'] for r in csv.DictReader(open(path)) if r['status'] == 'ok'}


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, '') for k in FIELDS})


def parse_log(log):
    txt = open(log, errors='replace').read() if os.path.exists(log) else ''
    vals = [(int(e), float(v)) for e, v in re.findall(r'Epoch: (\d+), Steps: \d+ \| Train Loss: [\d.eE+-]+ Vali Loss: ([\d.eE+-]+)', txt)]
    best = min(vals, key=lambda x: x[1]) if vals else (None, None)
    setting = re.findall(r'>>>>>>>start training : (\S+?)>', txt)
    return {'epochs_ran': len(vals), 'best_epoch': best[0], 'best_val_loss': best[1],
            'early_stopped': 'Early stopping' in txt, 'setting': setting[0] if setting else ''}


def wait_for_pilot(status_json, poll):
    print(f'[{now()}] waiting for pilot to finish: {status_json}', flush=True)
    while True:
        try:
            s = json.load(open(status_json))
            if s.get('completed', 0) >= s.get('total', 1) and not s.get('active'):
                print(f'[{now()}] pilot finished ({s.get("completed")}/{s.get("total")}, state={s.get("state")})', flush=True)
                return
        except Exception as e:  # noqa
            print(f'[{now()}] could not read pilot status: {e}', flush=True)
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', default=os.path.join(ROOT, 'configs', 'jobs.csv'))
    ap.add_argument('--data-root', required=True, help='dir containing ETT-small/, electricity/, traffic/, ...')
    ap.add_argument('--work-dir', required=True, help='runs/ and results/ are created here')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--max-parallel', type=int, default=2)
    ap.add_argument('--ram-budget-gb', type=float, default=60.0)
    ap.add_argument('--datasets', default='', help='comma list to restrict, e.g. ETTh1,ECL')
    ap.add_argument('--keep-arrays', action='store_true', help='keep pred.npy/true.npy and checkpoints')
    ap.add_argument('--wait-pilot-status', default='', help='block until this pilot status.json reports completion')
    ap.add_argument('--poll', type=int, default=20)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    res_dir = os.path.join(a.work_dir, 'results'); os.makedirs(res_dir, exist_ok=True)
    runs_dir = os.path.join(a.work_dir, 'runs'); os.makedirs(runs_dir, exist_ok=True)
    res_csv = os.path.join(res_dir, 'results.csv')
    status_path = os.path.join(res_dir, 'status.json')
    jobs = list(csv.DictReader(open(a.jobs)))
    if a.datasets:
        keep = set(a.datasets.split(','))
        jobs = [j for j in jobs if j['dataset'] in keep]
    done = load_done(res_csv)
    queue = [j for j in jobs if j['job_id'] not in done]
    print(f'[{now()}] {len(jobs)} jobs, {len(done & {j["job_id"] for j in jobs})} already done, {len(queue)} queued', flush=True)

    def cmd_for(j):
        return ([a.python, '-u', RUN_PY] + shlex.split(j['official_args']) +
                ['--root_path', os.path.join(a.data_root, j['data_subdir']) + '/', '--seed', j['seed']])

    if a.dry_run:
        for j in queue:
            print(j['job_id'], 'ram_est_gb=%s' % j['ram_est_gb'], ' '.join(shlex.quote(x) for x in cmd_for(j)))
        return
    if a.wait_pilot_status:
        wait_for_pilot(a.wait_pilot_status, 60)

    active = {}
    t0 = time.time()
    n_total, n_ok, n_fail = len(jobs), len(done & {j['job_id'] for j in jobs}), 0
    while queue or active:
        # launch
        while queue and len(active) < a.max_parallel:
            j = queue[0]
            ram_used = sum(float(x['job']['ram_est_gb']) for x in active.values())
            if active and ram_used + float(j['ram_est_gb']) > a.ram_budget_gb:
                break
            queue.pop(0)
            wd = os.path.join(runs_dir, j['job_id'])
            if os.path.exists(wd):
                shutil.rmtree(wd)  # stale partial run of this job (never completed)
            os.makedirs(wd)
            log = os.path.join(wd, 'train.log')
            cmd = cmd_for(j)
            open(os.path.join(wd, 'cmd.txt'), 'w').write(' '.join(shlex.quote(x) for x in cmd) + '\n')
            p = subprocess.Popen(cmd, cwd=wd, stdout=open(log, 'w'), stderr=subprocess.STDOUT)
            active[j['job_id']] = {'job': j, 'p': p, 'start': time.time(), 'start_local': now(), 'wd': wd, 'log': log}
            print(f'[{now()}] start {j["job_id"]} pid={p.pid}', flush=True)
        # reap
        for jid in list(active):
            x = active[jid]
            rc = x['p'].poll()
            if rc is None:
                continue
            j, wd = x['job'], x['wd']
            row = {'job_id': jid, 'dataset': j['dataset'], 'pred_len': j['pred_len'], 'seed': j['seed'],
                   'returncode': rc, 'elapsed_s': round(time.time() - x['start'], 1),
                   'start_local': x['start_local'], 'end_local': now(), 'gpu_name': gpu_snapshot().get('name', '')}
            row.update(parse_log(x['log']))
            m = glob.glob(os.path.join(wd, 'results', '*', 'metrics.npy'))
            if rc == 0 and len(m) == 1:
                import numpy as np
                mae, mse, rmse, mape, mspe = np.load(m[0]).tolist()
                row.update(status='ok', test_mse=mse, test_mae=mae, test_rmse=rmse)
                n_ok += 1
                if not a.keep_arrays:
                    for f in glob.glob(os.path.join(wd, 'results', '*', 'pred.npy')) + \
                             glob.glob(os.path.join(wd, 'results', '*', 'true.npy')):
                        os.remove(f)
                    shutil.rmtree(os.path.join(wd, 'checkpoints'), ignore_errors=True)
            else:
                row['status'] = 'failed'
                n_fail += 1
            append_row(res_csv, row)
            print(f'[{now()}] end {jid} status={row["status"]} mse={row.get("test_mse")} mae={row.get("test_mae")} '
                  f'epochs={row["epochs_ran"]} {row["elapsed_s"]}s', flush=True)
            del active[jid]
        st = {'state': 'running' if (queue or active) else 'done', 'completed_ok': n_ok, 'failed': n_fail,
              'total': n_total, 'queued': len(queue), 'elapsed_s': round(time.time() - t0),
              'active': [{'job_id': k, 'pid': v['p'].pid, 'running_s': round(time.time() - v['start'])}
                         for k, v in active.items()],
              'gpu': gpu_snapshot(), 'local_time': now()}
        json.dump(st, open(status_path + '.tmp', 'w'), indent=1); os.replace(status_path + '.tmp', status_path)
        if queue or active:
            time.sleep(a.poll)
    print(f'[{now()}] all done: ok={n_ok} failed={n_fail} total={n_total}', flush=True)


if __name__ == '__main__':
    main()
