#!/usr/bin/env python3
"""Aggregate per-seed results into mean ± std (sample std, ddof=1) per dataset/horizon,
compare with the paper reference (configs/paper_reference.csv) and optionally write the
table into the root README between the PATCHTST_RESULTS markers.

Primary comparison is PAC-Former Table 4 = iTransformer Table 10 (L=96), PatchTST column.
Relative difference = (ours_mean - paper) / paper * 100 (negative = lower error than paper).
"""
import argparse, csv, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ORDER = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Exchange', 'Weather', 'ECL', 'Traffic']
BEGIN, END = '<!-- PATCHTST_RESULTS:BEGIN -->', '<!-- PATCHTST_RESULTS:END -->'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(ROOT, 'results', 'results.csv'))
    ap.add_argument('--ref', default=os.path.join(ROOT, 'configs', 'paper_reference.csv'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'summary.csv'))
    ap.add_argument('--readme', default='', help='root README.md to update in place')
    a = ap.parse_args()
    ref = {(r['dataset'], int(r['pred_len'])): r for r in csv.DictReader(open(a.ref, encoding='utf-8'))}
    per = {}
    for r in csv.DictReader(open(a.results, encoding='utf-8')):
        if r['status'] != 'ok':
            continue
        per.setdefault((r['dataset'], int(r['pred_len'])), []).append(r)
    rows = []
    for key in sorted(per, key=lambda k: (ORDER.index(k[0]), k[1])):
        rs = sorted(per[key], key=lambda r: int(r['seed']))
        mse = [float(r['test_mse']) for r in rs]; mae = [float(r['test_mae']) for r in rs]
        p = ref.get(key, {})
        pm = float(p['paper_mse']) if p.get('paper_mse') else None
        pa = float(p['paper_mae']) if p.get('paper_mae') else None
        rows.append({'dataset': key[0], 'pred_len': key[1], 'n_seeds': len(rs),
                     'seeds': ' '.join(r['seed'] for r in rs),
                     'mse_mean': st.mean(mse), 'mse_std': st.stdev(mse) if len(mse) > 1 else 0.0,
                     'mae_mean': st.mean(mae), 'mae_std': st.stdev(mae) if len(mae) > 1 else 0.0,
                     'paper_mse': pm, 'paper_mae': pa,
                     'mse_rel_diff_pct': (st.mean(mse) - pm) / pm * 100 if pm else None,
                     'mae_rel_diff_pct': (st.mean(mae) - pa) / pa * 100 if pa else None,
                     **{f'mse_s{r["seed"]}': float(r['test_mse']) for r in rs},
                     **{f'mae_s{r["seed"]}': float(r['test_mae']) for r in rs}})
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    f3 = lambda v: '' if v is None else f'{v:.3f}'
    pct = lambda v: '' if v is None else f'{v:+.2f}%'
    md = ['| Dataset | Horizon | Seeds | MSE (mean ± std) | MAE (mean ± std) | Paper MSE | Paper MAE | ΔMSE vs paper | ΔMAE vs paper |',
          '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        md.append(f"| {r['dataset']} | {r['pred_len']} | {r['n_seeds']} | {r['mse_mean']:.4f} ± {r['mse_std']:.4f} | "
                  f"{r['mae_mean']:.4f} ± {r['mae_std']:.4f} | {f3(r['paper_mse'])} | {f3(r['paper_mae'])} | "
                  f"{pct(r['mse_rel_diff_pct'])} | {pct(r['mae_rel_diff_pct'])} |")
    table = '\n'.join(md)
    print(table)
    if a.readme:
        txt = open(a.readme, encoding='utf-8').read()
        i, j = txt.index(BEGIN) + len(BEGIN), txt.index(END)
        open(a.readme, 'w', encoding='utf-8').write(txt[:i] + '\n' + table + '\n' + txt[j:])


if __name__ == '__main__':
    main()
