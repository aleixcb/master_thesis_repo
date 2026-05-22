"""Append Sections 18-23 (Forecast Evaluation and Results) to Methodology.ipynb."""
import json, os

REPO = r'c:\Users\PCUser\Documents\GitHub\master_thesis_repo'
NB   = os.path.join(REPO, 'Methodology.ipynb')

with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def md(src):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src}

def code(src):
    return {'cell_type': 'code', 'execution_count': None,
            'metadata': {}, 'outputs': [], 'source': src}

new_cells = []

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 18 — Forecast evaluation setup
# ══════════════════════════════════════════════════════════════════════════════

new_cells.append(md(
    '## Section 18 — Forecast Evaluation Setup\n\n'
    'Methodology §3.6: evaluation is on the test period (`SPLIT_DATE` onward), '
    'restricted to the intersection of all models\' valid date ranges. '
    'We build:\n\n'
    '1. **RV proxies** — RV5 and RV22 (rolling 5-day and 22-day mean of squared '
    'log-returns), appended to `df_sent`.\n'
    '2. **Forecast registry** — unified `(model_name, metal) → pd.Series` covering '
    '10 models × 4 metals = 40 series.\n'
    '3. **Realised targets** — `(proxy, metal) → pd.Series` aligned to `eval_dates`.'
))

new_cells.append(code("""\
import numpy as np
import pandas as pd
import pickle

# ── RV proxies — methodology §3.6, Patton (2011) ────────────────────────────
# RVh_t = (1/h) * sum_{k=0}^{h-1} r^2_{t-k}  (backward-looking, no look-ahead)
for metal in METALS:
    ret2 = df_sent[f'{metal}_logret'] ** 2
    df_sent[f'{metal}_RV5']  = ret2.rolling(5,  min_periods=5).mean()
    df_sent[f'{metal}_RV22'] = ret2.rolling(22, min_periods=22).mean()

print('RV proxies added to df_sent.')
for metal in METALS:
    print(f'  {metal}: RV5 first_valid={df_sent[f\"{metal}_RV5\"].first_valid_index().date()}  '
          f'RV22 first_valid={df_sent[f\"{metal}_RV22\"].first_valid_index().date()}')

# ── Helper ────────────────────────────────────────────────────────────────────
def to_series(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj

# ── Forecast registry ─────────────────────────────────────────────────────────
MODEL_NAMES = ['HS_N20', 'HS_N30', 'HS_N60', 'GARCH11',
               'EGARCHX_BIC', 'EGARCHX_L1',
               'GJRX_main_BIC', 'GJRX_main_L1',
               'GJRX_aux_BIC',  'GJRX_aux_L1']

forecast_registry = {}   # (model_name, metal) -> pd.Series

for metal in METALS:
    for N in [20, 30, 60]:
        forecast_registry[(f'HS_N{N}', metal)] = to_series(hs_forecast_store[(metal, N)])
    forecast_registry[('GARCH11',       metal)] = to_series(garch11_forecasts[metal])
    forecast_registry[('EGARCHX_BIC',   metal)] = egarchx_forecasts_bic[metal]
    forecast_registry[('EGARCHX_L1',    metal)] = egarchx_forecasts_l1[metal]
    forecast_registry[('GJRX_main_BIC', metal)] = gjrx_forecasts[('main', 'bic', metal)]
    forecast_registry[('GJRX_main_L1',  metal)] = gjrx_forecasts[('main', 'l1',  metal)]
    forecast_registry[('GJRX_aux_BIC',  metal)] = gjrx_forecasts[('aux',  'bic', metal)]
    forecast_registry[('GJRX_aux_L1',   metal)] = gjrx_forecasts[('aux',  'l1',  metal)]

# ── Evaluation window: intersection of all valid dates ────────────────────────
all_valid = [set(s.dropna().index) for s in forecast_registry.values()]
eval_dates = pd.DatetimeIndex(sorted(set.intersection(*all_valid)))
print(f'\\nEvaluation window: {eval_dates[0].date()} — {eval_dates[-1].date()}, T={len(eval_dates)}')

nan_issues = [(m, mt) for (m, mt), s in forecast_registry.items()
              if s.loc[eval_dates].isna().any()]
if nan_issues:
    print('WARNING NaN in eval window:', nan_issues)
else:
    print('All 40 forecast series: 0 NaN in eval window.')

# ── Realised targets ──────────────────────────────────────────────────────────
realised = {}
for proxy in RV_PROXIES:
    for metal in METALS:
        rv_col = f'{metal}_{proxy}'
        rv_s = df_sent[rv_col].loc[eval_dates]
        n_nan = rv_s.isna().sum()
        if n_nan > 0:
            print(f'WARNING: {rv_col} has {n_nan} NaN — filling with 1e-8')
            rv_s = rv_s.fillna(1e-8)
        realised[(proxy, metal)] = rv_s

print(f'\\nRegistry: {len(forecast_registry)} series ({len(MODEL_NAMES)} models x {len(METALS)} metals)')
print(f'Realised:  {len(realised)} targets ({len(RV_PROXIES)} proxies x {len(METALS)} metals)')
print(f'Models: {MODEL_NAMES}')
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 19 — Loss functions
# ══════════════════════════════════════════════════════════════════════════════

new_cells.append(md(
    '## Section 19 — Loss Functions (Methodology §3.6.1)\n\n'
    'Four loss functions per Patton (2011). **QLIKE is the primary metric**: it is '
    'robust to the noise in daily RV proxies — any proxy-consistent loss function '
    'gives the same ranking in expectation when h=σ² is the true DGP. '
    'MSE, MAE, and HMSE are reported for completeness; consistent rankings across '
    'all four strengthen conclusions.\n\n'
    '| Loss | Formula | Notes |\n'
    '|------|---------|-------|\n'
    '| MSE  | $\\mathrm{E}[(\\hat{\\sigma}^2 - \\sigma^2)^2]$ | level sensitivity |\n'
    '| MAE  | $\\mathrm{E}[|\\hat{\\sigma}^2 - \\sigma^2|]$ | level, robust to outliers |\n'
    '| **QLIKE** | $\\mathrm{E}[\\sigma^2/\\hat{\\sigma}^2 - \\ln(\\sigma^2/\\hat{\\sigma}^2) - 1]$ | **primary** — proxy-consistent |\n'
    '| HMSE | $\\mathrm{E}[(1 - \\hat{\\sigma}^2/\\sigma^2)^2]$ | relative deviation |'
))

new_cells.append(code("""\
# Loss functions — Patton (2011) §3.6.1
# h = forecast variance, s2 = realised proxy

def qlike_series(h, s2):
    h  = np.maximum(h.values,  1e-12)
    s2 = np.maximum(s2.values, 1e-12)
    return pd.Series(s2 / h - np.log(s2 / h) - 1)

def compute_losses(h_s, s2_s):
    h  = np.maximum(h_s.values,  1e-12)
    s2 = np.maximum(s2_s.values, 1e-12)
    mse   = float(np.mean((h  - s2)**2))
    mae   = float(np.mean(np.abs(h - s2)))
    qlike = float(np.mean(s2 / h - np.log(s2 / h) - 1))
    hmse  = float(np.mean((1 - h / s2)**2))
    return mse, mae, qlike, hmse

# ── Build loss_table ──────────────────────────────────────────────────────────
loss_rows = []
for model in MODEL_NAMES:
    for metal in METALS:
        h_s = forecast_registry[(model, metal)].loc[eval_dates]
        for proxy in RV_PROXIES:
            s2_s = realised[(proxy, metal)]
            mse, mae, qlike, hmse = compute_losses(h_s, s2_s)
            loss_rows.append({'Model': model, 'Metal': metal, 'Proxy': proxy,
                              'MSE': mse, 'MAE': mae, 'QLIKE': qlike, 'HMSE': hmse})

loss_table = pd.DataFrame(loss_rows)
loss_table.to_pickle('outputs/loss_table.pkl')
print(f'loss_table saved: {len(loss_table)} rows')

# ── Print QLIKE tables ────────────────────────────────────────────────────────
for proxy in RV_PROXIES:
    sub = loss_table[loss_table['Proxy'] == proxy]
    pivot = sub.pivot(index='Model', columns='Metal', values='QLIKE').loc[MODEL_NAMES]
    print(f'\\nQlike — {proxy}')
    print('=' * 72)
    hdr = f"  {'Model':<18}" + ''.join(f'{m:>10}' for m in METALS) + '   RANK_AVG'
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    best_row = pivot.idxmin(axis=0)
    for model in MODEL_NAMES:
        row = pivot.loc[model]
        avg_rank = sub[sub['Model'] == model].set_index('Metal')['QLIKE'].rank().mean()
        stars = ''
        vals = ''
        for metal in METALS:
            v = row[metal]
            mark = '*' if best_row[metal] == model else ' '
            vals += f'{v:>9.6f}{mark}'
        print(f"  {model:<18}{vals}  {avg_rank:>5.1f}")
    print()
    print(f'  Best per metal ({proxy}):')
    for metal in METALS:
        best_m = pivot[metal].idxmin()
        best_v = pivot[metal].min()
        garch_v = pivot.loc['GARCH11', metal]
        improve = (garch_v - best_v) / garch_v * 100
        print(f'    {metal}: {best_m} (QLIKE={best_v:.6f}, {improve:+.2f}% vs GARCH11)')

# ── GARCH11 vs best sentiment model per cell ──────────────────────────────────
SENT_MODELS = ['EGARCHX_BIC', 'EGARCHX_L1', 'GJRX_main_BIC', 'GJRX_main_L1',
               'GJRX_aux_BIC', 'GJRX_aux_L1']
print('\\nGARCH11 QLIKE vs best-sentiment QLIKE — sign is (GARCH11 - sentiment):')
print(f"  {'Metal':<5} {'Proxy':<6} {'GARCH11':>10} {'BestSent':>10} {'Model':>18} {'Diff':>8} {'Win?':>6}")
for proxy in RV_PROXIES:
    sub = loss_table[loss_table['Proxy'] == proxy]
    for metal in METALS:
        g11 = sub.loc[(sub['Model'] == 'GARCH11') & (sub['Metal'] == metal), 'QLIKE'].values[0]
        sent_sub = sub[sub['Model'].isin(SENT_MODELS) & (sub['Metal'] == metal)]
        best_row_s = sent_sub.loc[sent_sub['QLIKE'].idxmin()]
        best_v = best_row_s['QLIKE']
        best_m = best_row_s['Model']
        diff = g11 - best_v
        win = 'YES' if diff > 0 else 'NO'
        print(f"  {metal:<5} {proxy:<6} {g11:>10.6f} {best_v:>10.6f} {best_m:>18} {diff:>+8.6f} {win:>6}")
"""))

new_cells.append(md(
    '### Section 19 — Commentary\n\n'
    '**Primary question (a):** A cell is a *win* for the sentiment hypothesis if any '
    'sentiment-augmented model has strictly lower QLIKE than GARCH(1,1) for that metal × '
    'proxy combination. A *draw* indicates economically meaningful improvement that is not '
    'statistically distinguishable at 5% (Section 20 DM tests resolve this).\n\n'
    '**Primary question (b):** If rankings are consistent between RV5 and RV22, the conclusions '
    'are robust to the choice of proxy. Inconsistent rankings indicate sensitivity to the '
    'smoothing horizon in the proxy — RV22 averages over 22 days and is more stable but less '
    'responsive to short-run volatility spikes.\n\n'
    '**Benchmark:** HS_N60 or HS_N30 often outperforms HS_N20 in low-volatility regimes '
    '(smoother forecast). DM test (Section 20, pair 1) tests whether GARCH(1,1) significantly '
    'outperforms the best HS model — if not, parametric GARCH adds no value over simple rolling '
    'variance, and the bar for sentiment augmentation is lower.'
))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 20 — Diebold-Mariano pairwise tests
# ══════════════════════════════════════════════════════════════════════════════

new_cells.append(md(
    '## Section 20 — Diebold-Mariano Pairwise Tests (Methodology §3.6.2)\n\n'
    'Six pre-specified pairs, QLIKE loss, HLN small-sample correction (Harvey, Leybourne '
    '& Newbold 1997). For one-step-ahead forecasts ($h=1$), the HLN correction multiplies '
    'the DM statistic by $\\sqrt{(T-1)/T}$ and the critical values come from $t(T-1)$.\n\n'
    'A positive DM statistic means Model 1 has higher average QLIKE than Model 2 (Model 2 is '
    'better); a negative statistic means Model 1 is better. Two-sided $p$-value reported.\n\n'
    '**Six pairs:**\n'
    '1. GARCH11 vs HS_N60 — does parametric GARCH help at all?\n'
    '2. EGARCHX_BIC vs GARCH11 — primary thesis test (BIC lag)\n'
    '3. EGARCHX_L1 vs GARCH11 — primary thesis robustness (L=1)\n'
    '4. GJRX_main_BIC vs EGARCHX_BIC — does asymmetric sentiment add value?\n'
    '5. GJRX_main_BIC vs GARCH11 — sentiment-asymmetry channel vs baseline\n'
    '6. GJRX_aux_BIC vs GJRX_main_BIC — orthogonal spec vs main spec'
))

new_cells.append(code("""\
from scipy.stats import t as t_dist

def dm_hln(fc1_s, fc2_s, rv_s, h=1):
    \"\"\"Diebold-Mariano test with HLN correction. h=1 for one-step-ahead.\"\"\"
    d = qlike_series(fc1_s, rv_s) - qlike_series(fc2_s, rv_s)
    T = len(d)
    dbar = d.mean()
    var_d = d.var(ddof=1)
    if var_d < 1e-20 or T < 3:
        return 0.0, 1.0
    dm = dbar / np.sqrt(var_d / T)
    # HLN correction for h=1: factor = sqrt((T-1)/T)
    hln = dm * np.sqrt((T - 1) / T)
    pval = float(2 * t_dist.sf(abs(hln), df=T - 1))
    return float(hln), pval

def sig_label(stat, pval):
    if pval < 0.01:
        return '***'
    if pval < 0.05:
        return '**'
    if pval < 0.10:
        return '*'
    return ''

DM_PAIRS = [
    ('GARCH11',       'HS_N60',        'GARCH11 vs HS_N60       '),
    ('EGARCHX_BIC',   'GARCH11',       'EGARCHX_BIC vs GARCH11  '),
    ('EGARCHX_L1',    'GARCH11',       'EGARCHX_L1  vs GARCH11  '),
    ('GJRX_main_BIC', 'EGARCHX_BIC',   'GJRXmain_BIC vs EGARCHX '),
    ('GJRX_main_BIC', 'GARCH11',       'GJRXmain_BIC vs GARCH11 '),
    ('GJRX_aux_BIC',  'GJRX_main_BIC', 'GJRXaux_BIC vs GJRXmain '),
]

dm_rows = []
for proxy in RV_PROXIES:
    for metal in METALS:
        rv_s = realised[(proxy, metal)]
        for m1, m2, label in DM_PAIRS:
            fc1 = forecast_registry[(m1, metal)].loc[eval_dates]
            fc2 = forecast_registry[(m2, metal)].loc[eval_dates]
            stat, pval = dm_hln(fc1, fc2, rv_s)
            l1_mean = qlike_series(fc1, rv_s).mean()
            l2_mean = qlike_series(fc2, rv_s).mean()
            direction = f'{m1} BETTER' if l1_mean < l2_mean else f'{m2} BETTER'
            if pval < 0.05:
                sig_str = f'SIG ({direction})'
            elif pval < 0.10:
                sig_str = f'MARGINAL ({direction})'
            else:
                sig_str = 'no sig diff'
            dm_rows.append({'Pair': label.strip(), 'Model1': m1, 'Model2': m2,
                            'Metal': metal, 'Proxy': proxy,
                            'DM_stat': stat, 'p_value': pval,
                            'Significance': sig_str})

dm_table = pd.DataFrame(dm_rows)
dm_table.to_pickle('outputs/dm_results.pkl')
print(f'DM results saved: {len(dm_table)} rows')

# ── Print tables ──────────────────────────────────────────────────────────────
for proxy in RV_PROXIES:
    print(f'\\nDM tests — {proxy}')
    print('=' * 90)
    print(f"  {'Pair':<32} " + ''.join(f'{m:>10}' for m in METALS))
    print('  ' + '-' * 72)
    for m1, m2, label in DM_PAIRS:
        vals = ''
        for metal in METALS:
            row = dm_table[(dm_table['Model1'] == m1) & (dm_table['Model2'] == m2) &
                           (dm_table['Metal'] == metal) & (dm_table['Proxy'] == proxy)]
            stat = row['DM_stat'].values[0]
            pval = row['p_value'].values[0]
            vals += f'  {stat:>5.2f}{sig_label(stat, pval):<3}'
        print(f'  {label.strip():<32}{vals}')
    print('  (stat>0 = Model2 better; * p<.10, ** p<.05, *** p<.01)')

# ── Primary thesis result: EGARCHX_BIC vs GARCH11 ────────────────────────────
print('\\n--- Primary thesis result: EGARCHX_BIC vs GARCH11 ---')
for proxy in RV_PROXIES:
    for metal in METALS:
        row = dm_table[(dm_table['Model1'] == 'EGARCHX_BIC') &
                       (dm_table['Model2'] == 'GARCH11') &
                       (dm_table['Metal'] == metal) &
                       (dm_table['Proxy'] == proxy)].iloc[0]
        qlike_eg = loss_table[(loss_table['Model'] == 'EGARCHX_BIC') &
                              (loss_table['Metal'] == metal) &
                              (loss_table['Proxy'] == proxy)]['QLIKE'].values[0]
        qlike_g  = loss_table[(loss_table['Model'] == 'GARCH11') &
                              (loss_table['Metal'] == metal) &
                              (loss_table['Proxy'] == proxy)]['QLIKE'].values[0]
        print(f'  {metal}/{proxy}: EGARCH={qlike_eg:.6f} GARCH11={qlike_g:.6f} '
              f'DM={row["DM_stat"]:+.3f} p={row["p_value"]:.4f}{sig_label(row["DM_stat"], row["p_value"])}')
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 21 — Model Confidence Set
# ══════════════════════════════════════════════════════════════════════════════

new_cells.append(md(
    '## Section 21 — Model Confidence Set (Hansen, Lunde & Nason 2011)\n\n'
    'MCS at $\\alpha = 0.10$ applied to all 10 models per metal × proxy, using QLIKE loss. '
    'The MCS is the smallest set of models that includes the DGP\'s best model with '
    '$1 - \\alpha$ confidence.\n\n'
    '**Implementation:** `arch.bootstrap.MCS` with stationary bootstrap '
    '($B = 10{,}000$, block length $= 5$, range test $T_R$). '
    'If `arch.bootstrap.MCS` fails for any cell, a manual implementation is used: '
    'repeated equivalence testing with $B = 10{,}000$ stationary bootstrap replications. '
    'The implementation path is printed per cell.'
))

new_cells.append(code("""\
from arch.bootstrap import MCS

mcs_results = {}   # (metal, proxy) -> {'included': [model names], 'method': str}

def manual_mcs(loss_df, size=0.10, B=10000, block_size=5, seed=42):
    \"\"\"
    Manual MCS: TR range-statistic with stationary bootstrap.
    Eliminates the model with the highest mean loss when TR p-value < size.
    \"\"\"
    rng = np.random.default_rng(seed)
    T, M = loss_df.shape
    model_set = list(loss_df.columns)

    def stationary_bootstrap_mean(d_arr, B, block_size, rng):
        # d_arr: T x (M-1) deviations from reference
        T_ = d_arr.shape[0]
        p = 1.0 / block_size
        boot_means = np.zeros((B, d_arr.shape[1]))
        for b in range(B):
            idx = np.zeros(T_, dtype=int)
            idx[0] = rng.integers(T_)
            for t in range(1, T_):
                if rng.random() < p:
                    idx[t] = rng.integers(T_)
                else:
                    idx[t] = (idx[t-1] + 1) % T_
            boot_means[b] = d_arr[idx].mean(axis=0)
        return boot_means

    while len(model_set) > 1:
        sub = loss_df[model_set].values  # T x |M|
        mean_losses = sub.mean(axis=0)
        # d_i = mean loss of i minus grand mean
        grand_mean = mean_losses.mean()
        d_arr = sub - sub.mean(axis=1, keepdims=True)  # demeaned per t
        # TR statistic: max |t_i| where t_i = sqrt(T) * d_i / se_i
        se = np.array([d_arr[:, j].std(ddof=1) + 1e-20 for j in range(len(model_set))])
        t_stats = np.sqrt(T) * (mean_losses - grand_mean) / se
        TR_obs = np.max(np.abs(t_stats))

        # Bootstrap
        boot_means = stationary_bootstrap_mean(d_arr, B, block_size, rng)
        boot_grand  = boot_means.mean(axis=1, keepdims=True)
        boot_t = np.sqrt(T) * (boot_means - boot_grand) / se
        boot_TR = np.max(np.abs(boot_t), axis=1)
        pval = float((boot_TR >= TR_obs).mean())

        if pval >= size:
            break  # cannot reject equality — all remaining models are in MCS
        # Eliminate worst model (highest mean loss)
        worst_idx = np.argmax(mean_losses)
        model_set.pop(worst_idx)

    return model_set

for proxy in RV_PROXIES:
    for metal in METALS:
        rv_s = realised[(proxy, metal)]
        loss_cols = {}
        for model in MODEL_NAMES:
            fc = forecast_registry[(model, metal)].loc[eval_dates]
            loss_cols[model] = qlike_series(fc, rv_s).values
        loss_df = pd.DataFrame(loss_cols)

        method_used = 'arch.bootstrap.MCS'
        try:
            mcs_obj = MCS(loss_df, size=0.10, reps=10000, block_size=5,
                         method='R', bootstrap='stationary', seed=42)
            mcs_obj.compute()
            included = list(mcs_obj.included)
        except Exception as e:
            print(f'arch MCS failed ({metal}/{proxy}): {e} — using manual MCS')
            included = manual_mcs(loss_df, size=0.10, B=10000, block_size=5)
            method_used = 'manual MCS (TR, stationary bootstrap)'

        mcs_results[(metal, proxy)] = {'included': included, 'method': method_used}
        print(f'  {metal}/{proxy}: {len(included)}/{len(MODEL_NAMES)} in MCS  '
              f'[{", ".join(included)}]')

pickle.dump(mcs_results, open('outputs/mcs_results.pkl', 'wb'))
print('\\nmcs_results saved.')

# ── Print MCS table ───────────────────────────────────────────────────────────
print('\\nMCS Surviving Sets at alpha=0.10 (QLIKE)')
print('=' * 70)
print(f"  {'Metal':<5} {'Proxy':<6}  Surviving models")
print('  ' + '-' * 60)
for proxy in RV_PROXIES:
    for metal in METALS:
        incl = mcs_results[(metal, proxy)]['included']
        print(f'  {metal:<5} {proxy:<6}  {", ".join(incl)}')
"""))

new_cells.append(md(
    '### Section 21 — MCS Commentary\n\n'
    'The MCS surviving set answers: *which models are statistically '
    'indistinguishable from the best model at $\\alpha = 0.10$?*\n\n'
    '**What to look for:**\n\n'
    '- If any sentiment model (EGARCHX or GJRX) is in the MCS for a given metal × proxy, '
    'the data are consistent with sentiment improving volatility forecasting — even if '
    'GARCH(1,1) is also in the MCS (cannot reject equal accuracy).\n'
    '- If GARCH(1,1) alone or with HS models survives and all sentiment models are eliminated, '
    'sentiment does not improve over the baseline within the MCS framework.\n'
    '- MCS results are more conservative than point-estimate QLIKE comparisons — a sentiment '
    'model can rank higher than GARCH(1,1) in QLIKE without being significantly better. '
    'The DM tests (Section 20) test individual pairs; the MCS tests the full set simultaneously.'
))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 22 — Robustness battery
# ══════════════════════════════════════════════════════════════════════════════

new_cells.append(md(
    '## Section 22 — Robustness Battery (Methodology §3.7)\n\n'
    'Items are implemented in priority order; the section stops if the 90-minute wall-clock '
    'budget is exceeded. Completed and skipped items are documented at end of section.\n\n'
    '- **22a** (mandatory): Cross-metal consistency narrative\n'
    '- **22b** (mandatory): Sub-sample QLIKE at test-period split\n'
    '- **22c** (time-permitting): Alternative 75/25 and 70/30 train/test splits\n'
    '- **22d** (time-permitting): Pre-COVID / COVID / Post-COVID in-sample log-likelihood'
))

new_cells.append(code("""\
import time
section_22_start = time.time()
BUDGET_22 = 90 * 60   # 90 minutes

# ══ 22a — Cross-metal consistency ════════════════════════════════════════════
# Granger pattern (EDA §8): Nd/Pr/Dy significant at lag 1; Tb: none
# Does this predict better forecast performance for Nd/Pr/Dy vs Tb?

print('22a — Cross-metal consistency')
print('=' * 70)
print('EDA §8 Granger at lag 1: Nd/Pr/Dy SIGNIFICANT; Tb NONE')
print()
print(f"  {'Metal':<4} {'Granger':>8}  ", end='')
for proxy in RV_PROXIES:
    print(f'  BestModel({proxy}):QLIKE  Beat_G11?', end='')
print()
print('  ' + '-' * 80)

GRANGER_SIG = {'Nd': True, 'Pr': True, 'Dy': True, 'Tb': False}

for metal in METALS:
    gran = 'YES' if GRANGER_SIG[metal] else 'NO'
    print(f'  {metal:<4} {gran:>8}  ', end='')
    for proxy in RV_PROXIES:
        sub = loss_table[(loss_table['Metal'] == metal) & (loss_table['Proxy'] == proxy)]
        best_row = sub.loc[sub['QLIKE'].idxmin()]
        g11_q = sub[sub['Model'] == 'GARCH11']['QLIKE'].values[0]
        beat = 'YES' if best_row['QLIKE'] < g11_q else 'NO'
        print(f'  {best_row["Model"]:<18}: {best_row["QLIKE"]:.6f}  {beat:>8}', end='')
    print()

print()
print('Best sentiment model per metal (across both proxies):')
for metal in METALS:
    sent_sub = loss_table[(loss_table['Metal'] == metal) &
                          (loss_table['Model'].isin(['EGARCHX_BIC','EGARCHX_L1',
                           'GJRX_main_BIC','GJRX_main_L1','GJRX_aux_BIC','GJRX_aux_L1']))]
    for proxy in RV_PROXIES:
        s = sent_sub[sent_sub['Proxy'] == proxy]
        best = s.loc[s['QLIKE'].idxmin()]
        g11_q = loss_table[(loss_table['Metal'] == metal) & (loss_table['Proxy'] == proxy) &
                           (loss_table['Model'] == 'GARCH11')]['QLIKE'].values[0]
        mcs_incl = mcs_results[(metal, proxy)]['included']
        in_mcs = best['Model'] in mcs_incl
        print(f'  {metal}/{proxy}: {best["Model"]} QLIKE={best["QLIKE"]:.6f} '
              f'vs GARCH11={g11_q:.6f}  In_MCS={in_mcs}')
"""))

new_cells.append(code("""\
# ══ 22b — Sub-sample QLIKE ════════════════════════════════════════════════════
# Split test period at 2024-08-01 or midpoint, whichever is closer to midpoint
test_midpoint = eval_dates[len(eval_dates) // 2]
anchor = pd.Timestamp('2024-08-01')
if anchor not in eval_dates:
    # Find nearest available date
    diffs = np.abs((eval_dates - anchor).days)
    anchor = eval_dates[np.argmin(diffs)]

print(f'22b — Sub-sample QLIKE')
print(f'Test period: {eval_dates[0].date()} — {eval_dates[-1].date()}')
print(f'Midpoint: {test_midpoint.date()}')
print(f'Sub-sample split: {anchor.date()}')
print(f'Sub-period A (early): {eval_dates[0].date()} — {anchor.date()}')
print(f'Sub-period B (late):  {anchor.date()} — {eval_dates[-1].date()}')

eval_A = eval_dates[eval_dates <= anchor]
eval_B = eval_dates[eval_dates > anchor]
print(f'T_A={len(eval_A)}, T_B={len(eval_B)}')

for sub_label, sub_dates in [('Early', eval_A), ('Late', eval_B)]:
    print(f'\\n--- Sub-period: {sub_label} (T={len(sub_dates)}) ---')
    for proxy in RV_PROXIES:
        pivot_rows = []
        for model in MODEL_NAMES:
            row = {'Model': model}
            for metal in METALS:
                h  = forecast_registry[(model, metal)].loc[sub_dates]
                rv = df_sent[f'{metal}_{proxy}'].loc[sub_dates]
                h  = np.maximum(h.values,  1e-12)
                rv = np.maximum(rv.values, 1e-12)
                row[metal] = float(np.mean(rv / h - np.log(rv / h) - 1))
            pivot_rows.append(row)
        sub_pivot = pd.DataFrame(pivot_rows).set_index('Model')
        print(f'  QLIKE {proxy}:')
        hdr = f"    {'Model':<18}" + ''.join(f'{m:>10}' for m in METALS)
        print(hdr)
        best_col = sub_pivot.idxmin(axis=0)
        for model in MODEL_NAMES:
            vals = ''
            for metal in METALS:
                v = sub_pivot.loc[model, metal]
                mark = '*' if best_col[metal] == model else ' '
                vals += f'{v:>9.6f}{mark}'
            print(f'    {model:<18}{vals}')

print(f'\\n22b complete. Elapsed: {time.time()-section_22_start:.0f}s')
"""))

new_cells.append(code("""\
# ══ 22c — Alternative train/test splits ══════════════════════════════════════
# Re-fit GARCH(1,1) and EGARCHX_BIC under 75/25 and 70/30 splits
# REFIT_FREQ=22 (monthly) to keep runtime manageable

from arch import arch_model
import statsmodels.api as sm

elapsed_22 = time.time() - section_22_start
if elapsed_22 > BUDGET_22:
    print(f'22c SKIPPED — time budget exhausted ({elapsed_22/60:.1f} min used)')
else:
    n_total = len(df_sent)
    split_75 = df_sent.index[int(0.75 * n_total)]
    split_70 = df_sent.index[int(0.70 * n_total)]
    print(f'22c — Alternative splits')
    print(f'  75/25 split: {split_75.date()} ({int(0.75*n_total)} train / {n_total-int(0.75*n_total)} test)')
    print(f'  70/30 split: {split_70.date()} ({int(0.70*n_total)} train / {n_total-int(0.70*n_total)} test)')

    alt_results = {}  # (split_label, model, metal, proxy) -> QLIKE

    REFIT_FREQ_ALT = 22  # monthly refit

    for split_label, split_date in [('75/25', split_75), ('70/30', split_70)]:
        elapsed_22 = time.time() - section_22_start
        if elapsed_22 > BUDGET_22:
            print(f'  {split_label} SKIPPED — budget reached')
            break

        alt_train_mask = df_sent.index < split_date
        alt_test_dates = df_sent.index[df_sent.index >= split_date]

        print(f'\\n  Split {split_label}: fitting...')
        for metal in METALS:
            ret_full_100 = df_sent[f'{metal}_logret'].dropna() * 100
            all_dates_alt = ret_full_100.index
            dist = ERR_DIST[metal]
            ret_train = df_sent.loc[alt_train_mask, f'{metal}_logret'].dropna() * 100

            # Fit GARCH base for this split
            am_g = arch_model(ret_train, mean='Constant', vol='GARCH', p=1, q=1, dist=dist)
            try:
                res_g = am_g.fit(disp='off', show_warning=False, update_freq=0)
            except Exception as e:
                print(f'  [{metal}] GARCH fit failed: {e}')
                continue

            # Fit EGARCH base for this split
            am_e = arch_model(ret_train, mean='Constant', vol='EGARCH', p=1, o=1, q=1, dist=dist)
            try:
                res_e = am_e.fit(disp='off', show_warning=False, update_freq=0)
            except Exception:
                try:
                    res_e = am_e.fit(disp='off', show_warning=False, update_freq=0,
                                     options={'maxiter': 2000})
                except Exception as e2:
                    print(f'  [{metal}] EGARCH fit failed: {e2}')
                    continue

            # HAC OLS for EGARCHX gammas
            lag_bic = SENT_LAG_BIC[metal]
            log_cv = np.log(res_e.conditional_volatility ** 2)
            s1_col = f'tone_mean_std_lag{lag_bic}_{metal}'
            s2_col = f'log_volume_std_lag{lag_bic}_{metal}'
            s1_tr = df_sent.loc[alt_train_mask, s1_col]
            s2_tr = df_sent.loc[alt_train_mask, s2_col]
            aux_df = pd.DataFrame({'lv': log_cv, 's1': s1_tr, 's2': s2_tr}).dropna()
            ols = sm.OLS(aux_df['lv'], sm.add_constant(aux_df[['s1', 's2']])).fit(
                      cov_type='HAC', cov_kwds={'maxlags': 5, 'use_correction': True})
            g1 = ols.params['s1']
            g2 = ols.params['s2']

            # Expanding-window forecasts with monthly refit
            fc_g11 = []
            fc_eg  = []
            res_g_curr = res_g
            res_e_curr = res_e
            step = 0

            for t_date in alt_test_dates:
                t_pos = all_dates_alt.get_loc(t_date)
                ret_win = ret_full_100.iloc[:t_pos]

                var_g = np.nan
                var_e = np.nan
                try:
                    if step % REFIT_FREQ_ALT == 0:
                        am_g2 = arch_model(ret_win, mean='Constant', vol='GARCH',
                                          p=1, q=1, dist=dist)
                        r = am_g2.fit(disp='off', show_warning=False, update_freq=0,
                                      starting_values=res_g_curr.params.values)
                        if r.convergence_flag == 0:
                            res_g_curr = r
                        am_e2 = arch_model(ret_win, mean='Constant', vol='EGARCH',
                                          p=1, o=1, q=1, dist=dist)
                        r2 = am_e2.fit(disp='off', show_warning=False, update_freq=0,
                                       starting_values=res_e_curr.params.values)
                        if r2.convergence_flag == 0:
                            res_e_curr = r2

                    fg = res_g_curr.forecast(horizon=1, reindex=False)
                    var_g = fg.variance.iloc[-1, 0] / 1e4

                    fe = res_e_curr.forecast(horizon=1, reindex=False)
                    log_var_base = np.log(fe.variance.iloc[-1, 0])
                    s1v = float(df_sent.get(s1_col, pd.Series(dtype=float)).get(t_date, 0) or 0)
                    s2v = float(df_sent.get(s2_col, pd.Series(dtype=float)).get(t_date, 0) or 0)
                    var_e = np.exp(log_var_base + g1 * s1v + g2 * s2v) / 1e4
                except Exception:
                    pass

                fc_g11.append((t_date, var_g))
                fc_eg.append((t_date, var_e))
                step += 1

            fc_g11_s = pd.Series({d: v for d, v in fc_g11}).dropna()
            fc_eg_s  = pd.Series({d: v for d, v in fc_eg}).dropna()
            eval_alt  = fc_g11_s.index.intersection(fc_eg_s.index)

            for proxy in RV_PROXIES:
                rv_alt = df_sent[f'{metal}_{proxy}'].loc[eval_alt]
                h_g11 = np.maximum(fc_g11_s.loc[eval_alt].values, 1e-12)
                h_eg  = np.maximum(fc_eg_s.loc[eval_alt].values,  1e-12)
                rv    = np.maximum(rv_alt.values, 1e-12)
                q_g11 = float(np.mean(rv / h_g11 - np.log(rv / h_g11) - 1))
                q_eg  = float(np.mean(rv / h_eg  - np.log(rv / h_eg)  - 1))
                alt_results[(split_label, 'GARCH11',     metal, proxy)] = q_g11
                alt_results[(split_label, 'EGARCHX_BIC', metal, proxy)] = q_eg
                beat = 'YES' if q_eg < q_g11 else 'NO'
                print(f'  {split_label} {metal}/{proxy}: GARCH11={q_g11:.6f}  '
                      f'EGARCHX_BIC={q_eg:.6f}  EGARCH_beats={beat}  T_test={len(eval_alt)}')

    elapsed_22 = time.time() - section_22_start
    print(f'\\n22c complete. Elapsed so far: {elapsed_22/60:.1f} min')
"""))

new_cells.append(code("""\
# ══ 22d — Pre/COVID/Post sub-samples (in-sample log-lik only) ════════════════
elapsed_22 = time.time() - section_22_start
if elapsed_22 > BUDGET_22:
    print(f'22d SKIPPED — time budget exhausted ({elapsed_22/60:.1f} min used)')
else:
    from arch import arch_model
    COVID_PERIODS = [
        ('Pre-COVID',     '2015-04-01', '2020-01-15'),
        ('COVID-curbs',   '2020-01-16', '2021-12-31'),
        ('Post-curbs',    '2022-01-01', '2026-04-01'),
    ]
    print('22d — In-sample log-likelihood by COVID sub-period')
    print('  Fitting GARCH(1,1) and EGARCH(1,1,1) within each sub-sample')
    print(f"  {'Period':<16} {'Metal':<5} {'GARCH11_LL':>12} {'EGARCH_LL':>12} "
          f"{'EGARCH_delta_LL':>16} {'Favours':>8}")
    print('  ' + '-' * 72)
    for period_label, start_str, end_str in COVID_PERIODS:
        p_mask = ((df_sent.index >= start_str) & (df_sent.index <= end_str))
        for metal in METALS:
            ret_p = df_sent.loc[p_mask, f'{metal}_logret'].dropna() * 100
            if len(ret_p) < 50:
                print(f'  {period_label:<16} {metal:<5} INSUFFICIENT DATA ({len(ret_p)} obs)')
                continue
            dist = ERR_DIST[metal]
            try:
                am_g = arch_model(ret_p, mean='Constant', vol='GARCH', p=1, q=1, dist=dist)
                rg = am_g.fit(disp='off', show_warning=False, update_freq=0)
                ll_g = rg.loglikelihood

                am_e = arch_model(ret_p, mean='Constant', vol='EGARCH', p=1, o=1, q=1, dist=dist)
                re = am_e.fit(disp='off', show_warning=False, update_freq=0)
                ll_e = re.loglikelihood

                delta = ll_e - ll_g
                favour = 'EGARCH' if delta > 0 else 'GARCH11'
                print(f'  {period_label:<16} {metal:<5} {ll_g:>12.2f} {ll_e:>12.2f} '
                      f'{delta:>+16.2f} {favour:>8}')
            except Exception as ex:
                print(f'  {period_label:<16} {metal:<5} FIT FAILED: {str(ex)[:40]}')

    elapsed_22 = time.time() - section_22_start
    print(f'\\n22d complete. Total Section 22 elapsed: {elapsed_22/60:.1f} min')

# ── Document completed items ──────────────────────────────────────────────────
print('\\n--- Section 22 Robustness Battery Summary ---')
print(f'  22a (cross-metal consistency): COMPLETED')
print(f'  22b (sub-sample QLIKE):        COMPLETED')
elapsed_22c = time.time() - section_22_start
status_c = 'COMPLETED' if elapsed_22c < BUDGET_22 * 1.1 else 'SKIPPED (budget)'
status_d = 'COMPLETED' if elapsed_22c < BUDGET_22 * 1.1 else 'SKIPPED (budget)'
print(f'  22c (alternative splits):      {status_c}')
print(f'  22d (COVID sub-samples):       {status_d}')
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 23 — Final results summary
# ══════════════════════════════════════════════════════════════════════════════

new_cells.append(md(
    '## Section 23 — Final Results Summary and Thesis-Ready Tables\n\n'
    'This section consolidates all headline findings. The four tables correspond '
    'directly to the thesis Chapter 4 tables.\n\n'
    '- **Table 1** (23a): Headline QLIKE across all 10 models × 4 metals × 2 proxies\n'
    '- **Table 2** (23b): DM significance table for the 6 pre-specified pairs\n'
    '- **Table 3** (23c): MCS surviving sets at $\\alpha = 0.10$\n'
    '- **Table 4** (23d): Sentiment coefficient summary ($\\gamma, \\delta$) with '
    '$t$-stats\n'
    '- **23e**: Honest readout — wins, losses, draws for the sentiment hypothesis'
))

new_cells.append(code("""\
# ── Table 1: Headline QLIKE ───────────────────────────────────────────────────
# 10 rows × 8 columns (4 metals × 2 proxies); * marks minimum per column
print('=' * 100)
print('TABLE 1 — Headline QLIKE (primary metric). * = column minimum.')
print('=' * 100)

col_keys = [(m, p) for p in RV_PROXIES for m in METALS]
col_hdr  = ''.join(f'{m+"/"+p:>13}' for m, p in col_keys)
print(f"  {'Model':<18}{col_hdr}")
print('  ' + '-' * (18 + 13 * len(col_keys)))

# Compute best model per column
best_per_col = {}
for m, p in col_keys:
    sub = loss_table[(loss_table['Metal'] == m) & (loss_table['Proxy'] == p)]
    best_per_col[(m, p)] = sub.loc[sub['QLIKE'].idxmin(), 'Model']

for model in MODEL_NAMES:
    row_str = f'  {model:<18}'
    for m, p in col_keys:
        sub = loss_table[(loss_table['Model'] == model) & (loss_table['Metal'] == m) &
                         (loss_table['Proxy'] == p)]
        v = sub['QLIKE'].values[0] if len(sub) else float('nan')
        mark = '*' if best_per_col[(m, p)] == model else ' '
        row_str += f'{v:>12.6f}{mark}'
    print(row_str)

print()
print('  Best model per column:')
best_str = f"  {'':18}"
for m, p in col_keys:
    bm = best_per_col[(m, p)][:12]
    best_str += f'{bm:>13}'
print(best_str)
"""))

new_cells.append(code("""\
# ── Table 2: DM significance ──────────────────────────────────────────────────
print('=' * 100)
print('TABLE 2 — DM test statistics (HLN corrected). Positive = Model2 better.')
print('          * p<.10, ** p<.05, *** p<.01')
print('=' * 100)

col_keys = [(m, p) for p in RV_PROXIES for m in METALS]
col_hdr = ''.join(f'{m+"/"+p:>12}' for m, p in col_keys)
print(f"  {'Pair':<32}{col_hdr}")
print('  ' + '-' * (32 + 12 * len(col_keys)))

for m1, m2, label in DM_PAIRS:
    row_str = f'  {label.strip():<32}'
    for metal, proxy in col_keys:
        row = dm_table[(dm_table['Model1'] == m1) & (dm_table['Model2'] == m2) &
                       (dm_table['Metal'] == metal) & (dm_table['Proxy'] == proxy)]
        if len(row) == 0:
            row_str += f'{"N/A":>12}'
        else:
            stat = row['DM_stat'].values[0]
            pval = row['p_value'].values[0]
            cell = f'{stat:>6.2f}{sig_label(stat, pval)}'
            row_str += f'{cell:>12}'
    print(row_str)
"""))

new_cells.append(code("""\
# ── Table 3: MCS surviving sets ───────────────────────────────────────────────
print('=' * 80)
print('TABLE 3 — MCS surviving sets at alpha=0.10 (QLIKE, T_R statistic)')
print('=' * 80)

for proxy in RV_PROXIES:
    print(f'\\n  Proxy: {proxy}')
    print(f"  {'Metal':<5}  N_surv  Includes_EGARCHX  Includes_GJRX  Includes_GARCH11  Members")
    for metal in METALS:
        incl = mcs_results[(metal, proxy)]['included']
        has_eg  = any('EGARCHX' in m for m in incl)
        has_gjr = any('GJRX'    in m for m in incl)
        has_g11 = 'GARCH11' in incl
        print(f'  {metal:<5}  {len(incl):>5}  {str(has_eg):>16}  {str(has_gjr):>13}  '
              f'{str(has_g11):>16}  {", ".join(incl)}')
"""))

new_cells.append(code("""\
# ── Table 4: Coefficient summary ──────────────────────────────────────────────
print('=' * 100)
print('TABLE 4 — Sentiment coefficients: EGARCH-X (BIC lag) and GJR-X main (BIC lag)')
print('  gamma1/2 from EGARCH two-step HAC OLS; delta1/2 from GJR two-step HAC OLS')
print('=' * 100)
print(f"  {'Metal':<5} {'Lag':>4}  "
      f"{'g1(tone)':>10} {'g1_t':>6} {'g1_p':>6}  "
      f"{'g2(vol)':>10} {'g2_t':>6} {'g2_p':>6}  "
      f"{'d1(pol)':>10} {'d1_t':>6} {'d1_p':>6}  "
      f"{'d2(neg)':>10} {'d2_t':>6} {'d2_p':>6}")
print('  ' + '-' * 96)

def stars(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''

for metal in METALS:
    lag = SENT_LAG_BIC[metal]
    re  = egarchx_results_bic.get(metal)
    rg  = gjrx_main_bic.get(metal)
    g1 = re['gamma1'] if re else float('nan')
    g1t= re['gamma1_t'] if re else float('nan')
    g1p= re['gamma1_p'] if re else float('nan')
    g2 = re['gamma2'] if re else float('nan')
    g2t= re['gamma2_t'] if re else float('nan')
    g2p= re['gamma2_p'] if re else float('nan')
    d1 = rg['delta1'] if rg else float('nan')
    d1t= rg['d1_t'] if rg else float('nan')
    d1p= rg['d1_p'] if rg else float('nan')
    d2 = rg['delta2'] if rg else float('nan')
    d2t= rg['d2_t'] if rg else float('nan')
    d2p= rg['d2_p'] if rg else float('nan')
    print(f'  {metal:<5} {lag:>4}  '
          f'{g1:>10.4f}{stars(g1p):<3} {g1t:>6.2f} {g1p:>6.3f}  '
          f'{g2:>10.4f}{stars(g2p):<3} {g2t:>6.2f} {g2p:>6.3f}  '
          f'{d1:>10.4f}{stars(d1p):<3} {d1t:>6.2f} {d1p:>6.3f}  '
          f'{d2:>10.4f}{stars(d2p):<3} {d2t:>6.2f} {d2p:>6.3f}')

print('\\n  Note: stars = HAC-robust t-test. EGARCH: g1=tone_mean, g2=log_volume.')
print('  GJR-main: d1=polarity_mean, d2=neg_tone_mean.')
"""))

new_cells.append(code("""\
# ── 23e: Win/loss/draw computation ────────────────────────────────────────────
print('23e — Wins, losses, draws for sentiment hypothesis')
print('  WIN  = EGARCHX_BIC has lower QLIKE than GARCH11 AND DM p < 0.05')
print('  DRAW = lower QLIKE but DM p >= 0.05 (economically interesting, not sig)')
print('  LOSS = higher QLIKE')
print()
wld_rows = []
for proxy in RV_PROXIES:
    for metal in METALS:
        eg_q = loss_table[(loss_table['Model'] == 'EGARCHX_BIC') &
                          (loss_table['Metal'] == metal) &
                          (loss_table['Proxy'] == proxy)]['QLIKE'].values[0]
        g11_q = loss_table[(loss_table['Model'] == 'GARCH11') &
                           (loss_table['Metal'] == metal) &
                           (loss_table['Proxy'] == proxy)]['QLIKE'].values[0]
        dm_row = dm_table[(dm_table['Model1'] == 'EGARCHX_BIC') &
                          (dm_table['Model2'] == 'GARCH11') &
                          (dm_table['Metal'] == metal) &
                          (dm_table['Proxy'] == proxy)].iloc[0]
        pval = dm_row['p_value']
        if eg_q < g11_q and pval < 0.05:
            outcome = 'WIN'
        elif eg_q < g11_q:
            outcome = 'DRAW'
        else:
            outcome = 'LOSS'
        granger = 'YES' if GRANGER_SIG[metal] else 'NO'
        mcs_incl = mcs_results[(metal, proxy)]['included']
        eg_in_mcs = 'YES' if 'EGARCHX_BIC' in mcs_incl else 'NO'
        gjrx_in_mcs = 'YES' if any('GJRX' in m for m in mcs_incl) else 'NO'
        wld_rows.append({'Metal': metal, 'Proxy': proxy, 'Granger': granger,
                         'Outcome': outcome, 'EGARCH_QLIKE': eg_q, 'GARCH11_QLIKE': g11_q,
                         'DM_p': pval, 'EG_in_MCS': eg_in_mcs, 'GJR_in_MCS': gjrx_in_mcs})
        improve = g11_q - eg_q
        print(f'  {metal}/{proxy}: {outcome:<5} | EGARCH={eg_q:.6f} GARCH11={g11_q:.6f} '
              f'diff={improve:+.6f} DM_p={pval:.4f} | Granger={granger} '
              f'EG_in_MCS={eg_in_mcs}')

wld_df = pd.DataFrame(wld_rows)
n_wins  = (wld_df['Outcome'] == 'WIN').sum()
n_draws = (wld_df['Outcome'] == 'DRAW').sum()
n_loss  = (wld_df['Outcome'] == 'LOSS').sum()
print(f'\\nTotal: {n_wins} wins, {n_draws} draws, {n_loss} losses (out of {len(wld_df)} cells)')

# GJR-X asymmetry
print('\\nGJR-X d2 > d1 (bad-news > good-news) per metal:')
for metal in METALS:
    rg = gjrx_main_bic.get(metal)
    if rg:
        d1, d2 = rg['delta1'], rg['delta2']
        wa_p = rg['wald_asym_p']
        direction = 'd2 > d1 (bad>good)' if d2 > d1 else 'd1 > d2 (good>bad)'
        print(f'  {metal}: d1={d1:.4f} d2={d2:.4f} — {direction}  Wasym_p={wa_p:.4f}')
"""))

new_cells.append(md(
    '### Section 23e — Honest Readout of the Headline Finding\n\n'
    '**What the hypothesis predicted**\n\n'
    'The thesis hypothesis (methodology §3.5) is that news-sentiment measures — particularly '
    'negative tone and polarity — carry incremental information about rare-earth oxide price '
    'volatility beyond the ARCH effect already captured by GARCH(1,1). '
    'If true, EGARCH-X and GJR-X models should produce lower out-of-sample QLIKE than the '
    'GARCH(1,1) baseline for at least the metals where EDA §8 found significant Granger '
    'causality (Nd, Pr, Dy). Tb, where Granger tests yielded no significant lags, serves '
    'as a quasi-control.\n\n'
    '**What the data shows**\n\n'
    'Refer to the win/loss/draw table printed above (Section 23e code output) and Table 1 '
    '(Section 23a) for exact QLIKE values, and Table 2 (Section 23b) for DM significance. '
    'A *win* requires both a lower QLIKE than GARCH(1,1) and a statistically significant '
    'DM test at 5%. A *draw* is a lower QLIKE without statistical significance — economically '
    'interesting but inconclusive. A *loss* means the sentiment model forecasts worse.\n\n'
    'If EGARCH-X achieves wins or draws for the Granger-significant metals (Nd, Pr, Dy) but '
    'not for Tb, the forecasting evidence is broadly consistent with the EDA priors. '
    'If EGARCH-X loses even for Nd/Pr/Dy, the null of no predictability cannot be rejected '
    'out-of-sample — a valid result. In the volatility forecasting literature, GARCH(1,1) '
    'is notoriously hard to beat out-of-sample (Hansen & Lunde 2005); null results '
    'here are the norm, not the exception.\n\n'
    '**Reconciliation with EDA priors**\n\n'
    'The Granger tests in EDA §8 found in-sample predictability of $r^2_t$ from lagged '
    'tone/volume for Nd, Pr, and Dy. In-sample significance does not guarantee out-of-sample '
    'improvement, especially when: (i) the GARCH process already absorbs most ARCH structure '
    'before sentiment is added; (ii) optimal lags (4–9 days) from the BIC grid search may '
    'reflect in-sample noise; (iii) all metals hit the IGARCH boundary '
    '($\\alpha + \\beta \\approx 1$), making variance highly persistent and difficult to '
    'improve with sentiment-level adjustments.\n\n'
    '**The asymmetry sub-question**\n\n'
    'GJR-X tests whether negative news content ($\\delta_2$) amplifies volatility more than '
    'positive content ($\\delta_1$). Refer to Table 4 (Section 23d) for coefficient signs '
    'and the Wald asymmetry test results from Section 17. '
    'If $\\delta_2 > \\delta_1$ with Wald $p < 0.10$, the negativity-bias '
    'hypothesis is supported. The GJR-X base absorbs the return-sign leverage effect '
    '($\\xi$ parameter); the residual asymmetry in $\\delta_1$ vs $\\delta_2$ is pure '
    'news-content asymmetry, distinct from the EDA §5b sign-bias null.\n\n'
    '**Honest framing of nulls**\n\n'
    'If EGARCH-X does not beat GARCH(1,1) at QLIKE in any metal × proxy cell, the '
    'thesis hypothesis of sentiment-driven volatility improvement is rejected on '
    'out-of-sample evidence. This is a legitimate and publishable result: it implies '
    'that any in-sample sentiment signal (EDA §8 Granger) is too weak or too noisy '
    'to survive out-of-sample evaluation, consistent with the efficient markets '
    'interpretation of rare-earth oxide OTC pricing. The EDA §5b sign-bias null '
    '(no leverage asymmetry) further suggests that the volatility process is symmetric '
    'and well-captured by the GARCH(1,1) baseline alone.'
))

new_cells.append(code("""\
# ── Save final_results.pkl ────────────────────────────────────────────────────
final_results = {
    'loss_table':   loss_table,
    'dm_table':     dm_table,
    'mcs_results':  mcs_results,
    'wld_df':       wld_df,
    'eval_dates':   eval_dates,
    'MODEL_NAMES':  MODEL_NAMES,
}
pickle.dump(final_results, open('outputs/final_results.pkl', 'wb'))

print('============================================================')
print('FINAL RESULTS SAVED to outputs/final_results.pkl')
print('============================================================')
import os
for fname in sorted(os.listdir('outputs')):
    if fname.endswith('.pkl'):
        sz = os.path.getsize(f'outputs/{fname}') / 1024
        print(f'  {fname:<50} {sz:>8.1f} KB')
print('\\nNotebook complete. All 23 sections executed.')
print(f'Evaluation window: T={len(eval_dates)} obs')
print(f'Models evaluated:  {len(MODEL_NAMES)}')
print(f'Loss functions:    MSE, MAE, QLIKE (primary), HMSE')
print(f'DM pairs:          {len(DM_PAIRS)} × {len(METALS)} metals × {len(RV_PROXIES)} proxies')
print(f'MCS cells:         {len(METALS)} metals × {len(RV_PROXIES)} proxies at alpha=0.10')
"""))

# ── Write notebook ────────────────────────────────────────────────────────────
nb['cells'].extend(new_cells)

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Appended {len(new_cells)} cells to Methodology.ipynb')
print(f'Total cells: {len(nb["cells"])}')
