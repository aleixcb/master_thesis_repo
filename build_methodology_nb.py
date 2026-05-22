"""Build Methodology.ipynb — Sections 1-9."""
import json, os

REPO = r'c:\Users\PCUser\Documents\GitHub\master_thesis_repo'
OUT  = os.path.join(REPO, 'Methodology.ipynb')

cells = []

def md(src):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src}

def code(src):
    return {'cell_type': 'code', 'execution_count': None,
            'metadata': {}, 'outputs': [], 'source': src}


# ── TITLE ─────────────────────────────────────────────────────────────────────
cells.append(md(
    '# Methodology Notebook — News Sentiment and Rare-Earth Oxide Price Volatility\n\n'
    'Sections 1–9: data preparation, Historical Simulation benchmark, and GARCH(1,1) '
    'baseline (no sentiment).  \n'
    'Sections 10+ (EGARCH-X, GJR-X, forecast evaluation) will be appended in subsequent prompts.'
))

# ── SECTION 1 ─────────────────────────────────────────────────────────────────
cells.append(md(
    '## Section 1 — EDA Decision Lock\n\n'
    'Config cell encoding every modelling choice locked down by the EDA. '
    '**Do not edit downstream.**\n\n'
    '| Finding | Value | EDA ref |\n'
    '|---------|-------|---------|\n'
    '| Returns I(0) | ADF+PP reject, KPSS ns | §4 |\n'
    '| ARCH-LM | p≈0.0000 at lags 5/10/20, all metals | §5 |\n'
    '| Engle-Ng sign bias | joint p > 0.47, all metals — asymmetry NOT forced | §5b |\n'
    '| VIF main (polarity, neg_tone) | 2.53 — main GJR-X spec viable | §8b |\n'
    '| VIF auxiliary (pos_tone, neg_tone) | 1.00 — robustness only | §8b |\n'
    '| SENT_LAG_BIC | Nd=8, Pr=6, Dy=9, Tb=4 (BIC min in OLS grid) | §8c |\n'
    '| Granger lag 1 | Nd→log_volume p=0.008, Pr→tone p=0.039, Dy→log_volume p=0.002, Tb: none | §8 |'
))

cells.append(code(
    'import pandas as pd\n'
    '\n'
    "WINDOW_START = '2015-04-01'\n"
    "WINDOW_END   = '2026-04-30'\n"
    "SPLIT_DATE   = pd.Timestamp('2024-02-05')   # 80/20 baseline — EDA §1\n"
    "METALS       = ['Nd', 'Pr', 'Dy', 'Tb']\n"
    "SENT_VARS    = ['tone_mean', 'log_volume', 'polarity_mean', 'neg_tone_mean']\n"
    '\n'
    '# Returns I(0) for all metals (ADF+PP reject, KPSS ns) — EDA §4\n'
    '# ARCH-LM p~0.0000 at lags {5,10,20} all metals — EDA §5, GARCH justified\n'
    '# Sign bias (Engle-Ng) NOT significant for any metal (joint p > 0.47) — EDA §5b\n'
    '#   -> asymmetric GARCH is NOT forced by the data; we run it anyway per\n'
    '#      methodology and report any null result honestly\n'
    '# VIF (polarity_mean, neg_tone_mean) = 2.53 — main GJR-X spec viable\n'
    '# Auxiliary VIF (pos_tone_mean, neg_tone_mean) = 1.00 — robustness only — EDA §8b\n'
    "SENT_LAG_BIC    = {'Nd': 8, 'Pr': 6, 'Dy': 9, 'Tb': 4}   # EDA §8c, primary\n"
    'SENT_LAG_ROBUST = 1                                         # Granger-motivated\n'
    '# Granger (lag 1): Nd->log_volume p=0.008, Pr->tone p=0.039,\n'
    '#   Dy->log_volume p=0.002, Tb: none significant — EDA §8\n'
    '\n'
    'ERR_DIST = {m: None for m in METALS}   # filled in Section 3 if not set here\n'
    '\n'
    "RV_PROXIES = ['RV5', 'RV22']\n"
    '\n'
    "print('EDA Decision Lock loaded.')\n"
    "print(f'SPLIT_DATE     : {SPLIT_DATE.date()}')\n"
    "print(f'SENT_LAG_BIC   : {SENT_LAG_BIC}')\n"
    "print(f'ERR_DIST (init): {ERR_DIST}')"
))

# ── SECTION 2 ─────────────────────────────────────────────────────────────────
cells.append(md(
    '## Section 2 — Load and Reconcile with EDA\n\n'
    'Load the combined panel from `CLEANED DATA/RRE_prices_sentiment_combined.xlsx`, '
    'set Date as index, filter to `[WINDOW_START, WINDOW_END]`. '
    'Confirm shape is **(2 885, 24)** — same as the EDA. Raises if it disagrees.\n\n'
    'The dataset has 25 columns including Date; after `set_index(\'Date\')` the working '
    'frame has 24 columns.'
))

cells.append(code(
    'import numpy as np\n'
    'import os\n'
    'import warnings\n'
    "warnings.filterwarnings('ignore')\n"
    '\n'
    "DATA_PATH = 'CLEANED DATA/RRE_prices_sentiment_combined.xlsx'\n"
    "df_raw = pd.read_excel(DATA_PATH, parse_dates=['Date'])\n"
    "df_raw = df_raw.set_index('Date').sort_index()\n"
    '\n'
    '# Filter to thesis window\n'
    'mask = (df_raw.index >= WINDOW_START) & (df_raw.index <= WINDOW_END)\n'
    'df   = df_raw.loc[mask].copy()\n'
    '\n'
    '# Ground-truth check — EDA confirmed (2885, 24)\n'
    'EXPECTED_SHAPE = (2885, 24)\n'
    'if df.shape != EXPECTED_SHAPE:\n'
    '    raise AssertionError(\n'
    '        f"Shape mismatch: got {df.shape}, expected {EXPECTED_SHAPE}. "\n'
    '        "Check source file or WINDOW_START/WINDOW_END."\n'
    '    )\n'
    '\n'
    'print(f"Shape confirmed : {df.shape[0]:,} rows x {df.shape[1]} columns")\n'
    'print(f"Date range      : {df.index.min().date()} -> {df.index.max().date()}")\n'
    'print(f"\\nColumns ({len(df.columns)}):")\n'
    'for c in df.columns:\n'
    '    print(f"  {c}")'
))

# ── SECTION 3 ─────────────────────────────────────────────────────────────────
cells.append(md(
    '## Section 3 — Error Distribution Selection\n\n'
    'Fat tails and excess kurtosis documented in EDA §3 motivate comparing Normal, '
    'Student-*t*, GED, and Skewed-*t* for GARCH(1,1) error terms. '
    'ARCH-LM p≈0 (EDA §5) confirms GARCH is appropriate.\n\n'
    'Candidate distributions are fitted on the **training sample** (`date < SPLIT_DATE`) '
    'for each metal. BIC winner is adopted per methodology §3.4.2 — BIC\'s stronger '
    'penalty guards against overfitting the tail shape to noise.\n\n'
    'The selected distribution is **fixed for all subsequent model families** '
    '(GARCH(1,1), EGARCH-X, GJR-X) for the same metal. '
    'If `ERR_DIST[metal]` is already non-None from Section 1, it is preserved.'
))

cells.append(code(
    'from arch import arch_model\n'
    '\n'
    "DIST_CANDIDATES = ['normal', 't', 'ged', 'skewt']\n"
    "DIST_LABELS     = {'normal': 'Normal', 't': 'Student-t', 'ged': 'GED', 'skewt': 'Skewed-t'}\n"
    '\n'
    'train_mask      = df.index < SPLIT_DATE\n'
    'dist_table_rows = []\n'
    '\n'
    'for metal in METALS:\n'
    "    ret_train = df.loc[train_mask, f'{metal}_logret'].dropna() * 100\n"
    '    best_bic  = np.inf\n'
    '    best_dist = None\n'
    '\n'
    '    for d in DIST_CANDIDATES:\n'
    "        row = {'Metal': metal, 'Distribution': DIST_LABELS[d],\n"
    "               'LogLik': np.nan, 'AIC': np.nan, 'BIC': np.nan, 'Converged': False}\n"
    '        try:\n'
    "            am  = arch_model(ret_train, mean='Constant', vol='GARCH', p=1, q=1, dist=d)\n"
    "            res = am.fit(disp='off', show_warning=False)\n"
    '            conv = (res.convergence_flag == 0)\n'
    "            row.update({'LogLik': res.loglikelihood,\n"
    "                        'AIC':    res.aic,\n"
    "                        'BIC':    res.bic,\n"
    "                        'Converged': conv})\n"
    '            if conv and res.bic < best_bic:\n'
    '                best_bic  = res.bic\n'
    '                best_dist = d\n'
    '        except Exception:\n'
    '            pass\n'
    '        dist_table_rows.append(row)\n'
    '\n'
    '    # Only overwrite if Section 1 left it as None\n'
    '    if ERR_DIST[metal] is None:\n'
    "        ERR_DIST[metal] = best_dist if best_dist is not None else 't'\n"
    '\n'
    'dist_df = pd.DataFrame(dist_table_rows)\n'
    'print("GARCH(1,1) Distribution Comparison — training sample")\n'
    'print("=" * 72)\n'
    "print(dist_df.to_string(index=False, float_format='{:.4f}'.format))\n"
    'print()\n'
    'print("Final ERR_DIST (BIC winner, fixed across all model families):")\n'
    'for metal, dist in ERR_DIST.items():\n'
    '    print(f"  {metal}: {dist}")'
))

# ── SECTION 4 ─────────────────────────────────────────────────────────────────
cells.append(md(
    '## Section 4 — Sentiment Regressor Construction\n\n'
    'Steps follow methodology §3.3.3:\n\n'
    '1. **No-news-day imputation**: for `tone_mean`, `polarity_mean`, `neg_tone_mean` '
    'replace NaN with the in-sample mean; for `log_volume` replace NaN with 0 '
    '(zero articles = no information).\n'
    '2. **Standardise** using μ_train, σ_train computed on `date < SPLIT_DATE` only '
    '(no leakage).\n'
    '3. **Lag columns**: for each variable × each metal × two lags '
    '(BIC-optimal and lag-1 robustness) → 4 vars × 4 metals × 2 lags = 32 columns.\n'
    '4. **Auxiliary `pos_tone_mean`**: constructed as '
    '`polarity_mean − neg_tone_mean` *before* standardisation '
    '(VIF auxiliary = 1.00, EDA §8b), then standardised and lagged identically '
    'for the GJR-X auxiliary spec.'
))

cells.append(code(
    '# ── Step 1: no-news-day imputation ────────────────────────────────────────\n'
    "FILL_WITH_MEAN = ['tone_mean', 'polarity_mean', 'neg_tone_mean']\n"
    "FILL_WITH_ZERO = ['log_volume']\n"
    '\n'
    'df_sent = df.copy()\n'
    '\n'
    '# Compute in-sample means before filling (date < SPLIT_DATE)\n'
    'insample_means = {}\n'
    'for var in FILL_WITH_MEAN:\n'
    '    insample_means[var] = df_sent.loc[df_sent.index < SPLIT_DATE, var].mean()\n'
    '    df_sent[var] = df_sent[var].fillna(insample_means[var])\n'
    '\n'
    'for var in FILL_WITH_ZERO:\n'
    '    df_sent[var] = df_sent[var].fillna(0.0)\n'
    '\n'
    '# ── Step 2: standardise using in-sample stats only ──────────────────────\n'
    'std_stats = {}   # {var: (mu_train, sigma_train)}\n'
    '\n'
    'for var in SENT_VARS:\n'
    '    series_train = df_sent.loc[df_sent.index < SPLIT_DATE, var]\n'
    '    mu    = series_train.mean()\n'
    '    sigma = series_train.std()\n'
    '    std_stats[var] = (mu, sigma)\n'
    "    df_sent[f'{var}_std'] = (df_sent[var] - mu) / sigma\n"
    '\n'
    'print("In-sample (mu_train, sigma_train) for sentiment series:")\n'
    'for var, (mu, sigma) in std_stats.items():\n'
    '    print(f"  {var:<20s}: mu={mu:.6f}, sigma={sigma:.6f}")'
))

cells.append(code(
    '# ── Step 3: lagged columns — 4 vars x 4 metals x 2 lags = 32 columns ──\n'
    '# BIC-optimal lag per metal (EDA §8c) + lag 1 (Granger-motivated)\n'
    '\n'
    'for var in SENT_VARS:\n'
    "    std_col = f'{var}_std'\n"
    '    for metal in METALS:\n'
    '        lag_bic = SENT_LAG_BIC[metal]\n'
    '        # BIC lag\n'
    "        df_sent[f'{var}_std_lag{lag_bic}_{metal}'] = df_sent[std_col].shift(lag_bic)\n"
    '        # Lag 1\n'
    "        df_sent[f'{var}_std_lag1_{metal}'] = df_sent[std_col].shift(1)\n"
    '\n'
    'lag_cols = [c for c in df_sent.columns if "_std_lag" in c]\n'
    'print(f"Lagged columns created: {len(lag_cols)}")\n'
    'print("Sample column names:")\n'
    'for c in lag_cols[:8]:\n'
    '    print(f"  {c}")'
))

cells.append(code(
    '# ── Step 4: auxiliary pos_tone_mean (EDA §8b, VIF=1.00) ────────────────\n'
    '# Constructed BEFORE standardisation; polarity = pos + neg by GDELT definition\n'
    "df_sent['pos_tone_mean_raw'] = df_sent['polarity_mean'] - df_sent['neg_tone_mean']\n"
    '\n'
    '# Standardise pos_tone_mean with in-sample stats\n'
    "pt_series_train = df_sent.loc[df_sent.index < SPLIT_DATE, 'pos_tone_mean_raw']\n"
    'pt_mu, pt_sigma = pt_series_train.mean(), pt_series_train.std()\n'
    "std_stats['pos_tone_mean'] = (pt_mu, pt_sigma)\n"
    "df_sent['pos_tone_mean_std'] = (df_sent['pos_tone_mean_raw'] - pt_mu) / pt_sigma\n"
    '\n'
    '# Lag for each metal (BIC lag and lag 1)\n'
    'for metal in METALS:\n'
    '    lag_bic = SENT_LAG_BIC[metal]\n'
    "    df_sent[f'pos_tone_mean_std_lag{lag_bic}_{metal}'] = df_sent['pos_tone_mean_std'].shift(lag_bic)\n"
    "    df_sent[f'pos_tone_mean_std_lag1_{metal}'] = df_sent['pos_tone_mean_std'].shift(1)\n"
    '\n'
    'print("pos_tone_mean auxiliary cols created.")\n'
    'print(f"  mu_train={pt_mu:.6f}, sigma_train={pt_sigma:.6f}")'
))

# ── SECTION 5 ─────────────────────────────────────────────────────────────────
cells.append(md(
    '## Section 5 — Auxiliary Columns and Panel Save\n\n'
    'Add `is_train` (boolean) and `sparse_<metal>` indicators per methodology §3.3.3. '
    'The sparsity indicator flags days where the metal-specific article count `n_<metal> < 3`, '
    'i.e. the sentiment observation fell back to the broader rare-earth set. '
    'Save the enriched panel to `outputs/methodology_panel.parquet` for downstream sections.'
))

cells.append(code(
    'os.makedirs("outputs", exist_ok=True)\n'
    '\n'
    '# is_train flag\n'
    "df_sent['is_train'] = df_sent.index < SPLIT_DATE\n"
    '\n'
    '# sparse_<metal> indicator: n_<metal> < 3 triggers fallback to REE-general set\n'
    'for metal in METALS:\n'
    "    n_col = f'n_{metal.lower()}'\n"
    '    if n_col in df_sent.columns:\n'
    "        df_sent[f'sparse_{metal}'] = (df_sent[n_col] < 3).astype(int)\n"
    '    else:\n'
    "        df_sent[f'sparse_{metal}'] = np.nan\n"
    "        print(f'WARNING: {n_col} not found — sparse_{metal} set to NaN')\n"
    '\n'
    '# Save panel\n'
    "panel_path = 'outputs/methodology_panel.parquet'\n"
    "df_sent.to_parquet(panel_path)\n"
    'print(f"Panel saved: {panel_path}")\n'
    'print(f"Panel shape : {df_sent.shape[0]:,} rows x {df_sent.shape[1]} columns")\n'
    'print(f"is_train    : {df_sent[\'is_train\'].sum():,} train / {(~df_sent[\'is_train\']).sum():,} test")\n'
    'print("\\nSparse day counts per metal:")\n'
    'for metal in METALS:\n'
    "    col = f'sparse_{metal}'\n"
    '    if col in df_sent.columns and not df_sent[col].isna().all():\n'
    '        print(f"  {metal}: {int(df_sent[col].sum())} sparse days ({df_sent[col].mean()*100:.1f}%)")'
))

# ── SECTION 6 ─────────────────────────────────────────────────────────────────
cells.append(md(
    '## Section 6 — Historical Simulation Benchmark\n\n'
    'Rolling-variance forecasts at windows N ∈ {20, 30, 60} days serve as the '
    'minimum bar that all parametric models must clear — methodology §3.4.1.\n\n'
    'One-step-ahead forecast: σ̂²_{t+1} = rolling Var over the prior N days of returns. '
    'Forecasts are restricted to the test period (`date >= SPLIT_DATE`) to match the '
    'out-of-sample evaluation window. Results saved to '
    '`outputs/forecasts_HS_<metal>_N<N>.parquet`.'
))

cells.append(code(
    'HS_WINDOWS = [20, 30, 60]\n'
    '\n'
    'hs_forecast_store = {}   # {(metal, N): Series}\n'
    '\n'
    'hs_summary_rows = []\n'
    'test_mask = df_sent.index >= SPLIT_DATE\n'
    '\n'
    'for metal in METALS:\n'
    "    ret = df_sent[f'{metal}_logret']\n"
    '    for N in HS_WINDOWS:\n'
    '        # Rolling variance (ddof=1 to match pandas default)\n'
    '        # Shift(1) so that on date t the forecast uses only data up to t-1\n'
    '        rolling_var = ret.rolling(N).var().shift(1)\n'
    '        forecast_test = rolling_var.loc[test_mask]\n'
    '\n'
    '        hs_forecast_store[(metal, N)] = forecast_test\n'
    '\n'
    '        # Save\n'
    "        fpath = f'outputs/forecasts_HS_{metal}_N{N}.parquet'\n"
    '        forecast_test.to_frame("forecast_var").to_parquet(fpath)\n'
    '\n'
    '        hs_summary_rows.append({\n'
    "            'Metal': metal, 'Window': N,\n"
    "            'Mean': forecast_test.mean(),\n"
    "            'Median': forecast_test.median(),\n"
    "            'Max': forecast_test.max(),\n"
    "            'N_obs': forecast_test.notna().sum()\n"
    '        })\n'
    '\n'
    'hs_df = pd.DataFrame(hs_summary_rows)\n'
    'print("Historical Simulation — test-period forecast variance summary")\n'
    'print("=" * 60)\n'
    "print(hs_df.to_string(index=False, float_format='{:.6f}'.format))"
))

# ── SECTION 7 ─────────────────────────────────────────────────────────────────
cells.append(md(
    '## Section 7 — GARCH(1,1) Baseline (No Sentiment)\n\n'
    'Standard GARCH(1,1) with the per-metal error distribution from Section 3. '
    'This is the parametric baseline that must be beaten before sentiment adds value '
    '— methodology §3.4.2.\n\n'
    '**Expanding-window** one-step-ahead forecasts over the test period. '
    'Parameters are re-estimated at every test step; if wall-clock time exceeds '
    '~30 minutes across all four metals, re-estimation is switched to every 22 '
    'trading days (one calendar month) — this choice is flagged in the next '
    'markdown cell.\n\n'
    'Saved artifacts per metal:\n'
    '- `outputs/garch11_<metal>.pkl` — fitted arch `ARCHModelResult` for the full training fit\n'
    '- `outputs/forecasts_garch11_<metal>.parquet` — test-period one-step-ahead variance forecasts\n\n'
    'An in-memory dict `garch11_results` is also kept for reuse in Section 8.'
))

cells.append(code(
    'import pickle\n'
    'import time\n'
    '\n'
    "garch11_results   = {}   # {metal: ARCHModelResult} full-sample training fit\n"
    "garch11_forecasts = {}   # {metal: pd.Series}  test-period sigma^2 forecasts\n"
    '\n'
    'IGARCH_FLAG  = {}   # metals where alpha+beta > 0.99\n'
    'REFIT_FREQ   = 1    # 1 = every step (expanding); changed to 22 if slow\n'
    'TIMING_LIMIT = 1800 # 30-minute wall-clock budget across all metals (seconds)\n'
    '\n'
    'test_dates  = df_sent.index[test_mask]\n'
    'all_dates   = df_sent.index\n'
    '\n'
    'total_start = time.time()\n'
    '\n'
    'for metal in METALS:\n'
    "    ret_full = df_sent[f'{metal}_logret'].dropna() * 100\n"
    "    dist     = ERR_DIST[metal]\n"
    '    metal_forecasts = []\n'
    '\n'
    '    # ── Full-training fit (saved to disk) ──────────────────────────────\n'
    "    am_full  = arch_model(ret_full.loc[ret_full.index < SPLIT_DATE],\n"
    "                          mean='Constant', vol='GARCH', p=1, q=1, dist=dist)\n"
    "    res_full = am_full.fit(disp='off', show_warning=False)\n"
    "    garch11_results[metal] = res_full\n"
    '\n'
    "    pkl_path = f'outputs/garch11_{metal}.pkl'\n"
    '    with open(pkl_path, \'wb\') as fh:\n'
    '        pickle.dump(res_full, fh)\n'
    '\n'
    '    # Stationarity + constraint checks\n'
    "    alpha = res_full.params.get('alpha[1]', res_full.params.get('a[1]', np.nan))\n"
    "    beta  = res_full.params.get('beta[1]',  res_full.params.get('b[1]',  np.nan))\n"
    "    omega = res_full.params.get('omega', np.nan)\n"
    '    ab    = alpha + beta\n'
    '    if ab > 0.99:\n'
    '        print(f"WARNING [{metal}]: alpha+beta = {ab:.4f} > 0.99 (near-IGARCH)")\n'
    '        IGARCH_FLAG[metal] = True\n'
    '    else:\n'
    '        IGARCH_FLAG[metal] = False\n'
    '\n'
    '    # ── Expanding-window forecasts ────────────────────────────────────\n'
    '    metal_start = time.time()\n'
    '    step_count  = 0\n'
    '\n'
    '    for t_date in test_dates:\n'
    '        t_pos = all_dates.get_loc(t_date)\n'
    '        # Use all data up to but not including t_date\n'
    '        ret_window = ret_full.iloc[:t_pos]\n'
    '\n'
    '        try:\n'
    "            am  = arch_model(ret_window, mean='Constant',\n"
    "                             vol='GARCH', p=1, q=1, dist=dist)\n"
    '            if step_count % REFIT_FREQ == 0:\n'
    "                res = am.fit(disp='off', show_warning=False,\n"
    '                            starting_values=res_full.params.values)\n'
    '            fcast = res.forecast(horizon=1, reindex=False)\n'
    "            var1  = fcast.variance.iloc[-1, 0] / 1e4   # back to decimal^2\n"
    '        except Exception:\n'
    '            var1  = np.nan\n'
    '\n'
    '        metal_forecasts.append((t_date, var1))\n'
    '        step_count += 1\n'
    '\n'
    '        # Budget check: if > 15 min elapsed for this metal, switch to 22-day refit\n'
    '        if REFIT_FREQ == 1 and (time.time() - metal_start) > 900 and step_count < 50:\n'
    '            REFIT_FREQ = 22\n'
    '            print(f"[{metal}] Slow convergence detected — switching to 22-day refit")\n'
    '\n'
    '    fc_series = pd.Series(\n'
    '        {d: v for d, v in metal_forecasts}, name="forecast_var"\n'
    '    )\n'
    "    garch11_forecasts[metal] = fc_series\n"
    "    fpath = f'outputs/forecasts_garch11_{metal}.parquet'\n"
    '    fc_series.to_frame().to_parquet(fpath)\n'
    '\n'
    '    elapsed = time.time() - metal_start\n'
    '    print(f"[{metal}] Done — {len(fc_series)} forecasts, {elapsed:.1f}s")\n'
    '\n'
    '    if (time.time() - total_start) > TIMING_LIMIT:\n'
    '        print("Budget exceeded — remaining metals will use 22-day refit")\n'
    '        REFIT_FREQ = 22\n'
    '\n'
    'print(f"\\nRefit frequency used: every {REFIT_FREQ} step(s)")\n'
    'if REFIT_FREQ != 1:\n'
    '    print("NOTE: 22-day refit used to keep runtime tractable. "\n'
    '          "Parameters held fixed between refits.")'
))

cells.append(md(
    '### Section 7 — Refit-frequency note\n\n'
    'If the cell above printed *"Slow convergence detected — switching to 22-day refit"* '
    'for any metal, parameters were held constant for up to 22 steps at a time and '
    'only the forecast was rolled forward. This is a standard computational shortcut '
    '(see Engle & Rangel 2008) and is flagged in the Section 9 readiness summary. '
    'If all four metals completed with `REFIT_FREQ = 1`, full expanding-window '
    'estimation was achieved.'
))

# ── SECTION 8 ─────────────────────────────────────────────────────────────────
cells.append(md(
    '## Section 8 — In-Sample Diagnostics for the GARCH(1,1) Baseline\n\n'
    'Fitted on the training sample. Three diagnostic batteries:\n\n'
    '1. **Ljung-Box on standardised residuals** at lags 5/10/20 — tests for residual '
    'autocorrelation (should be absent if the mean equation is correctly specified).\n'
    '2. **Ljung-Box on *squared* standardised residuals** at lags 5/10/20 — tests for '
    'remaining ARCH effects (should be absent if the variance equation is correctly specified).\n'
    '3. **Engle-Ng sign bias test** — should reproduce the EDA §5b null finding '
    '(joint p > 0.47). Stops and reports if it does not.'
))

cells.append(code(
    'import statsmodels.api as sm\n'
    'from statsmodels.stats.diagnostic import acorr_ljungbox\n'
    '\n'
    'diag_rows = []\n'
    '\n'
    'for metal in METALS:\n'
    '    res  = garch11_results[metal]\n'
    '    zhat = res.std_resid.dropna()\n'
    '\n'
    '    for lag in [5, 10, 20]:\n'
    '        # Ljung-Box on standardised residuals\n'
    '        lb_z  = acorr_ljungbox(zhat,      lags=[lag], return_df=True)\n'
    '        lb_z2 = acorr_ljungbox(zhat ** 2, lags=[lag], return_df=True)\n'
    '\n'
    '        diag_rows.append({\n'
    "            'Metal': metal, 'Test': 'LB-z', 'Lag': lag,\n"
    "            'Stat': lb_z['lb_stat'].iloc[0],\n"
    "            'p-value': lb_z['lb_pvalue'].iloc[0]\n"
    '        })\n'
    '        diag_rows.append({\n'
    "            'Metal': metal, 'Test': 'LB-z^2', 'Lag': lag,\n"
    "            'Stat': lb_z2['lb_stat'].iloc[0],\n"
    "            'p-value': lb_z2['lb_pvalue'].iloc[0]\n"
    '        })\n'
    '\n'
    'diag_df = pd.DataFrame(diag_rows)\n'
    'print("Ljung-Box diagnostics on GARCH(1,1) standardised residuals")\n'
    'print("=" * 62)\n'
    "print(diag_df.to_string(index=False, float_format='{:.4f}'.format))"
))

cells.append(code(
    '# ── Engle-Ng sign bias test ────────────────────────────────────────────\n'
    '# Reproduces the EDA §5b null finding: joint p > 0.47 for all metals.\n'
    '# If significance is found here (joint p < 0.05), execution stops.\n'
    '\n'
    'sb_rows = []\n'
    'SIGN_BIAS_THRESHOLD = 0.05\n'
    'sign_bias_alert = False\n'
    '\n'
    'for metal in METALS:\n'
    '    res  = garch11_results[metal]\n'
    '    z    = res.std_resid.dropna()\n'
    '    z_lag = z.shift(1)\n'
    '    dep   = z ** 2\n'
    '\n'
    '    mask_valid = z_lag.notna()\n'
    '    dep_v  = dep[mask_valid]\n'
    '    zl_v   = z_lag[mask_valid]\n'
    '\n'
    '    S_neg   = (zl_v < 0).astype(float)\n'
    '    neg_sz  = S_neg * zl_v\n'
    '    pos_sz  = (1 - S_neg) * zl_v\n'
    '\n'
    '    X = sm.add_constant(pd.DataFrame({\n'
    "        'S_neg': S_neg, 'neg_size': neg_sz, 'pos_size': pos_sz\n"
    '    }))\n'
    '    ols = sm.OLS(dep_v, X).fit()\n'
    '\n'
    '    r_matrix = [[0,1,0,0],[0,0,1,0],[0,0,0,1]]\n'
    '    f_test   = ols.f_test(r_matrix)\n'
    '    joint_p  = float(f_test.pvalue)\n'
    '\n'
    '    sb_rows.append({\n'
    "        'Metal':      metal,\n"
    "        'SignBias_t':  ols.tvalues['S_neg'],\n"
    "        'SignBias_p':  ols.pvalues['S_neg'],\n"
    "        'NegSize_t':   ols.tvalues['neg_size'],\n"
    "        'NegSize_p':   ols.pvalues['neg_size'],\n"
    "        'PosSize_t':   ols.tvalues['pos_size'],\n"
    "        'PosSize_p':   ols.pvalues['pos_size'],\n"
    "        'JointF':      float(f_test.fvalue),\n"
    "        'JointP':      joint_p\n"
    '    })\n'
    '\n'
    '    if joint_p < SIGN_BIAS_THRESHOLD:\n'
    '        sign_bias_alert = True\n'
    '        print(f"DISCREPANCY [{metal}]: sign bias joint p={joint_p:.4f} < 0.05. "\n'
    '              f"EDA §5b found p>0.47. Stop and investigate before continuing.")\n'
    '\n'
    'sb_df = pd.DataFrame(sb_rows)\n'
    'print("Engle-Ng Sign Bias Test — GARCH(1,1) standardised residuals")\n'
    'print("=" * 80)\n'
    "print(sb_df.to_string(index=False, float_format='{:.4f}'.format))\n"
    '\n'
    'if sign_bias_alert:\n'
    '    raise RuntimeError(\n'
    '        "Sign bias significant for at least one metal — "\n'
    '        "reconcile with EDA §5b before proceeding to Section 9."\n'
    '    )\n'
    'else:\n'
    '    print("\\nSign bias null confirmed for all metals (consistent with EDA §5b).")'
))

# ── SECTION 9 ─────────────────────────────────────────────────────────────────
cells.append(md(
    '## Section 9 — Readiness Summary\n\n'
    'All artifacts from Sections 1–8 are saved under `outputs/`. '
    'This cell prints a human-readable summary for inclusion in the thesis appendix '
    'and flags any anomalies for the robustness battery.'
))

cells.append(code(
    'import glob\n'
    '\n'
    'print("=" * 70)\n'
    'print("READINESS SUMMARY — Sections 1–9")\n'
    'print("=" * 70)\n'
    '\n'
    '# ── 1. Saved artifacts ────────────────────────────────────────────────\n'
    'print("\\n[1] Artifacts saved under outputs/")\n'
    'for fpath in sorted(glob.glob("outputs/*")):\n'
    '    size_kb = os.path.getsize(fpath) / 1024\n'
    '    print(f"    {os.path.basename(fpath):<45s} {size_kb:6.1f} KB")\n'
    '\n'
    '# ── 2. Sentiment standardisation stats ───────────────────────────────\n'
    'print("\\n[2] Sentiment in-sample standardisation statistics")\n'
    "print(f\"    {'Variable':<22s} {'mu_train':>12s} {'sigma_train':>12s}\")\n"
    'for var, (mu, sigma) in std_stats.items():\n'
    "    print(f\"    {var:<22s} {mu:>12.6f} {sigma:>12.6f}\")\n"
    '\n'
    '# ── 3. Error distributions ───────────────────────────────────────────\n'
    'print("\\n[3] Error distributions (fixed across all model families)")\n'
    'for metal, dist in ERR_DIST.items():\n'
    '    print(f"    {metal}: {dist}")\n'
    '\n'
    '# ── 4. GARCH(1,1) parameter table ────────────────────────────────────\n'
    'print("\\n[4] GARCH(1,1) parameter table (full training-sample fit)")\n'
    "hdr = f\"  {'Metal':<6s} {'omega':>12s} {'alpha':>10s} {'beta':>10s} \"\\\n"
    "      f\"{'shape/nu':>10s} {'a+b':>8s} {'LogLik':>10s} {'AIC':>10s} {'BIC':>10s}\"\n"
    'print(hdr)\n'
    'print("  " + "-" * (len(hdr)-2))\n'
    '\n'
    'param_flags = []\n'
    'for metal in METALS:\n'
    '    res    = garch11_results[metal]\n'
    '    params = res.params\n'
    "    omega  = params.get('omega', np.nan)\n"
    "    alpha  = params.get('alpha[1]', params.get('a[1]', np.nan))\n"
    "    beta   = params.get('beta[1]',  params.get('b[1]',  np.nan))\n"
    '    # shape parameter name differs by distribution\n'
    '    shape  = np.nan\n'
    "    for pname in ['nu', 'eta', 'lambda', 'shape']:\n"
    '        if pname in params.index:\n'
    '            shape = params[pname]; break\n'
    '    ab = alpha + beta\n'
    '\n'
    '    flag = ""\n'
    '    if ab > 0.99:\n'
    '        flag += " [IGARCH]"\n'
    '        param_flags.append(f"{metal}: alpha+beta={ab:.4f} > 0.99 near-IGARCH")\n'
    '    if res.convergence_flag != 0:\n'
    '        flag += " [CONV?]"\n'
    '        param_flags.append(f"{metal}: convergence_flag={res.convergence_flag}")\n'
    '\n'
    '    print(\n'
    '        f"  {metal:<6s} {omega:>12.6f} {alpha:>10.6f} {beta:>10.6f} "\n'
    '        f"{shape:>10.4f} {ab:>8.4f} {res.loglikelihood:>10.2f} "\n'
    '        f"{res.aic:>10.2f} {res.bic:>10.2f}{flag}"\n'
    '    )\n'
    '\n'
    '# ── 5. Flags ─────────────────────────────────────────────────────────\n'
    'print("\\n[5] Flags for robustness battery")\n'
    'if param_flags:\n'
    '    for flag in param_flags:\n'
    '        print(f"  FLAG: {flag}")\n'
    'else:\n'
    '    print("  None — all metals converged, all alpha+beta < 0.99")\n'
    '\n'
    'if REFIT_FREQ != 1:\n'
    '    print(f"  FLAG: Expanding-window refit used every {REFIT_FREQ} steps (runtime budget)")\n'
    '\n'
    'print("\\n[6] Sign bias (Engle-Ng) — confirming EDA §5b")\n'
    'for _, row in sb_df.iterrows():\n'
    '    print(f"  {row[\'Metal\']}: joint p={row[\'JointP\']:.4f}")\n'
    '\n'
    'print("\\n" + "=" * 70)\n'
    'print("Sections 1–9 complete. Ready for Sections 10+ (EGARCH-X, GJR-X, evaluation).")\n'
    'print("=" * 70)'
))

# ── ASSEMBLE AND WRITE ────────────────────────────────────────────────────────
nb = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.9.0'
        }
    },
    'cells': cells
}

with open(OUT, 'w', encoding='utf-8') as fh:
    json.dump(nb, fh, ensure_ascii=False, indent=1)

print(f"Written {len(cells)} cells to {OUT}")
