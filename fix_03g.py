"""
Patch 03-GP-models-gaussian.ipynb
Bugs addressed:
  Bug 1  -- remove .shift(1) double-lag in GARCHND
  Bug 3  -- swap column refs to winsorized variants
  Bug 4  -- zero-mean returns + mean='Zero' baseline
  Bug 5  -- relax theta/gamma bounds
  Bug 6  -- boundary-solution detection in every custom-MLE cell
  Bug 7  -- data-density diagnostic cell (new cell after data load)
"""
import json, copy

NB_PATH = '03-GP-models-gaussian.ipynb'

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

def get_cell(nb, cell_id):
    for c in nb['cells']:
        if c.get('id','') == cell_id:
            return c
    raise RuntimeError(f'Cell {cell_id} not found')

def set_source(cell, new_src):
    lines = new_src.splitlines()
    cell['source'] = [line + '\n' for line in lines]
    if cell['source']:
        cell['source'][-1] = cell['source'][-1].rstrip('\n')

def src(cell):
    return ''.join(cell['source'])

# ── Cell IDs (from earlier read) ─────────────────────────────────────────────
DATA_LOAD_ID       = 'a7fc7832'
BASELINE_ID        = '2f2da668'
GARCHX_TONE_ID     = 'ffd80124'
GARCHX_ART_ID      = '596c642d'
GARCHAND_TONE_ID   = 'c8fff7e2'
GARCHAND_ART_ID    = '5600ad19'
GARCHND_ID         = '1d09f2a7'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Bug 4 — data-load cell: add zero-mean demeaning
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
c = get_cell(nb, DATA_LOAD_ID)
s = src(c)
old = "df['r'] = np.log(df['Close'] / df['Close'].shift(1))\nr = df['r'].dropna()"
new = "df['r'] = np.log(df['Close'] / df['Close'].shift(1))\ndf['r'] = df['r'] - df['r'].mean()  # zero-mean (Bug 4)\nr = df['r'].dropna()"
if old not in s:
    raise RuntimeError(f'Bug4 data-load pattern not found in {DATA_LOAD_ID}')
set_source(c, s.replace(old, new))
print('Bug 4 data-load: OK')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Bug 7 — new diagnostic cell (insert after data-load cell)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIAG_CELL_SOURCE = (
    "# Bug-7 diagnostic: data-density check\n"
    "# 02-EDA.ipynb does not contain these article-density diagnostics,\n"
    "# so we add them here. Read-only — no model changes.\n"
    "\n"
    "_ac  = df['article_count'] if 'article_count' in df.columns else pd.Series(dtype=float)\n"
    "_tm  = df['tone_mean_x100_winsor'] if 'tone_mean_x100_winsor' in df.columns else df['tone_mean']\n"
    "_ag  = df['art_growth_winsor'] if 'art_growth_winsor' in df.columns else df['art_growth']\n"
    "_r2  = df['r'].dropna() ** 2  # crude RV proxy (squared returns)\n"
    "\n"
    "if len(_ac) > 0:\n"
    "    print('=== Article-count distribution ===')\n"
    "    print(_ac.describe(percentiles=[.01,.05,.25,.50,.75,.95,.99]).to_string())\n"
    "    zero_art_frac = float((_ac == 0).mean())\n"
    "    print(f'Share of trading days with 0 articles: {zero_art_frac:.3%}')\n"
    "\n"
    "    # tone_mean fill value for zero-article days is 0 (see 01-data-cleaning)\n"
    "    _fill_val = 0.0\n"
    "    zero_tone_frac = float((df['tone_mean'] == _fill_val).mean())\n"
    "    print(f'Share of trading days where tone_mean == fill value ({_fill_val}): {zero_tone_frac:.3%}')\n"
    "\n"
    "print()\n"
    "print('=== tone_mean_x100 and art_growth: raw vs winsorized ===')\n"
    "for _col in ['tone_mean', 'tone_mean_x100', 'tone_mean_x100_winsor',\n"
    "             'art_growth', 'art_growth_winsor']:\n"
    "    if _col in df.columns:\n"
    "        s_ = df[_col]\n"
    "        print(f'{_col:30s}: mean={s_.mean():+.4f}  std={s_.std():.4f}  '\n"
    "              f'min={s_.min():.4f}  max={s_.max():.4f}')\n"
    "\n"
    "print()\n"
    "print('=== Correlations with squared returns (crude RV proxy) ===')\n"
    "print(f'Lag     tone    |tone|   art_growth')\n"
    "for _lag in [1, 5, 22]:\n"
    "    _rv = _r2.shift(-_lag)  # forward shift so we see r^2_{t+lag} ~ x_t\n"
    "    _df_tmp = pd.DataFrame({'tone': _tm, 'abs_tone': _tm.abs(),\n"
    "                            'ag': _ag, 'rv': _rv}).dropna()\n"
    "    ct = _df_tmp[['tone','rv']].corr().iloc[0,1]\n"
    "    ca = _df_tmp[['abs_tone','rv']].corr().iloc[0,1]\n"
    "    cg = _df_tmp[['ag','rv']].corr().iloc[0,1]\n"
    "    print(f'{_lag:>3d}   {ct:+.4f}  {ca:+.4f}   {cg:+.4f}')\n"
)

# Build new diagnostic cell dict
diag_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "diag_density_g01",
    "metadata": {},
    "outputs": [],
    "source": [line + '\n' for line in DIAG_CELL_SOURCE.splitlines()]
}
if diag_cell['source']:
    diag_cell['source'][-1] = diag_cell['source'][-1].rstrip('\n')

# Insert after DATA_LOAD_ID
data_load_idx = next(i for i, c in enumerate(nb['cells'])
                     if c.get('id','') == DATA_LOAD_ID)
nb['cells'].insert(data_load_idx + 1, diag_cell)
print('Bug 7 diagnostic cell: inserted after data-load')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Bug 4 — baseline GARCH: mean='Constant' -> mean='Zero'
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
c = get_cell(nb, BASELINE_ID)
s = src(c)
if "mean='Constant'" not in s:
    raise RuntimeError(f"mean='Constant' not found in {BASELINE_ID}")
set_source(c, s.replace("mean='Constant'", "mean='Zero'"))
print('Bug 4 baseline mean=Zero: OK')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Boundary-detection helper (appended to every custom-MLE cell)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BOUND_CHECK = (
    "\n# Bug-6: boundary-solution check\n"
    "on_bound = []\n"
    "for _nm, _v, (_lo, _hi) in zip(PARAM_NAMES, theta, bounds):\n"
    "    if _lo is not None and abs(_v - _lo) < 1e-6: on_bound.append((_nm, _v, 'lower', _lo))\n"
    "    if _hi is not None and abs(_v - _hi) < 1e-6: on_bound.append((_nm, _v, 'upper', _hi))\n"
    "if on_bound:\n"
    "    print('WARNING: boundary solution — SE/t/p not interpretable for:', on_bound)\n"
)

# ── GARCH-X tone (ffd80124) ──────────────────────────────────────────────────
c = get_cell(nb, GARCHX_TONE_ID)
s = src(c)
# Bug 3: column ref
s = s.replace("tone = df['tone_mean']", "tone = df['tone_mean_x100_winsor']")
# Bug 6: append boundary check (before _report call)
report_call = "_report(PARAM_NAMES, theta, se, ll, T_obs,"
if report_call not in s:
    raise RuntimeError(f'GARCH-X tone: _report call not found')
s = s.replace(report_call, BOUND_CHECK + report_call, 1)
set_source(c, s)
print('GARCH-X tone (Bug 3+6): OK')

# ── GARCH-X art (596c642d) ───────────────────────────────────────────────────
c = get_cell(nb, GARCHX_ART_ID)
s = src(c)
s = s.replace("artg = df['art_growth']", "artg = df['art_growth_winsor']")
report_call = "_report(PARAM_NAMES, theta, se, ll, T_obs,"
if report_call not in s:
    raise RuntimeError(f'GARCH-X art: _report call not found')
s = s.replace(report_call, BOUND_CHECK + report_call, 1)
set_source(c, s)
print('GARCH-X art (Bug 3+6): OK')

# ── GARCHAND tone (c8fff7e2) ─────────────────────────────────────────────────
c = get_cell(nb, GARCHAND_TONE_ID)
s = src(c)
# Bug 3
s = s.replace("tone = df['tone_mean']", "tone = df['tone_mean_x100_winsor']")
# Bug 5: theta bound (1e-8, None) -> (-0.999, None)
s = s.replace("(1e-8, None), (0.0, 0.999), (0.0, 0.999), (None, None), (1e-8, None)",
              "(1e-8, None), (0.0, 0.999), (0.0, 0.999), (None, None), (-0.999, None)")
# Bug 6
report_call = "_report(PARAM_NAMES, theta_hat, se, ll, T_obs,"
if report_call not in s:
    raise RuntimeError(f'GARCHAND tone: _report call not found')
s = s.replace(report_call,
    "\n# Bug-6: boundary-solution check\n"
    "on_bound = []\n"
    "for _nm, _v, (_lo, _hi) in zip(PARAM_NAMES, theta_hat, bounds):\n"
    "    if _lo is not None and abs(_v - _lo) < 1e-6: on_bound.append((_nm, _v, 'lower', _lo))\n"
    "    if _hi is not None and abs(_v - _hi) < 1e-6: on_bound.append((_nm, _v, 'upper', _hi))\n"
    "if on_bound:\n"
    "    print('WARNING: boundary solution -- SE/t/p not interpretable for:', on_bound)\n"
    + report_call, 1)
set_source(c, s)
print('GARCHAND tone (Bug 3+5+6): OK')

# ── GARCHAND art (5600ad19) ──────────────────────────────────────────────────
c = get_cell(nb, GARCHAND_ART_ID)
s = src(c)
# Bug 3
s = s.replace("artg = df['art_growth']", "artg = df['art_growth_winsor']")
# Bug 5: gamma bound (1e-8, None) -> (None, None)  for GARCHAND art
s = s.replace("(1e-8, None), (0.0, 0.999), (0.0, 0.999), (1e-8, None)",
              "(1e-8, None), (0.0, 0.999), (0.0, 0.999), (None, None)")
# Bug 6
report_call = "_report(PARAM_NAMES, theta, se, ll, T_obs,"
if report_call not in s:
    raise RuntimeError(f'GARCHAND art: _report call not found')
s = s.replace(report_call, BOUND_CHECK + report_call, 1)
set_source(c, s)
print('GARCHAND art (Bug 3+5+6): OK')

# ── GARCHND (1d09f2a7) ───────────────────────────────────────────────────────
c = get_cell(nb, GARCHND_ID)
s = src(c)
# Bug 1: remove .shift(1) double-lag
s = s.replace("tone_lag = df['tone_mean'].shift(1)", "tone_lag = df['tone_mean_x100_winsor']")
s = s.replace("artg_lag = df['art_growth'].shift(1)", "artg_lag = df['art_growth_winsor']")
# Bug 6: GARCHND uses _report inside fit_garchnd, add boundary check after _report call
# The boundary check must use the local 'bounds' and 'theta' from fit_garchnd
GARCHND_BOUND_CHECK = (
    "    # Bug-6: boundary-solution check\n"
    "    on_bound = []\n"
    "    for _nm, _v, (_lo, _hi) in zip(PARAM_NAMES, theta, bounds):\n"
    "        if _lo is not None and abs(_v - _lo) < 1e-6: on_bound.append((_nm, _v, 'lower', _lo))\n"
    "        if _hi is not None and abs(_v - _hi) < 1e-6: on_bound.append((_nm, _v, 'upper', _hi))\n"
    "    if on_bound:\n"
    "        print('WARNING: boundary solution -- SE/t/p not interpretable for:', on_bound)\n"
    "    print()\n"
)
# Replace the trailing `    print()` inside fit_garchnd with the check + print
s = s.replace(
    "    _report(PARAM_NAMES, theta, se, ll, T_obs,\n"
    "            persist_alpha_beta=theta[1] + theta[2],\n"
    "            extra_notes=extra)\n"
    "    print()\n",
    "    _report(PARAM_NAMES, theta, se, ll, T_obs,\n"
    "            persist_alpha_beta=theta[1] + theta[2],\n"
    "            extra_notes=extra)\n"
    + GARCHND_BOUND_CHECK
)
set_source(c, s)
print('GARCHND (Bug 1+3+6): OK')

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f'\n{NB_PATH} patched successfully.')
