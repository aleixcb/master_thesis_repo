"""
Patch 03-GP-models-student.ipynb
Bugs: 1 (double-lag GARCHND), 3 (column refs), 4 (zero-mean), 5 (bounds), 6 (boundary check)
"""
import json

NB_PATH = '03-GP-models-student.ipynb'

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

def get_cell(nb, cid):
    for c in nb['cells']:
        if c.get('id','') == cid:
            return c
    raise RuntimeError(f'Cell {cid} not found')

def set_source(cell, new_src):
    lines = new_src.splitlines()
    cell['source'] = [line + '\n' for line in lines]
    if cell['source']:
        cell['source'][-1] = cell['source'][-1].rstrip('\n')

def src(cell):
    return ''.join(cell['source'])

DATA_LOAD_ID     = 'e0d560eb'
BASELINE_ID      = '6701d8b4'
GARCHX_TONE_ID   = 'f3b1f0dc'
GARCHX_ART_ID    = '4f37f9a2'
GARCHAND_TONE_ID = 'faed8fb4'
GARCHAND_ART_ID  = '89e3bbc9'
GARCHND_ID       = '4a5a5d2d'

# ── Bug 4: data-load — add zero-mean demeaning ───────────────────────────────
c = get_cell(nb, DATA_LOAD_ID)
s = src(c)
old = "df['r'] = np.log(df['Close'] / df['Close'].shift(1))\nr = df['r'].dropna()"
new = "df['r'] = np.log(df['Close'] / df['Close'].shift(1))\ndf['r'] = df['r'] - df['r'].mean()  # zero-mean (Bug 4)\nr = df['r'].dropna()"
if old not in s:
    raise RuntimeError(f'Bug4 pattern not found in {DATA_LOAD_ID}: {repr(s[:200])}')
set_source(c, s.replace(old, new))
print('Bug 4 data-load: OK')

# ── Bug 4: baseline GARCH mean='Constant' -> mean='Zero' ─────────────────────
c = get_cell(nb, BASELINE_ID)
s = src(c)
if "mean='Constant'" not in s:
    raise RuntimeError(f"mean='Constant' not in {BASELINE_ID}")
set_source(c, s.replace("mean='Constant'", "mean='Zero'"))
print('Bug 4 baseline mean=Zero: OK')

BOUND_CHECK = (
    "\n# Bug-6: boundary-solution check\n"
    "on_bound = []\n"
    "for _nm, _v, (_lo, _hi) in zip(PARAM_NAMES, theta, bounds):\n"
    "    if _lo is not None and abs(_v - _lo) < 1e-6: on_bound.append((_nm, _v, 'lower', _lo))\n"
    "    if _hi is not None and abs(_v - _hi) < 1e-6: on_bound.append((_nm, _v, 'upper', _hi))\n"
    "if on_bound:\n"
    "    print('WARNING: boundary solution -- SE/t/p not interpretable for:', on_bound)\n"
)

# ── GARCH-X tone (Bug 3+6) ───────────────────────────────────────────────────
c = get_cell(nb, GARCHX_TONE_ID)
s = src(c)
s = s.replace("tone = df['tone_mean']", "tone = df['tone_mean_x100_winsor']")
report_call = "_report(PARAM_NAMES, theta, se, ll, T_obs,"
if report_call not in s:
    raise RuntimeError(f'GARCH-X tone _report not found')
s = s.replace(report_call, BOUND_CHECK + report_call, 1)
set_source(c, s)
print('GARCH-X tone (Bug 3+6): OK')

# ── GARCH-X art (Bug 3+6) ────────────────────────────────────────────────────
c = get_cell(nb, GARCHX_ART_ID)
s = src(c)
s = s.replace("artg = df['art_growth']", "artg = df['art_growth_winsor']")
report_call = "_report(PARAM_NAMES, theta, se, ll, T_obs,"
if report_call not in s:
    raise RuntimeError(f'GARCH-X art _report not found')
s = s.replace(report_call, BOUND_CHECK + report_call, 1)
set_source(c, s)
print('GARCH-X art (Bug 3+6): OK')

# ── GARCHAND tone (Bug 3+5+6) ───────────────────────────────────────────────
c = get_cell(nb, GARCHAND_TONE_ID)
s = src(c)
s = s.replace("tone = df['tone_mean']", "tone = df['tone_mean_x100_winsor']")
# Bug 5: theta bound for student-t includes nu, so pattern is:
# (1e-8, None), (2.05, 100.0)  -- the (1e-8, None) before nu is theta
s = s.replace(
    "(1e-8, None), (0.0, 0.999), (0.0, 0.999), (None, None),\n          (1e-8, None), (2.05, 100.0)",
    "(1e-8, None), (0.0, 0.999), (0.0, 0.999), (None, None),\n          (-0.999, None), (2.05, 100.0)"
)
# Bug 6: theta_hat used instead of theta in GARCHAND
report_call = "_report(PARAM_NAMES, theta_hat, se, ll, T_obs,"
if report_call not in s:
    raise RuntimeError(f'GARCHAND tone _report not found')
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

# ── GARCHAND art (Bug 3+5+6) ─────────────────────────────────────────────────
c = get_cell(nb, GARCHAND_ART_ID)
s = src(c)
s = s.replace("artg = df['art_growth']", "artg = df['art_growth_winsor']")
# Bug 5: gamma bound (1e-8, None) -> (None, None)
# Pattern: (1e-8, None), (0.0, 0.999), (0.0, 0.999), (1e-8, None), (2.05, 100.0)
s = s.replace(
    "(1e-8, None), (0.0, 0.999), (0.0, 0.999), (1e-8, None), (2.05, 100.0)",
    "(1e-8, None), (0.0, 0.999), (0.0, 0.999), (None, None), (2.05, 100.0)"
)
report_call = "_report(PARAM_NAMES, theta, se, ll, T_obs,"
if report_call not in s:
    raise RuntimeError(f'GARCHAND art _report not found')
s = s.replace(report_call, BOUND_CHECK + report_call, 1)
set_source(c, s)
print('GARCHAND art (Bug 3+5+6): OK')

# ── GARCHND (Bug 1+3+6) ──────────────────────────────────────────────────────
c = get_cell(nb, GARCHND_ID)
s = src(c)
# Bug 1: remove .shift(1) double-lag
s = s.replace("tone_lag = df['tone_mean'].shift(1)", "tone_lag = df['tone_mean_x100_winsor']")
s = s.replace("artg_lag = df['art_growth'].shift(1)", "artg_lag = df['art_growth_winsor']")
# Bug 6: boundary check inside fit_garchnd
GARCHND_BOUND = (
    "    # Bug-6: boundary-solution check\n"
    "    on_bound = []\n"
    "    for _nm, _v, (_lo, _hi) in zip(PARAM_NAMES, theta, bounds):\n"
    "        if _lo is not None and abs(_v - _lo) < 1e-6: on_bound.append((_nm, _v, 'lower', _lo))\n"
    "        if _hi is not None and abs(_v - _hi) < 1e-6: on_bound.append((_nm, _v, 'upper', _hi))\n"
    "    if on_bound:\n"
    "        print('WARNING: boundary solution -- SE/t/p not interpretable for:', on_bound)\n"
    "    print()\n"
)
old_trailing = (
    "    _report(PARAM_NAMES, theta, se, ll, T_obs,\n"
    "            persist_alpha_beta=theta[1] + theta[2],\n"
    "            extra_notes=extra)\n"
    "    print()\n"
)
new_trailing = (
    "    _report(PARAM_NAMES, theta, se, ll, T_obs,\n"
    "            persist_alpha_beta=theta[1] + theta[2],\n"
    "            extra_notes=extra)\n"
    + GARCHND_BOUND
)
if old_trailing not in s:
    raise RuntimeError('GARCHND fit_garchnd trailing pattern not found')
s = s.replace(old_trailing, new_trailing)
set_source(c, s)
print('GARCHND (Bug 1+3+6): OK')

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f'\n{NB_PATH} patched successfully.')
