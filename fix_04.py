"""
Patch 04-extension-models.ipynb
Bugs: 2 (ensure tone not tone_lag in student cell), 3 (column refs), 6 (boundary detection)
"""
import json

NB_PATH = '04-extension-models.ipynb'

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

GAUSSIAN_ID  = 'e696abc5'
STUDENT_ID   = 'f3bd14f1'

# ── Gaussian EGARCH-X (Bug 2+3+6) ────────────────────────────────────────────
c = get_cell(nb, GAUSSIAN_ID)
s = src(c)

# Bug 3: column refs
s = s.replace("tone       = df['tone_mean']", "tone       = df['tone_mean_x100_winsor']")
s = s.replace("art_growth = df['art_growth']", "art_growth = df['art_growth_winsor']")

# Bug 6: Gaussian cell has no explicit bounds; add a note + null check after theta= line
GAUSS_BOUND = (
    "\n# Bug-6: boundary-solution check (no explicit bounds in this cell, so nothing pinned)\n"
    "# If bounds are later added, insert the standard check here.\n"
)
# Insert after: theta = opt.x
s = s.replace(
    "theta = opt.x\nll    = -opt.fun\n",
    "theta = opt.x\nll    = -opt.fun\n" + GAUSS_BOUND
)
set_source(c, s)
print('Gaussian EGARCH (Bug 3+6): OK')

# ── Student-t EGARCH-X (Bug 2+3+6) ───────────────────────────────────────────
c = get_cell(nb, STUDENT_ID)
s = src(c)

# Bug 2: ensure tone is NOT tone_lag (check and fix if needed)
if 'tone_lag' in s:
    s = s.replace("tone_lag", "tone")
    print('Bug 2: tone_lag reference fixed')
else:
    print('Bug 2: no tone_lag reference found (already correct)')

# Bug 3: column refs
s = s.replace("tone       = df['tone_mean']", "tone       = df['tone_mean_x100_winsor']")
s = s.replace("art_growth = df['art_growth']", "art_growth = df['art_growth_winsor']")

# Bug 6: Student-t cell has explicit bounds, add check after theta= line
ST_BOUND = (
    "\n# Bug-6: boundary-solution check\n"
    "on_bound = []\n"
    "for _nm, _v, (_lo, _hi) in zip(PARAM_NAMES, theta, bounds):\n"
    "    if _lo is not None and abs(_v - _lo) < 1e-6: on_bound.append((_nm, _v, 'lower', _lo))\n"
    "    if _hi is not None and abs(_v - _hi) < 1e-6: on_bound.append((_nm, _v, 'upper', _hi))\n"
    "if on_bound:\n"
    "    print('WARNING: boundary solution -- SE/t/p not interpretable for:', on_bound)\n"
)
s = s.replace(
    "theta = opt.x\nll    = -opt.fun\n",
    "theta = opt.x\nll    = -opt.fun\n" + ST_BOUND
)
set_source(c, s)
print('Student-t EGARCH (Bug 2+3+6): OK')

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f'\n{NB_PATH} patched successfully.')
