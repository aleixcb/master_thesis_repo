"""Patch 01-data-cleaning.ipynb -- Bug 3: add winsorization columns."""
import json

NB_PATH = '01-data-cleaning.ipynb'

WINSOR_CODE = (
    "\n"
    "# -- Bug-3 fix: rescale tone x100 and winsorize both series --\n"
    "# New columns only — raw tone_mean and art_growth are kept for 02/05.\n"
    "# Winsorization uses full-sample percentiles by default; pass an\n"
    "# in_sample_mask to switch to in-sample-only cutoffs after OOS split.\n"
    "\n"
    "def _winsorize(series, lo_pct, hi_pct, in_sample_mask=None):\n"
    "    ref = series[in_sample_mask] if in_sample_mask is not None else series\n"
    "    lo = np.nanpercentile(ref.dropna(), lo_pct * 100.0)\n"
    "    hi = np.nanpercentile(ref.dropna(), hi_pct * 100.0)\n"
    "    return series.clip(lo, hi), lo, hi\n"
    "\n"
    "combined['tone_mean_x100'] = combined['tone_mean'] * 100.0\n"
    "combined['tone_mean_x100_winsor'], lo_t, hi_t = _winsorize(\n"
    "    combined['tone_mean_x100'], 0.01, 0.99)\n"
    "combined['art_growth_winsor'], lo_a, hi_a = _winsorize(\n"
    "    combined['art_growth'], 0.01, 0.99)\n"
    "\n"
    "def _winsor_summary(name, raw, win, lo, hi):\n"
    "    n_clip = int(((raw < lo) | (raw > hi)).sum())\n"
    "    print(f'{name}:')\n"
    "    print(f'  Raw    : min={raw.min():.4f}  max={raw.max():.4f}  std={raw.std():.4f}')\n"
    "    print(f'  Winsor : min={win.min():.4f}  max={win.max():.4f}  std={win.std():.4f}')\n"
    "    print(f'  Cutoffs: [{lo:.4f}, {hi:.4f}]  rows clipped: {n_clip}')\n"
    "    print()\n"
    "\n"
    "print('=== Winsorization summary ===')\n"
    "_winsor_summary('tone_mean_x100',\n"
    "    combined['tone_mean_x100'], combined['tone_mean_x100_winsor'], lo_t, hi_t)\n"
    "_winsor_summary('art_growth',\n"
    "    combined['art_growth'], combined['art_growth_winsor'], lo_a, hi_a)\n"
    "\n"
)

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

target_id = 'aa4f4119'
cell = next((c for c in nb['cells'] if c.get('id','') == target_id), None)
if cell is None:
    raise RuntimeError(f'Cell {target_id} not found')

src = ''.join(cell['source'])

save_marker = '# --- 5. Save combined file ---'
if save_marker not in src:
    raise RuntimeError(f'Marker not found: {save_marker}')

new_src = src.replace(save_marker, WINSOR_CODE + save_marker)

cell['source'] = [line + '\n' for line in new_src.splitlines()]
if cell['source'] and cell['source'][-1].endswith('\n'):
    cell['source'][-1] = cell['source'][-1].rstrip('\n')

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('01-data-cleaning.ipynb patched (Bug 3 winsorization added).')
