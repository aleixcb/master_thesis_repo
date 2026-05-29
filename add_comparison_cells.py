"""Add old-vs-new comparison markdown cells at the bottom of model notebooks."""
import json

def get_last_code_cell_idx(nb):
    for i in range(len(nb['cells']) - 1, -1, -1):
        if nb['cells'][i]['cell_type'] == 'code':
            return i
    return len(nb['cells']) - 1

def append_markdown_cell(nb, content, cell_id):
    cell = {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": [line + '\n' for line in content.splitlines()]
    }
    if cell['source']:
        cell['source'][-1] = cell['source'][-1].rstrip('\n')
    nb['cells'].append(cell)

# ── 03-GP-models-gaussian.ipynb ──────────────────────────────────────────────
GAUSSIAN_COMPARISON = """## Old-vs-new: sign and significance of γ and θ (Gaussian innovations)

Reference (OLD): last report in `REPORTS/03-GP-models-gaussian.txt` (before any bug fixes).
Fixes applied: Bug 1 (GARCHND double-lag removed), Bug 3 (tone×100 + winsorization),
Bug 4 (zero-mean returns), Bug 5 (θ/γ bounds relaxed in GARCHAND).
NEW values: re-run this notebook top-to-bottom and read from the cells above.

| Model | Param | OLD sign | OLD signif. | NEW sign | NEW signif. | Notes |
|---|---|---|---|---|---|---|
| GARCH-X tone | γ | + | n.s. | TBD | TBD | Bug 3: tone×100 winsorized |
| GARCH-X art | γ | − | 5% | TBD | TBD | Bug 3: art_growth winsorized |
| GARCHAND tone | γ | − | n.s. | TBD | TBD | Bug 3+5: tone×100 winsorized; bounds unchanged |
| GARCHAND tone | θ | + | n.s. | TBD | TBD | Bug 5: bound relaxed to (−0.999, None); old run was boundary solution |
| GARCHAND art | γ | +0.00 | n.s. | TBD | TBD | Bug 5: bound relaxed to (None, None); old γ was pinned at 0 |
| GARCHND tone κ=30% | γ | + | n.s. | TBD | TBD | Bug 1: lag corrected; Bug 3: col ref |
| GARCHND tone κ=50% | γ | − | 1% | TBD | TBD | Bug 1: lag corrected; Bug 3: col ref |
| GARCHND art κ=30% | γ | + | n.s. | TBD | TBD | Bug 1: lag corrected; Bug 3: col ref |
| GARCHND art κ=50% | γ | − | n.s. | TBD | TBD | Bug 1: lag corrected; Bug 3: col ref |

**Interpretation note (deferred)**: Do not read into sign flips here — the point of this table is
to record what the data say after the implementation bugs are corrected, not to interpret results.
"""

# ── 03-GP-models-student.ipynb ───────────────────────────────────────────────
STUDENT_COMPARISON = """## Old-vs-new: sign and significance of γ and θ (Student-t innovations)

Reference (OLD): last report in `REPORTS/03-GP-models-student.txt` (before bug fixes).
Fixes applied: Bug 1 (GARCHND double-lag), Bug 3 (tone×100 + winsorization),
Bug 4 (zero-mean), Bug 5 (θ/γ bounds relaxed in GARCHAND).
NEW values: re-run this notebook and read from the cells above.

| Model | Param | OLD sign | OLD signif. | NEW sign | NEW signif. | Notes |
|---|---|---|---|---|---|---|
| GARCH-X tone | γ | − | n.s. | TBD | TBD | Bug 3: tone×100 winsorized |
| GARCH-X art | γ | − | 1% | TBD | TBD | Bug 3: art_growth winsorized |
| GARCHAND tone | γ | − | 10% | TBD | TBD | Bug 3+5: tone×100 winsorized |
| GARCHAND tone | θ | + | n.s. | TBD | TBD | Bug 5: bound relaxed to (−0.999, None) |
| GARCHAND art | γ | +0.00 | n.s. | TBD | TBD | Bug 5: bound relaxed to (None, None); old γ pinned at 0 |
| GARCHND tone κ=30% | γ | + | n.s. | TBD | TBD | Bug 1+3: lag + col ref fixed |
| GARCHND tone κ=50% | γ | − | 1% | TBD | TBD | Bug 1+3: lag + col ref fixed |
| GARCHND art κ=30% | γ | − | n.s. | TBD | TBD | Bug 1+3: lag + col ref fixed |
| GARCHND art κ=50% | γ | − | n.s. | TBD | TBD | Bug 1+3: lag + col ref fixed |

**Interpretation note (deferred)**: Record only; no interpretation here.
"""

# ── 04-extension-models.ipynb ────────────────────────────────────────────────
EXT_COMPARISON = """## Old-vs-new: sign and significance of γ₁ (tone) and γ₂ (art_growth) — EGARCH-X

Reference (OLD): `REPORTS/04-extension-models-gaussian.txt` and
`REPORTS/04-extension-models-student.txt` (before bug fixes).
Fixes applied: Bug 3 (tone×100 winsorized, art_growth winsorized), Bug 6 (boundary checks).
NEW values: re-run this notebook and read from the cells above.

| Dist | Param | OLD sign | OLD signif. | NEW sign | NEW signif. | Notes |
|---|---|---|---|---|---|---|
| Gaussian | γ_tone | + | n.s. | TBD | TBD | Bug 3: tone×100 winsorized |
| Gaussian | γ_artgrowth | − | 10% | TBD | TBD | Bug 3: art_growth winsorized |
| Student-t | γ_tone | + | 5% | TBD | TBD | Bug 3: tone×100 winsorized |
| Student-t | γ_artgrowth | − | 10% | TBD | TBD | Bug 3: art_growth winsorized |

OLD values are from the stale-global run (tone was unshifted and unscaled in both cells;
Bug 2 was not present in the latest code version on disk).
**Interpretation note (deferred)**: Record only; no interpretation here.
"""

# Apply to gaussian notebook
with open('03-GP-models-gaussian.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
append_markdown_cell(nb, GAUSSIAN_COMPARISON, 'oldvsnew_gaussian')
with open('03-GP-models-gaussian.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('03-GP-models-gaussian.ipynb: comparison cell added.')

# Apply to student notebook
with open('03-GP-models-student.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
append_markdown_cell(nb, STUDENT_COMPARISON, 'oldvsnew_student')
with open('03-GP-models-student.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('03-GP-models-student.ipynb: comparison cell added.')

# Apply to extension models notebook
with open('04-extension-models.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
append_markdown_cell(nb, EXT_COMPARISON, 'oldvsnew_ext')
with open('04-extension-models.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('04-extension-models.ipynb: comparison cell added.')
