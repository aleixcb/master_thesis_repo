"""Fix cell 31 (visualization) to handle Date-as-index or Date-as-column."""
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

NB_PATH = '01-data-cleaning.ipynb'
CELL_ID = 'e98927f8'

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

cell = next((c for c in nb['cells'] if c.get('id','') == CELL_ID), None)
if cell is None:
    raise RuntimeError(f'Cell {CELL_ID} not found')

s = ''.join(cell['source'])

OLD = "    after_urls = set(df_without_outlier.loc[df_without_outlier['Date'] == d, 'url'])"
NEW = (
    "    # Handle both Date-as-column and Date-as-index (pandas version compat.)\n"
    "    _wo = (df_without_outlier.reset_index()\n"
    "           if 'Date' not in df_without_outlier.columns else df_without_outlier)\n"
    "    after_urls = set(_wo.loc[_wo['Date'] == d, 'url'])"
)

found = OLD in s
with open('fix_diag.txt', 'w', encoding='utf-8') as diag:
    diag.write(f'OLD found: {found}\n')
    for i, line in enumerate(s.splitlines()):
        if 'after_urls' in line or 'df_without_outlier' in line:
            diag.write(f'  L{i}: {repr(line)}\n')

if not found:
    print('Pattern not found, see fix_diag.txt')
else:
    new_s = s.replace(OLD, NEW)
    lines = new_s.splitlines()
    cell['source'] = [line + '\n' for line in lines]
    if cell['source']:
        cell['source'][-1] = cell['source'][-1].rstrip('\n')
    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print('Cell 31 fixed.')
