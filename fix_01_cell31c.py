"""Fix remaining df_without_outlier['Date'] reference in cell 31, line 80."""
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

OLD = "    y_after    = df_without_outlier.loc[df_without_outlier['Date'] == d, 'polarity'].values"
NEW = "    y_after    = _wo.loc[_wo['Date'] == d, 'polarity'].values"

if OLD not in s:
    raise RuntimeError('Pattern not found')

new_s = s.replace(OLD, NEW)
lines = new_s.splitlines()
cell['source'] = [line + '\n' for line in lines]
if cell['source']:
    cell['source'][-1] = cell['source'][-1].rstrip('\n')

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('y_after line fixed.')
