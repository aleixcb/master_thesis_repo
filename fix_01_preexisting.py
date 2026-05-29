"""
Fix pre-existing bug in 01-data-cleaning.ipynb:
remove_polarity_outliers() can leave 'Date' as the index instead of a column
in some pandas versions, causing KeyError in the visualization cell below it.
"""
import json

NB_PATH = '01-data-cleaning.ipynb'

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

CELL_ID = 'eec6bae2'  # the remove_polarity_outliers function cell

cell = next((c for c in nb['cells'] if c.get('id','') == CELL_ID), None)
if cell is None:
    raise RuntimeError(f'Cell {CELL_ID} not found')

s = ''.join(cell['source'])

OLD = (
    "    return (\n"
    "        df.groupby(date_col, group_keys=False)\n"
    "          .apply(_filter)\n"
    "          .reset_index(drop=True)\n"
    "    )"
)

NEW = (
    "    result = (\n"
    "        df.groupby(date_col, group_keys=False)\n"
    "          .apply(_filter)\n"
    "    )\n"
    "    # Pandas version compatibility: keep date_col as a column, not the index\n"
    "    if result.index.name == date_col:\n"
    "        result = result.reset_index()\n"
    "    else:\n"
    "        result = result.reset_index(drop=True)\n"
    "    return result"
)

if OLD not in s:
    raise RuntimeError('Expected pattern not found in remove_polarity_outliers')

new_s = s.replace(OLD, NEW)

lines = new_s.splitlines()
cell['source'] = [line + '\n' for line in lines]
if cell['source']:
    cell['source'][-1] = cell['source'][-1].rstrip('\n')

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Pre-existing bug fixed: remove_polarity_outliers now robust to pandas version.')
