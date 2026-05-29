"""
Fix remove_polarity_outliers for pandas 3.0+.
In pandas 3.0, groupby.apply no longer passes the groupby column to the function.
Replace with explicit iteration which preserves all columns.
"""
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

NB_PATH = '01-data-cleaning.ipynb'
CELL_ID = 'eec6bae2'

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

cell = next((c for c in nb['cells'] if c.get('id','') == CELL_ID), None)
if cell is None:
    raise RuntimeError(f'Cell {CELL_ID} not found')

s = ''.join(cell['source'])

OLD = (
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

NEW = (
    "    # pandas 3.0+: groupby.apply no longer passes the groupby column to the\n"
    "    # function, so use explicit iteration which preserves all columns.\n"
    "    groups = [_filter(grp) for _, grp in df.groupby(date_col)]\n"
    "    if groups:\n"
    "        result = pd.concat(groups, ignore_index=True)\n"
    "    else:\n"
    "        result = df.iloc[0:0].copy()\n"
    "    return result"
)

if OLD not in s:
    raise RuntimeError('Pattern not found')

new_s = s.replace(OLD, NEW)
lines = new_s.splitlines()
cell['source'] = [line + '\n' for line in lines]
if cell['source']:
    cell['source'][-1] = cell['source'][-1].rstrip('\n')

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('remove_polarity_outliers: pandas 3.0 fix applied.')
