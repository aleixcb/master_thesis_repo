"""Fix all df_without_outlier['Date'] references in cell 31."""
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

# Insert _wo assignment right after the for-loop declaration and before_day line
# and change all df_without_outlier['Date'] references to _wo
OLD_BLOCK = (
    "for row, d in enumerate(days):\n"
    "    before_day = sent_dedup[sent_dedup['Date'] == d]\n"
    "    after_urls = set(df_without_outlier.loc[df_without_outlier['Date'] == d, 'url'])"
)
NEW_BLOCK = (
    "# Ensure Date is a column, not index (pandas version compat.)\n"
    "_df_wo = (df_without_outlier.reset_index()\n"
    "          if 'Date' not in df_without_outlier.columns else df_without_outlier)\n"
    "\n"
    "for row, d in enumerate(days):\n"
    "    before_day = sent_dedup[sent_dedup['Date'] == d]\n"
    "    after_urls = set(_df_wo.loc[_df_wo['Date'] == d, 'url'])"
)

OLD2 = "    y_after    = df_without_outlier.loc[df_without_outlier['Date'] == d, 'polarity'].values"
NEW2 = "    y_after    = _df_wo.loc[_df_wo['Date'] == d, 'polarity'].values"

if OLD_BLOCK not in s:
    print('ERROR: OLD_BLOCK not found')
elif OLD2 not in s:
    print('ERROR: OLD2 not found')
else:
    new_s = s.replace(OLD_BLOCK, NEW_BLOCK)
    new_s = new_s.replace(OLD2, NEW2)
    lines = new_s.splitlines()
    cell['source'] = [line + '\n' for line in lines]
    if cell['source']:
        cell['source'][-1] = cell['source'][-1].rstrip('\n')
    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print('Cell 31: all df_without_outlier Date references fixed.')
