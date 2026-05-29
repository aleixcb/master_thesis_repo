import json, sys

nb_path = sys.argv[1] if len(sys.argv) > 1 else '01-data-cleaning.ipynb'
keywords = sys.argv[2].split(',') if len(sys.argv) > 2 else ['tone_mean']

with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

out = []
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if any(k in src for k in keywords):
        cid = cell.get('id', '?')
        ctype = cell['cell_type']
        out.append(f'=== Cell {i} id={cid} type={ctype} ===')
        out.append(src[:5000])
        out.append('')

with open('parse_out.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('Written to parse_out.txt')
print(f'Total cells: {len(nb["cells"])}')
