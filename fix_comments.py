"""Fix stale comments after Bug 5 bound relaxation."""
import json

def fix_nb(path):
    with open(path, encoding='utf-8') as f:
        nb = json.load(f)
    for c in nb['cells']:
        src = ''.join(c.get('source', []))
        if '# gamma > 0' in src and 'GARCHAND' in src:
            new_src = src.replace(
                "(None, None)]  # gamma > 0",
                "(None, None)]  # Bug 5: gamma unrestricted (relaxed from 1e-8)"
            )
            new_src = new_src.replace(
                "# d2 = 1 if art_growth_{t-1} > 0, else 0;  gamma > 0.",
                "# d2 = 1 if art_growth_{t-1} > 0, else 0;  gamma unrestricted (Bug 5)."
            )
            lines = new_src.splitlines()
            c['source'] = [line + '\n' for line in lines]
            if c['source']:
                c['source'][-1] = c['source'][-1].rstrip('\n')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f'{path}: stale comments fixed.')

fix_nb('03-GP-models-gaussian.ipynb')
fix_nb('03-GP-models-student.ipynb')
