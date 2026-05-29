"""Fix gamma x0 for tone-based models in 03-GP-models-student.ipynb."""
import json

NB = '03-GP-models-student.ipynb'
with open(NB, encoding='utf-8') as f:
    nb = json.load(f)

changed = []
for c in nb['cells']:
    src = ''.join(c.get('source', []))
    new_src = src

    # GARCH-X tone (Student-t): 5-element x0, gamma at index 3
    if ("tone = df['tone_mean_x100_winsor']" in src and
        'x0     = np.array([0.05, 0.10, 0.85, 0.001, 8.0])' in src):
        new_src = new_src.replace(
            'x0     = np.array([0.05, 0.10, 0.85, 0.001, 8.0])',
            'x0     = np.array([0.05, 0.10, 0.85, 1e-7, 8.0])'
        )
        changed.append('GARCH-X tone (Student): gamma x0 fixed')

    # GARCHND Student-t: 5-element x0 [omega, alpha, beta, gamma, nu]
    if ('tone_lag = df[\'tone_mean_x100_winsor\']' in src and
        'fit_garchnd' in src):
        old_fn = (
            "def fit_garchnd(label, x_series_lag, kappa):\n"
            "    r_pct   = df['r'] * 100.0\n"
            "    aligned = pd.concat([r_pct, x_series_lag], axis=1, keys=['r', 'x']).dropna()\n"
            "    r_arr = aligned['r'].values\n"
            "    x_arr = aligned['x'].values\n"
            "    T_obs = len(r_arr)\n"
            "    print(f'---- {label} ----')\n"
            "    print(f'Sample: T = {T_obs} obs '\n"
            "          f'({aligned.index.min().date()} -> {aligned.index.max().date()})  '\n"
            "          f'kappa = {kappa:.4f}')\n"
            "\n"
            "    x0     = np.array([0.086, 0.089, 0.898, 0.01, 8.0])\n"
            "    bounds = [(1e-8, None), (0.0, 0.999), (0.0, 0.999),\n"
            "              (None, None), (2.05, 100.0)]"
        )
        new_fn = (
            "def fit_garchnd(label, x_series_lag, kappa, gamma_x0=0.01):\n"
            "    r_pct   = df['r'] * 100.0\n"
            "    aligned = pd.concat([r_pct, x_series_lag], axis=1, keys=['r', 'x']).dropna()\n"
            "    r_arr = aligned['r'].values\n"
            "    x_arr = aligned['x'].values\n"
            "    T_obs = len(r_arr)\n"
            "    print(f'---- {label} ----')\n"
            "    print(f'Sample: T = {T_obs} obs '\n"
            "          f'({aligned.index.min().date()} -> {aligned.index.max().date()})  '\n"
            "          f'kappa = {kappa:.4f}')\n"
            "\n"
            "    x0     = np.array([0.086, 0.089, 0.898, gamma_x0, 8.0])\n"
            "    bounds = [(1e-8, None), (0.0, 0.999), (0.0, 0.999),\n"
            "              (None, None), (2.05, 100.0)]"
        )
        if old_fn in new_src:
            new_src = new_src.replace(old_fn, new_fn)
            new_src = new_src.replace(
                "fit_garchnd('GARCHND  x = Tone           kappa = 30% annual', tone_lag, KAPPA_LOW)",
                "fit_garchnd('GARCHND  x = Tone           kappa = 30% annual', tone_lag, KAPPA_LOW,  gamma_x0=1e-7)"
            )
            new_src = new_src.replace(
                "fit_garchnd('GARCHND  x = Tone           kappa = 50% annual', tone_lag, KAPPA_HIGH)",
                "fit_garchnd('GARCHND  x = Tone           kappa = 50% annual', tone_lag, KAPPA_HIGH, gamma_x0=1e-7)"
            )
            changed.append('GARCHND Student-t: fit_garchnd accepts gamma_x0, tone uses 1e-7')
        else:
            changed.append('WARNING: GARCHND Student pattern not found')

    if new_src != src:
        lines = new_src.splitlines()
        c['source'] = [line + '\n' for line in lines]
        if c['source']:
            c['source'][-1] = c['source'][-1].rstrip('\n')

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f'{NB}: {changed}')
