"""
Fix initial gamma values for tone-based GARCH models after tone_mean x100 rescaling.
tone_mean_x100_winsor is 100x larger than tone_mean, so x^2 is 10000x larger.
gamma_init must decrease by 10000x to give gamma*x^2 the same order as variance.
"""
import json

def fix_nb(path):
    with open(path, encoding='utf-8') as f:
        nb = json.load(f)

    changed = []
    for c in nb['cells']:
        src = ''.join(c.get('source', []))
        new_src = src

        # GARCH-X tone: x0 gamma 0.001 -> 1e-7
        # Identify by context: GARCH-X tone cell has both tone and 4 params
        # The GARCH-X tone x0 line is: x0 = np.array([0.05, 0.10, 0.85, 0.001])
        # The GARCH-X art x0 line is:  x0 = np.array([0.05, 0.10, 0.85, 0.01])
        # GARCHAND tone: x0 = np.array([0.05, 0.10, 0.85, 0.001, 0.5]) (Gaussian)
        #                x0 = np.array([0.05, 0.10, 0.85, 0.001, 0.5, 8.0]) (Student-t)

        # Only fix GARCH-X tone (4-param, gamma at index 3 = 0.001)
        if ("tone = df['tone_mean_x100_winsor']" in src and
            'x0     = np.array([0.05, 0.10, 0.85, 0.001])' in src):
            new_src = new_src.replace(
                'x0     = np.array([0.05, 0.10, 0.85, 0.001])',
                'x0     = np.array([0.05, 0.10, 0.85, 1e-7])'
            )
            changed.append('GARCH-X tone: gamma x0 fixed')

        # GARCHAND tone (Gaussian, 5 params): gamma at index 3
        if ("tone = df['tone_mean_x100_winsor']" in src and
            'x0     = np.array([0.05, 0.10, 0.85, 0.001, 0.5])' in src):
            new_src = new_src.replace(
                'x0     = np.array([0.05, 0.10, 0.85, 0.001, 0.5])',
                'x0     = np.array([0.05, 0.10, 0.85, 1e-7, 0.5])'
            )
            changed.append('GARCHAND tone (Gauss): gamma x0 fixed')

        # GARCHAND tone (Student-t, 6 params): gamma at index 3
        if ("tone = df['tone_mean_x100_winsor']" in src and
            'x0     = np.array([0.05, 0.10, 0.85, 0.001, 0.5, 8.0])' in src):
            new_src = new_src.replace(
                'x0     = np.array([0.05, 0.10, 0.85, 0.001, 0.5, 8.0])',
                'x0     = np.array([0.05, 0.10, 0.85, 1e-7, 0.5, 8.0])'
            )
            changed.append('GARCHAND tone (Student): gamma x0 fixed')

        # GARCHND cell: uses fit_garchnd with x0 = [0.086, 0.089, 0.898, 0.01]
        # The tone variant needs gamma scaled; art variant uses the same cell
        # Fix: use different gamma_x0 per call by adding a gamma_x0 parameter
        # to fit_garchnd and passing 1e-7 for tone calls
        if ('tone_lag = df[\'tone_mean_x100_winsor\']' in src and
            'fit_garchnd' in src):
            # Fix fit_garchnd function signature to accept gamma_x0
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
                "    x0     = np.array([0.086, 0.089, 0.898, 0.01])\n"
                "    bounds = [(1e-8, None), (0.0, 0.999), (0.0, 0.999), (None, None)]"
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
                "    x0     = np.array([0.086, 0.089, 0.898, gamma_x0])\n"
                "    bounds = [(1e-8, None), (0.0, 0.999), (0.0, 0.999), (None, None)]"
            )
            if old_fn in new_src:
                new_src = new_src.replace(old_fn, new_fn)
                # Update the tone calls to pass gamma_x0=1e-7
                new_src = new_src.replace(
                    "fit_garchnd('GARCHND  x = Tone           kappa = 30% annual', tone_lag, KAPPA_LOW)",
                    "fit_garchnd('GARCHND  x = Tone           kappa = 30% annual', tone_lag, KAPPA_LOW,  gamma_x0=1e-7)"
                )
                new_src = new_src.replace(
                    "fit_garchnd('GARCHND  x = Tone           kappa = 50% annual', tone_lag, KAPPA_HIGH)",
                    "fit_garchnd('GARCHND  x = Tone           kappa = 50% annual', tone_lag, KAPPA_HIGH, gamma_x0=1e-7)"
                )
                changed.append('GARCHND: fit_garchnd accepts gamma_x0, tone calls use 1e-7')

        if new_src != src:
            lines = new_src.splitlines()
            c['source'] = [line + '\n' for line in lines]
            if c['source']:
                c['source'][-1] = c['source'][-1].rstrip('\n')

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f'{path}: {changed if changed else "NO CHANGES (check cell content)"}')

fix_nb('03-GP-models-gaussian.ipynb')
fix_nb('03-GP-models-student.ipynb')
