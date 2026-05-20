"""Generate the ARMA + GARCH validation notebook.

Run once with anaconda's Python:
    /Applications/anaconda3/bin/python3 scripts/_build_arma_garch_notebook.py
"""
import json
import os

NOTEBOOK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ARMA_GARCH_validation_notebook.ipynb",
)


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = []

# ============================================================
# Header
# ============================================================
cells.append(md(
"""# ARMA Identification & GARCH(1,1) Volatility Validation

Pipeline applied to each of the four rare-earth metals (Nd, Pr, Dy, Tb):

1. **Identification** — ACF / PACF inspection on log returns to choose an ARMA(p,q) order for the conditional mean.
2. **Estimation** — fit by **OLS** (AR component, equation-by-equation) and **MLE** (full ARMA via state-space).
3. **Residual validation**
   - Homoskedasticity: residual & sqrt-|residual| plots, Breusch-Pagan test
   - Normality: histogram, QQ-plot, Shapiro-Wilk, Jarque-Bera
   - **Independence (most important)**: ACF/PACF of residuals, Ljung-Box test
4. **Realized variance** — computed from squared daily returns (no intraday data → no Parkinson/Garman-Klass available), plus 5- and 22-day rolling RV proxies.
5. **GARCH(1,1)** — fitted on returns; coefficient t-tests, stationarity check (α + β < 1), Mincer-Zarnowitz regression of RV on the GARCH variance forecast, and standardised-residual diagnostics.

Heteroskedasticity in the mean residuals is *expected* and is addressed by the subsequent GARCH step. Non-normality is acceptable for GARCH(1,1) under quasi-MLE; if severe, EGARCH/TARCH with Student-t innovations are the natural extension.

**Data**: `CLEANED DATA/RRE_prices_sentiment_combined.xlsx`
"""))

# ============================================================
# Imports
# ============================================================
cells.append(md("## 0. Imports"))

cells.append(code(
"""import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera

from arch import arch_model

warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.6f}'.format)
pd.set_option('display.max_columns', 30)
np.random.seed(42)
%matplotlib inline

print('Libraries loaded.')
"""))

# ============================================================
# Data loading
# ============================================================
cells.append(md(
"""## 1. Load Data

Log returns are scaled by 100 (percentage returns) — standard practice to improve numerical stability of MLE optimisers.
"""))

cells.append(code(
"""DATA_PATH = 'CLEANED DATA/RRE_prices_sentiment_combined.xlsx'
combined = pd.read_excel(DATA_PATH, parse_dates=['Date'], index_col='Date')

METALS = {
    'Nd': 'Neodymium',
    'Pr': 'Praseodymium',
    'Dy': 'Dysprosium',
    'Tb': 'Terbium',
}

returns = pd.DataFrame({
    code: combined[f'{code}_logret'].dropna() * 100.0
    for code in METALS
})
returns = returns.dropna(how='any')

print(f'Date range : {returns.index.min().date()} -> {returns.index.max().date()}')
print(f'Obs / metal: {len(returns)}')
returns.head()
"""))

# ============================================================
# Descriptive
# ============================================================
cells.append(md("## 2. Exploratory Analysis"))

cells.append(code(
"""fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
for ax, code in zip(axes, METALS):
    axes_ = ax
    axes_.plot(returns.index, returns[code], lw=0.6, color='steelblue')
    axes_.axhline(0, color='black', lw=0.5)
    axes_.set_title(f'{METALS[code]} ({code}) — daily log returns (%)')
    axes_.set_ylabel('r_t (%)')
axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.show()
"""))

cells.append(code(
"""def descriptive(s: pd.Series) -> dict:
    jb_stat, jb_p, skew, kurt = jarque_bera(s.values)
    return {
        'n': len(s),
        'mean': s.mean(),
        'std': s.std(),
        'min': s.min(),
        'max': s.max(),
        'skew': skew,
        'excess_kurt': kurt - 3.0,
        'JB stat': jb_stat,
        'JB p-val': jb_p,
    }

desc = pd.DataFrame({c: descriptive(returns[c]) for c in METALS}).T
desc
"""))

cells.append(md(
"""### Stationarity tests on returns

ADF (H0: unit root) and KPSS (H0: stationary) are run jointly — confident stationarity requires ADF p < 0.05 *and* KPSS p > 0.05.
"""))

cells.append(code(
"""def stationarity(s: pd.Series) -> dict:
    adf_stat, adf_p, *_ = adfuller(s.dropna(), autolag='AIC')
    kpss_stat, kpss_p, *_ = kpss(s.dropna(), regression='c', nlags='auto')
    return {
        'ADF stat': adf_stat, 'ADF p': adf_p,
        'KPSS stat': kpss_stat, 'KPSS p': kpss_p,
    }

stat_tbl = pd.DataFrame({c: stationarity(returns[c]) for c in METALS}).T
stat_tbl
"""))

# ============================================================
# Step 1 — ACF/PACF identification
# ============================================================
cells.append(md(
"""## 3. Step 1 — ARMA Identification via ACF / PACF

Reading rules:
- **AR(p)**: PACF cuts off at lag p, ACF tails off.
- **MA(q)**: ACF cuts off at lag q, PACF tails off.
- **ARMA(p,q)**: both tail off; in practice we use AIC/BIC across a small grid as a tie-breaker.

Bartlett ±1.96/√T bands flag significant autocorrelations.
"""))

cells.append(code(
"""MAX_LAGS = 30

fig, axes = plt.subplots(4, 2, figsize=(14, 12))
for i, code in enumerate(METALS):
    plot_acf(returns[code], lags=MAX_LAGS, ax=axes[i, 0])
    axes[i, 0].set_title(f'{code} — ACF')
    plot_pacf(returns[code], lags=MAX_LAGS, ax=axes[i, 1], method='ywm')
    axes[i, 1].set_title(f'{code} — PACF')
plt.tight_layout()
plt.show()
"""))

cells.append(md(
"""### Automatic order selection (AIC + BIC grid)

We complement the visual reading with a small ARMA(p,q) grid for p, q ∈ {0, 1, 2, 3} and pick the order minimising AIC/BIC. The grid result is a guide; the visual reading remains the primary justification.
"""))

cells.append(code(
"""def grid_arma(s, p_max=3, q_max=3):
    rows = []
    for p in range(p_max + 1):
        for q in range(q_max + 1):
            if p == 0 and q == 0:
                continue
            try:
                res = ARIMA(s, order=(p, 0, q),
                            trend='c',
                            enforce_stationarity=True,
                            enforce_invertibility=True).fit(method_kwargs={'warn_convergence': False})
                rows.append({'p': p, 'q': q, 'AIC': res.aic, 'BIC': res.bic, 'loglik': res.llf})
            except Exception as e:
                rows.append({'p': p, 'q': q, 'AIC': np.nan, 'BIC': np.nan, 'loglik': np.nan})
    return pd.DataFrame(rows).sort_values('BIC').reset_index(drop=True)

selection = {}
for code in METALS:
    g = grid_arma(returns[code])
    best = g.iloc[0]
    selection[code] = (int(best['p']), int(best['q']))
    print(f'{code} — best ARMA(p,q) by BIC: ({int(best[\"p\"])}, {int(best[\"q\"])})  '
          f'AIC={best[\"AIC\"]:.2f}  BIC={best[\"BIC\"]:.2f}')

print()
print('Selected orders:', selection)
"""))

# ============================================================
# Step 2 — OLS & MLE estimation
# ============================================================
cells.append(md(
"""## 4. Step 2 — Estimation by OLS and MLE

- **OLS**: fit the **AR(p) component** by ordinary least squares on lagged returns. For a pure AR model with Gaussian innovations, OLS and conditional MLE coincide asymptotically.
- **MLE**: fit the full **ARMA(p, q)** by maximum likelihood via state-space (statsmodels `ARIMA`). This is required whenever q > 0 since pure OLS cannot recover MA parameters.

If the BIC-selected order is ARMA(0,0) (white noise), we still fit AR(1) by OLS for comparison.
"""))

cells.append(code(
"""def ols_ar(s: pd.Series, p: int) -> dict:
    p_eff = max(p, 1)
    y = s.iloc[p_eff:].values
    X = np.column_stack([s.shift(k).iloc[p_eff:].values for k in range(1, p_eff + 1)])
    X = add_constant(X)
    res = OLS(y, X).fit()
    names = ['const'] + [f'ar.L{k}' for k in range(1, p_eff + 1)]
    return {
        'p': p_eff,
        'params': pd.Series(res.params, index=names),
        'tvalues': pd.Series(res.tvalues, index=names),
        'pvalues': pd.Series(res.pvalues, index=names),
        'sigma2': res.mse_resid,
        'resid': pd.Series(res.resid, index=s.index[p_eff:]),
        'r2': res.rsquared,
        'aic': res.aic,
        'bic': res.bic,
    }


def mle_arma(s: pd.Series, p: int, q: int) -> dict:
    res = ARIMA(s, order=(p, 0, q), trend='c',
                enforce_stationarity=True, enforce_invertibility=True).fit()
    return {
        'order': (p, q),
        'params': res.params,
        'tvalues': res.tvalues,
        'pvalues': res.pvalues,
        'sigma2': res.params.get('sigma2', np.nan),
        'resid': res.resid,
        'aic': res.aic,
        'bic': res.bic,
        'loglik': res.llf,
        'fitted': res,
    }


ols_fits = {}
mle_fits = {}
for code in METALS:
    p_sel, q_sel = selection[code]
    ols_fits[code] = ols_ar(returns[code], p_sel)
    mle_fits[code] = mle_arma(returns[code], p_sel, q_sel)
    print(f'\\n--- {code} | ARMA({p_sel},{q_sel}) ---')
    print('OLS (AR component only):')
    print(pd.DataFrame({
        'coef': ols_fits[code]['params'],
        't':    ols_fits[code]['tvalues'],
        'p':    ols_fits[code]['pvalues'],
    }))
    print('MLE (full ARMA):')
    print(pd.DataFrame({
        'coef': mle_fits[code]['params'],
        't':    mle_fits[code]['tvalues'],
        'p':    mle_fits[code]['pvalues'],
    }))
"""))

cells.append(code(
"""summary = pd.DataFrame({
    code: {
        'ARMA order': selection[code],
        'OLS AIC': ols_fits[code]['aic'],
        'OLS BIC': ols_fits[code]['bic'],
        'MLE AIC': mle_fits[code]['aic'],
        'MLE BIC': mle_fits[code]['bic'],
        'MLE loglik': mle_fits[code]['loglik'],
    }
    for code in METALS
}).T
summary
"""))

# ============================================================
# Step 3 — Residual validation
# ============================================================
cells.append(md(
"""## 5. Step 3 — Residual Validation (MLE residuals)

All diagnostics below are run on the **MLE residuals** (the OLS residuals are very similar when the ARMA order has q=0; we report MLE because it is the full-information estimator).
"""))

cells.append(code(
"""residuals = {code: mle_fits[code]['resid'].dropna() for code in METALS}
{c: len(r) for c, r in residuals.items()}
"""))

cells.append(md(
"""### 5.1 Homoskedasticity

- **Residual vs time plot** — should look like noise with constant spread; volatility clusters reveal ARCH effects (expected; motivates the GARCH step).
- **√|residual| plot** — clearer visualisation of changing scale.
- **Breusch-Pagan test** — H0: homoskedastic; small p-value rejects.
"""))

cells.append(code(
"""fig, axes = plt.subplots(4, 2, figsize=(14, 11))
for i, code in enumerate(METALS):
    r = residuals[code]
    axes[i, 0].plot(r.index, r.values, lw=0.6, color='steelblue')
    axes[i, 0].axhline(0, color='black', lw=0.5)
    axes[i, 0].set_title(f'{code} — residuals')
    axes[i, 0].set_ylabel('e_t')
    axes[i, 1].plot(r.index, np.sqrt(np.abs(r.values)), lw=0.6, color='darkorange')
    axes[i, 1].set_title(f'{code} — sqrt(|residuals|)')
    axes[i, 1].set_ylabel('sqrt|e_t|')
plt.tight_layout()
plt.show()
"""))

cells.append(code(
"""def breusch_pagan(resid: pd.Series, returns_series: pd.Series) -> dict:
    aligned = pd.concat([resid.rename('e'), returns_series.shift(1).rename('lag_r')], axis=1).dropna()
    lm, lm_p, f, f_p = het_breuschpagan(aligned['e'], add_constant(aligned[['lag_r']]))
    return {'BP LM': lm, 'BP LM p': lm_p, 'BP F': f, 'BP F p': f_p}

bp = pd.DataFrame({code: breusch_pagan(residuals[code], returns[code]) for code in METALS}).T
bp
"""))

cells.append(md(
"""### 5.2 Normality

- Histogram with overlaid normal density
- QQ plot vs Normal
- **Shapiro-Wilk** (small p → reject normality; sensitive in large samples)
- **Jarque-Bera** (skewness + excess kurtosis joint test)

Severe non-normality is expected for daily financial returns. It does *not* invalidate the GARCH fit under quasi-MLE — coefficient estimates remain consistent; only the standard errors need a robust adjustment (Bollerslev-Wooldridge).
"""))

cells.append(code(
"""fig, axes = plt.subplots(4, 2, figsize=(13, 12))
for i, code in enumerate(METALS):
    r = residuals[code].values
    axes[i, 0].hist(r, bins=80, density=True, color='steelblue', alpha=0.7)
    xs = np.linspace(r.min(), r.max(), 200)
    axes[i, 0].plot(xs, stats.norm.pdf(xs, r.mean(), r.std()), 'r-', lw=1.5, label='Normal')
    axes[i, 0].set_title(f'{code} — residual histogram')
    axes[i, 0].legend()
    stats.probplot(r, dist='norm', plot=axes[i, 1])
    axes[i, 1].set_title(f'{code} — QQ plot')
plt.tight_layout()
plt.show()
"""))

cells.append(code(
"""def normality_tests(r: np.ndarray) -> dict:
    sw_stat, sw_p = stats.shapiro(r[:5000] if len(r) > 5000 else r)  # SW limit ~5000
    jb_stat, jb_p, skew, kurt = jarque_bera(r)
    return {
        'Shapiro stat': sw_stat, 'Shapiro p': sw_p,
        'JB stat': jb_stat, 'JB p': jb_p,
        'skew': skew, 'excess_kurt': kurt - 3.0,
    }

norm_tbl = pd.DataFrame({code: normality_tests(residuals[code].values) for code in METALS}).T
norm_tbl
"""))

cells.append(md(
"""### 5.3 Independence — the most important diagnostic

If the ARMA mean equation is correctly specified, residuals must be **serially uncorrelated**. Remaining autocorrelation signals an under-specified mean — the GARCH step that follows assumes a white-noise mean residual.

- **ACF / PACF of residuals**: ideally inside Bartlett bands.
- **Ljung-Box** (H0: no autocorrelation up to lag h): tested at h ∈ {5, 10, 20}. Degrees-of-freedom adjusted by the number of fitted ARMA parameters (p + q).
"""))

cells.append(code(
"""fig, axes = plt.subplots(4, 2, figsize=(14, 12))
for i, code in enumerate(METALS):
    plot_acf(residuals[code], lags=MAX_LAGS, ax=axes[i, 0])
    axes[i, 0].set_title(f'{code} — residual ACF')
    plot_pacf(residuals[code], lags=MAX_LAGS, ax=axes[i, 1], method='ywm')
    axes[i, 1].set_title(f'{code} — residual PACF')
plt.tight_layout()
plt.show()
"""))

cells.append(code(
"""def ljung_box(resid: pd.Series, p: int, q: int, lags=(5, 10, 20)) -> pd.DataFrame:
    dof = p + q  # subtract number of fitted ARMA params
    rows = []
    for h in lags:
        lb = acorr_ljungbox(resid, lags=[h], model_df=dof, return_df=True)
        rows.append({'lag': h, 'LB stat': lb['lb_stat'].iloc[0], 'LB p': lb['lb_pvalue'].iloc[0]})
    return pd.DataFrame(rows).set_index('lag')

print('Ljung-Box on ARMA residuals (H0: no serial correlation):')
for code in METALS:
    p_sel, q_sel = selection[code]
    print(f'\\n--- {code} | ARMA({p_sel},{q_sel}) ---')
    print(ljung_box(residuals[code], p_sel, q_sel))
"""))

cells.append(md(
"""### 5.4 ARCH effects in residuals (motivates GARCH)

A Ljung-Box on **squared** residuals tests for remaining heteroskedasticity / ARCH effects. Strong rejection here is the empirical justification for moving to a GARCH variance equation.
"""))

cells.append(code(
"""print('Ljung-Box on SQUARED residuals (H0: no ARCH effects):')
for code in METALS:
    r2 = residuals[code] ** 2
    lb = acorr_ljungbox(r2, lags=[5, 10, 20], return_df=True)
    print(f'\\n--- {code} ---')
    print(lb)
"""))

# ============================================================
# Step 4 — Realized variance
# ============================================================
cells.append(md(
"""## 6. Step 4 — Realized Variance

Because rare-earth prices are OTC daily closes — no intraday data and no OHLC — the only feasible variance proxies are:

| Proxy | Formula | Use |
|---|---|---|
| Squared returns | RV1_t = r_t² | Unbiased but very noisy; 1-day evaluation |
| Rolling RV (5d) | RV5_t = Σ_{i=t-4..t} r_i² | Smoothed weekly proxy |
| Rolling RV (22d) | RV22_t = Σ_{i=t-21..t} r_i² | Smoothed monthly proxy (standard when intraday unavailable) |

Squared returns are the proxy used for the 1-day-ahead Mincer-Zarnowitz regression. The 5- and 22-day rolling versions are shown only for visualisation — they cannot be used as a true 1-day proxy because they overlap with the evaluation window.
"""))

cells.append(code(
"""rv = pd.DataFrame(index=returns.index)
for code in METALS:
    r = returns[code]
    rv[f'{code}_RV1']  = r ** 2
    rv[f'{code}_RV5']  = (r ** 2).rolling(5).sum()
    rv[f'{code}_RV22'] = (r ** 2).rolling(22).sum()

print('Realized variance — summary (percent² units):')
rv.describe().T[['mean', 'std', 'min', '50%', 'max']]
"""))

cells.append(code(
"""fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
for ax, code in zip(axes, METALS):
    ax.plot(rv.index, rv[f'{code}_RV22'], lw=0.8, color='steelblue', label='22-day rolling RV')
    ax.plot(rv.index, rv[f'{code}_RV1'], lw=0.4, color='tomato', alpha=0.4, label='r_t² (RV1)')
    ax.set_title(f'{code} — realized variance')
    ax.set_ylabel('variance (%²)')
    ax.legend(loc='upper right')
axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.show()
"""))

# ============================================================
# Step 5 — GARCH(1,1)
# ============================================================
cells.append(md(
"""## 7. Step 5 — GARCH(1,1)

Model:
$$ r_t = \\mu + \\varepsilon_t, \\qquad \\varepsilon_t = \\sigma_t z_t, \\qquad z_t \\sim \\mathcal{N}(0, 1) $$
$$ \\sigma_t^2 = \\omega + \\alpha\\, \\varepsilon_{t-1}^2 + \\beta\\, \\sigma_{t-1}^2 $$

Stationarity requires α + β < 1; positivity requires ω > 0, α ≥ 0, β ≥ 0.

Fitted with normal innovations (quasi-MLE — robust standard errors hold even if returns are non-normal).
"""))

cells.append(code(
"""garch_fits = {}
for code in METALS:
    am = arch_model(returns[code], mean='Constant', vol='GARCH', p=1, q=1, dist='normal')
    res = am.fit(disp='off')
    garch_fits[code] = res
    print(f'\\n========== {code} — GARCH(1,1) ==========')
    print(res.summary())
"""))

cells.append(md(
"""### 7.1 Coefficient significance & stationarity

We extract t-statistics, p-values, and the persistence α + β.
"""))

cells.append(code(
"""rows = []
for code in METALS:
    res = garch_fits[code]
    p = res.params
    pv = res.pvalues
    persistence = p['alpha[1]'] + p['beta[1]']
    half_life = np.log(0.5) / np.log(persistence) if persistence < 1 else np.inf
    rows.append({
        'metal': code,
        'mu': p['mu'], 'mu p': pv['mu'],
        'omega': p['omega'], 'omega p': pv['omega'],
        'alpha': p['alpha[1]'], 'alpha p': pv['alpha[1]'],
        'beta': p['beta[1]'], 'beta p': pv['beta[1]'],
        'alpha+beta': persistence,
        'half-life (days)': half_life,
        'loglik': res.loglikelihood,
        'AIC': res.aic, 'BIC': res.bic,
    })
sig_tbl = pd.DataFrame(rows).set_index('metal')
sig_tbl
"""))

cells.append(md(
"""### 7.2 Standardised-residual diagnostics

If the GARCH(1,1) is well specified:
- standardised residuals **z_t = ε_t / σ_t** should be serially uncorrelated → Ljung-Box on z_t
- squared standardised residuals **z_t²** should be free of ARCH effects → Ljung-Box on z_t²

Both p-values should be > 0.05.
"""))

cells.append(code(
"""print('Ljung-Box on standardised residuals and their squares (lag 10):')
diag_rows = []
for code in METALS:
    res = garch_fits[code]
    z = res.std_resid.dropna()
    lb_z  = acorr_ljungbox(z,    lags=[10], return_df=True).iloc[0]
    lb_z2 = acorr_ljungbox(z**2, lags=[10], return_df=True).iloc[0]
    diag_rows.append({
        'metal': code,
        'LB(z, 10) stat':  lb_z['lb_stat'],
        'LB(z, 10) p':     lb_z['lb_pvalue'],
        'LB(z², 10) stat': lb_z2['lb_stat'],
        'LB(z², 10) p':    lb_z2['lb_pvalue'],
    })
pd.DataFrame(diag_rows).set_index('metal')
"""))

cells.append(md(
"""### 7.3 GARCH conditional variance vs realized variance

Visual comparison: the GARCH conditional variance σ̂_t² should track the realized variance. We overlay σ̂_t² against the 22-day rolling RV (smoothed reference) and against r_t² (raw 1-day proxy).
"""))

cells.append(code(
"""fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
for ax, code in zip(axes, METALS):
    res = garch_fits[code]
    sigma2 = res.conditional_volatility ** 2
    ax.plot(rv.index, rv[f'{code}_RV1'], color='lightgrey', lw=0.4, label='r_t² (RV1)')
    ax.plot(rv.index, rv[f'{code}_RV22'], color='tomato', lw=0.9, label='RV (22-day)')
    ax.plot(sigma2.index, sigma2.values, color='steelblue', lw=0.9, label='GARCH σ̂_t²')
    ax.set_title(f'{code} — GARCH(1,1) conditional variance vs realized variance')
    ax.set_ylabel('variance (%²)')
    ax.legend(loc='upper right')
axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.show()
"""))

cells.append(md(
"""### 7.4 Mincer-Zarnowitz regression — does GARCH predict RV?

Regress the realized variance on the GARCH conditional variance:
$$ RV_t = a + b\\, \\hat\\sigma_t^2 + u_t $$

A well-calibrated forecast satisfies **a = 0 and b = 1**. We report individual t-tests for each coefficient and a Wald joint test of H0: (a, b) = (0, 1).
"""))

cells.append(code(
"""def mincer_zarnowitz(rv_series: pd.Series, sigma2_series: pd.Series) -> dict:
    aligned = pd.concat([rv_series.rename('rv'), sigma2_series.rename('s2')], axis=1).dropna()
    X = add_constant(aligned['s2'].values)
    res = OLS(aligned['rv'].values, X).fit()
    # Wald test of joint hypothesis a=0, b=1
    R = np.eye(2)
    q = np.array([0.0, 1.0])
    wald = res.wald_test((R, q), use_f=False)
    return {
        'a (intercept)':  res.params[0],
        'a t-stat':       res.tvalues[0],
        'a p-val':        res.pvalues[0],
        'b (slope)':      res.params[1],
        'b t-stat':       res.tvalues[1],
        'b p-val':        res.pvalues[1],
        'R²':             res.rsquared,
        'Wald chi² (a=0,b=1)': float(np.asarray(wald.statistic).squeeze()),
        'Wald p-val':     float(np.asarray(wald.pvalue).squeeze()),
        'n':              int(res.nobs),
    }


print('Mincer-Zarnowitz regressions (RV_t = a + b·σ̂_t²):')
mz_rows = {}
for code in METALS:
    sigma2 = garch_fits[code].conditional_volatility ** 2
    mz_rows[code] = mincer_zarnowitz(rv[f'{code}_RV1'], sigma2)

mz_tbl = pd.DataFrame(mz_rows).T
mz_tbl
"""))

cells.append(md(
"""### 7.5 Loss-function evaluation

Three loss functions compare σ̂_t² against r_t² (the unbiased 1-day RV proxy):

- **MSE**: penalises large errors quadratically — sensitive to outliers
- **MAE**: linear penalty, robust
- **QLIKE** = σ²/σ̂² − ln(σ²/σ̂²) − 1: *robust to proxy noise* in the Patton (2011) sense, so its ranking is reliable even with the very noisy r_t² proxy. **This is the preferred loss for daily-close volatility evaluation.**

Lower is better in all three.
"""))

cells.append(code(
"""def loss_metrics(rv_series: pd.Series, sigma2_series: pd.Series) -> dict:
    aligned = pd.concat([rv_series.rename('rv'), sigma2_series.rename('s2')], axis=1).dropna()
    aligned = aligned[(aligned['rv'] > 0) & (aligned['s2'] > 0)]
    err = aligned['rv'].values - aligned['s2'].values
    ratio = aligned['rv'].values / aligned['s2'].values
    return {
        'MSE':   float(np.mean(err ** 2)),
        'MAE':   float(np.mean(np.abs(err))),
        'QLIKE': float(np.mean(ratio - np.log(ratio) - 1.0)),
        'n':     int(len(aligned)),
    }

loss_rows = {}
for code in METALS:
    sigma2 = garch_fits[code].conditional_volatility ** 2
    loss_rows[code] = loss_metrics(rv[f'{code}_RV1'], sigma2)

loss_tbl = pd.DataFrame(loss_rows).T
loss_tbl
"""))

# ============================================================
# Conclusions
# ============================================================
cells.append(md(
"""## 8. Conclusions

This notebook covered the full identification → estimation → validation → GARCH chain. Read the result tables above per metal:

- **Identification**: ACF/PACF + BIC grid select the ARMA(p, q) order for the mean.
- **Estimation**: OLS (AR component) and full-information MLE largely agree when q = 0.
- **Validation**: independence of mean residuals is the load-bearing check — if Ljung-Box on residuals fails, revisit the ARMA order. Heteroskedasticity (Breusch-Pagan and ARCH-test rejections) is *expected* and motivates GARCH. Non-normality is also expected; QMLE inference remains valid.
- **GARCH(1,1)**: significant α and β with α + β < 1 confirms a well-specified, stationary variance process. Standardised-residual Ljung-Box (z and z²) p-values > 0.05 indicate the GARCH has absorbed the residual ARCH.
- **GARCH vs RV**: the Mincer-Zarnowitz slope close to 1, low QLIKE, and tight tracking in the σ̂_t² vs RV22 plot together demonstrate that GARCH(1,1) is producing reasonable variance forecasts at the daily horizon.

If any metal fails the Mincer-Zarnowitz joint test or shows residual ARCH in z², that is the empirical signal to escalate to EGARCH or TARCH (asymmetric variance dynamics) and/or Student-t innovations — exactly the path laid out in the methodology document.
"""))


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.9.6",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(NOTEBOOK_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Wrote {NOTEBOOK_PATH}")
print(f"  cells: {len(cells)}")
