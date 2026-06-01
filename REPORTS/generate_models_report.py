"""
PDF report generator for the GARCH/EGARCH volatility models.

Reads the results captured in the REPORTS folder and builds a structured
PDF with one section per model. Each section contains the variance
expression, a short motivation, and the main results (Gaussian first,
Student-t second). A final part covers the six robustness checks from
notebooks 06 and 07.

Usage:
    cd REPORTS
    python generate_models_report.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPORTS_DIR = Path(__file__).resolve().parent
OUT_PDF = REPORTS_DIR / "models_report.pdf"

FILES = {
    "gp_gaussian":  REPORTS_DIR / "03-GP-models-gaussian.txt",
    "gp_student":   REPORTS_DIR / "03-GP-models-student.txt",
    "ext_gaussian": REPORTS_DIR / "04-extension-models-gaussian.txt",
    "ext_student":  REPORTS_DIR / "04-extension-models-student.txt",
}

# --------------------------------------------------------------------------- #
# Model metadata
# --------------------------------------------------------------------------- #
MODELS: List[Dict] = [
    {
        "title": "1. GARCH(1,1) — Baseline",
        "variance": (
            "&sigma;<sup>2</sup><sub>t</sub> = &omega; "
            "+ &alpha; &middot; r<sup>2</sup><sub>t-1</sub> "
            "+ &beta; &middot; &sigma;<sup>2</sup><sub>t-1</sub>"
        ),
        "explanation": (
            "Canonical Bollerslev (1986) specification. It serves as the "
            "benchmark against which every sentiment-augmented model is "
            "compared. The goal is to confirm persistence (&alpha;+&beta; "
            "close to 1) and that the basic parameters are significant; "
            "all later extensions are evaluated relative to this fit."
        ),
        "gauss_block": "GARCH(1,1) -- Gaussian baseline",
        "stud_block":  "GARCH(1,1) -- Student-t baseline",
    },
    {
        "title": "2. GARCH-X (tone) — G&amp;P Eq. (4)",
        "variance": (
            "&sigma;<sup>2</sup><sub>t</sub> = &omega; "
            "+ &alpha; &middot; r<sup>2</sup><sub>t-1</sub> "
            "+ &beta; &middot; &sigma;<sup>2</sup><sub>t-1</sub> "
            "+ &gamma; &middot; tone<sup>2</sup><sub>t-1</sub>"
        ),
        "explanation": (
            "Symmetric extension that injects the daily average news tone "
            "as an exogenous regressor in the variance equation. We want to "
            "test whether news intensity (regardless of sign) generates "
            "additional volatility in REMX, i.e. whether &gamma; is "
            "statistically significant."
        ),
        "gauss_block": "GARCH-X (tone)",
        "stud_block":  "GARCH-X (tone) -- Student-t",
    },
    {
        "title": "3. GARCH-X (log article growth) — G&amp;P Eq. (4)",
        "variance": (
            "&sigma;<sup>2</sup><sub>t</sub> = &omega; "
            "+ &alpha; &middot; r<sup>2</sup><sub>t-1</sub> "
            "+ &beta; &middot; &sigma;<sup>2</sup><sub>t-1</sub> "
            "+ &gamma; &middot; log_art_growth<sup>2</sup><sub>t-1</sub>"
        ),
        "explanation": (
            "Same structure as the previous GARCH-X, but using the log-growth "
            "in article volume (log(1+N<sub>t</sub>) &minus; log(1+N<sub>t-1</sub>)) "
            "as a proxy for news intensity. The log transform replaces the raw "
            "growth rate to remove a single-day outlier (art_growth = 61 on "
            "2015-10-23) that would otherwise dominate the squared term. We want "
            "to see whether sustained changes in media coverage (a significant "
            "&gamma;) anticipate next-day volatility."
        ),
        "gauss_block": "GARCH-X (log article growth)",
        "stud_block":  "GARCH-X (log article growth) -- Student-t",
    },
    {
        "title": "4. GARCHAND (tone) — G&amp;P Eq. (5)",
        "variance": (
            "&sigma;<sup>2</sup><sub>t</sub> = &omega; "
            "+ &alpha; &middot; r<sup>2</sup><sub>t-1</sub> "
            "+ &beta; &middot; &sigma;<sup>2</sup><sub>t-1</sub> "
            "+ &gamma; (1 + d<sub>1</sub> &middot; &theta;) "
            "&middot; tone<sup>2</sup><sub>t-1</sub>,   "
            "d<sub>1</sub> = 1{tone<sub>t-1</sub> &lt; 0}"
        ),
        "explanation": (
            "Asymmetric version of GARCH-X: when the previous day's tone is "
            "negative, its impact on variance is amplified by the factor "
            "(1 + &theta;). We want to demonstrate that negative news drives "
            "a stronger volatility response than positive news (&theta; &gt; "
            "0 and significant)."
        ),
        "gauss_block": "GARCHAND (tone)",
        "stud_block":  "GARCHAND (tone) -- Student-t",
    },
    {
        "title": "5. GARCHAND (log article growth) — G&amp;P Eq. (6)",
        "variance": (
            "&sigma;<sup>2</sup><sub>t</sub> = &omega; "
            "+ &alpha; &middot; r<sup>2</sup><sub>t-1</sub> "
            "+ &beta; &middot; &sigma;<sup>2</sup><sub>t-1</sub> "
            "+ &gamma; &middot; d<sub>2</sub> "
            "&middot; log_art_growth<sup>2</sup><sub>t-1</sub>,   "
            "d<sub>2</sub> = 1{log_art_growth<sub>t-1</sub> &gt; 0}"
        ),
        "explanation": (
            "The article-volume effect is switched on only on days in which "
            "the news flow GROWS relative to the previous day (log_art_growth "
            "&gt; 0). We want to show that novelty (rising coverage) is what "
            "moves volatility, rather than low news volumes."
        ),
        "gauss_block": "GARCHAND (log article growth)",
        "stud_block":  "GARCHAND (log article growth) -- Student-t",
    },
    {
        "title": "6. GARCHND (tone &amp; log article growth) — G&amp;P Eq. (7)",
        "variance": (
            "&sigma;<sup>2</sup><sub>t</sub> = &omega; "
            "+ &alpha; &middot; r<sup>2</sup><sub>t-1</sub> "
            "+ &beta; &middot; &sigma;<sup>2</sup><sub>t-1</sub> "
            "+ &gamma; &middot; d<sub>3</sub> "
            "&middot; x<sup>2</sup><sub>t-1</sub>,   "
            "d<sub>3</sub> = 1{&sigma;<sup>2</sup><sub>t-1</sub> "
            "&ge; &kappa;},   x &isin; {tone, log_art_growth}"
        ),
        "explanation": (
            "The news impact activates only when the previous day's variance "
            "exceeds a threshold &kappa;, calibrated from an annualised "
            "volatility level (30% and 50%). We want to test whether the "
            "sentiment effect is non-linear in the volatility regime: a "
            "significant &gamma; in the high-volatility regime would imply "
            "that news amplifies volatility only when the market is already "
            "stressed."
        ),
        "gauss_block": "GARCHND (tone, log_art_growth)",
        "stud_block":  "GARCHND (tone, log_art_growth) -- Student-t",
        "is_garchnd": True,
    },
    {
        "title": "7. EGARCH(1,1)-X — Extension model",
        "variance": (
            "ln(&sigma;<sup>2</sup><sub>t</sub>) = &omega; "
            "+ &alpha; [|z<sub>t-1</sub>| - E|z<sub>t-1</sub>|] "
            "+ &xi; &middot; z<sub>t-1</sub> "
            "+ &beta; &middot; ln(&sigma;<sup>2</sup><sub>t-1</sub>) "
            "+ &gamma;<sub>1</sub> &middot; tone<sub>t-1</sub> "
            "+ &gamma;<sub>2</sub> &middot; log_art_growth<sub>t-1</sub>"
        ),
        "explanation": (
            "Log-variance specification (Nelson, 1991) with two exogenous "
            "regressors. It (i) guarantees positivity of &sigma;<sup>2</sup> "
            "without parameter restrictions, (ii) captures asymmetry through "
            "the leverage term &xi; (negative shocks weigh more than "
            "positive ones), and (iii) allows tone and log_art_growth to enter "
            "linearly (not as squares). We want to show that EGARCH-X "
            "captures asymmetry more cleanly than GARCHAND, and to test "
            "whether &gamma;<sub>1</sub> and &gamma;<sub>2</sub> are "
            "significant in log-variance rather than in squared variance."
        ),
        "is_extension": True,
    },
]

# --------------------------------------------------------------------------- #
# Robustness check data (hardcoded from notebook 06 and 07 outputs)
# --------------------------------------------------------------------------- #

# Check 1 — Rolling-window OOS forecasting (notebook 06, cells 5 and 8)
# OOS: 2023-01-03 -> 2026-03-31, 813 origins, parameters re-estimated daily
# on a 1,260-day rolling sample. Both RV proxies (Parkinson, Rogers-Satchell)
# in %²/day. Losses: HRMSE, QLIKE, RMSE, MAE. DM (two-sided), CW (one-sided).
# Sort order: model -> proxy -> h.

# [Model, Proxy, h, HRMSE, QLIKE, RMSE, MAE, N]
OOS_LOSS = [
    ["GARCH(1,1)",            "Parkinson",       "1",   "5.4023", "2.0760", "4.8866", "3.9585", "813"],
    ["GARCH(1,1)",            "Parkinson",       "5",   "2.9753", "2.0933", "4.4833", "3.7785", "809"],
    ["GARCH(1,1)",            "Parkinson",      "22",   "2.6582", "2.1211", "4.1826", "3.6778", "792"],
    ["GARCH(1,1)",            "Rogers-Satchell", "1",  "12.5808", "2.0617", "4.8178", "3.9280", "813"],
    ["GARCH(1,1)",            "Rogers-Satchell", "5",   "2.9436", "2.0791", "4.3905", "3.7677", "809"],
    ["GARCH(1,1)",            "Rogers-Satchell","22",   "2.6068", "2.1039", "4.1738", "3.6484", "792"],
    ["GARCH-X tone",          "Parkinson",       "1",   "5.4271", "2.0793", "4.9458", "4.0022", "813"],
    ["GARCH-X tone",          "Parkinson",       "5",   "2.9987", "2.0971", "4.5588", "3.8321", "809"],
    ["GARCH-X tone",          "Parkinson",      "22",   "2.6664", "2.1250", "4.2746", "3.7617", "792"],
    ["GARCH-X tone",          "Rogers-Satchell", "1",  "12.6381", "2.0646", "4.8757", "3.9737", "813"],
    ["GARCH-X tone",          "Rogers-Satchell", "5",   "2.9624", "2.0825", "4.4649", "3.8186", "809"],
    ["GARCH-X tone",          "Rogers-Satchell","22",   "2.6122", "2.1073", "4.2596", "3.7338", "792"],
    ["GARCH-X log_artg",      "Parkinson",       "1",   "5.4685", "2.0832", "4.9566", "4.0203", "813"],
    ["GARCH-X log_artg",      "Parkinson",       "5",   "3.0431", "2.1041", "4.5634", "3.8583", "809"],
    ["GARCH-X log_artg",      "Parkinson",      "22",   "2.8084", "2.1412", "4.3044", "3.8024", "792"],
    ["GARCH-X log_artg",      "Rogers-Satchell", "1",  "12.8300", "2.0684", "4.8846", "3.9919", "813"],
    ["GARCH-X log_artg",      "Rogers-Satchell", "5",   "3.0172", "2.0895", "4.4682", "3.8420", "809"],
    ["GARCH-X log_artg",      "Rogers-Satchell","22",   "2.7595", "2.1235", "4.2922", "3.7602", "792"],
    ["GARCHAND tone",         "Parkinson",       "1",   "5.6238", "2.1006", "5.0643", "4.1395", "813"],
    ["GARCHAND tone",         "Parkinson",       "5",   "3.0913", "2.1121", "4.6541", "3.9394", "809"],
    ["GARCHAND tone",         "Parkinson",      "22",   "2.6320", "2.1167", "4.2332", "3.6960", "792"],
    ["GARCHAND tone",         "Rogers-Satchell", "1",  "14.5021", "2.0868", "4.9992", "4.1233", "813"],
    ["GARCHAND tone",         "Rogers-Satchell", "5",   "3.0861", "2.0983", "4.5657", "3.9263", "809"],
    ["GARCHAND tone",         "Rogers-Satchell","22",   "2.5894", "2.0994", "4.2198", "3.6645", "792"],
    ["GARCHAND log_artg",     "Parkinson",       "1",   "5.4361", "2.0799", "4.9393", "3.9990", "813"],
    ["GARCHAND log_artg",     "Parkinson",       "5",   "3.0202", "2.0993", "4.5381", "3.8200", "809"],
    ["GARCHAND log_artg",     "Parkinson",      "22",   "2.7417", "2.1279", "4.2219", "3.7027", "792"],
    ["GARCHAND log_artg",     "Rogers-Satchell", "1",  "12.6889", "2.0653", "4.8685", "3.9704", "813"],
    ["GARCHAND log_artg",     "Rogers-Satchell", "5",   "2.9907", "2.0848", "4.4437", "3.8048", "809"],
    ["GARCHAND log_artg",     "Rogers-Satchell","22",   "2.6878", "2.1103", "4.2095", "3.6630", "792"],
    ["GARCHND κ=30% tone",    "Parkinson",       "1",   "5.4311", "2.0787", "4.9183", "3.9863", "813"],
    ["GARCHND κ=30% tone",    "Parkinson",       "5",   "2.9983", "2.0970", "4.5302", "3.8153", "809"],
    ["GARCHND κ=30% tone",    "Parkinson",      "22",   "2.6775", "2.1271", "4.2534", "3.7571", "792"],
    ["GARCHND κ=30% tone",    "Rogers-Satchell", "1",  "12.6402", "2.0639", "4.8479", "3.9530", "813"],
    ["GARCHND κ=30% tone",    "Rogers-Satchell", "5",   "2.9653", "2.0825", "4.4361", "3.8023", "809"],
    ["GARCHND κ=30% tone",    "Rogers-Satchell","22",   "2.6262", "2.1093", "4.2392", "3.7267", "792"],
    ["GARCHND κ=30% log_artg","Parkinson",       "1",   "5.4728", "2.0808", "4.9281", "3.9974", "813"],
    ["GARCHND κ=30% log_artg","Parkinson",       "5",   "3.0606", "2.1033", "4.5380", "3.8333", "809"],
    ["GARCHND κ=30% log_artg","Parkinson",      "22",   "2.8800", "2.1401", "4.3128", "3.7856", "792"],
    ["GARCHND κ=30% log_artg","Rogers-Satchell", "1",  "12.7754", "2.0670", "4.8603", "3.9673", "813"],
    ["GARCHND κ=30% log_artg","Rogers-Satchell", "5",   "3.0416", "2.0901", "4.4502", "3.8278", "809"],
    ["GARCHND κ=30% log_artg","Rogers-Satchell","22",   "2.8225", "2.1244", "4.3106", "3.7493", "792"],
    ["GARCHND κ=50% tone",    "Parkinson",       "1",   "5.0987", "2.0599", "4.6817", "3.8280", "813"],
    ["GARCHND κ=50% tone",    "Parkinson",       "5",   "2.8735", "2.0820", "4.3645", "3.7098", "809"],
    ["GARCHND κ=50% tone",    "Parkinson",      "22",   "2.7304", "2.1394", "4.5071", "3.9632", "792"],
    ["GARCHND κ=50% tone",    "Rogers-Satchell", "1",  "12.1220", "2.0447", "4.6146", "3.7828", "813"],
    ["GARCHND κ=50% tone",    "Rogers-Satchell", "5",   "2.8209", "2.0673", "4.2739", "3.7039", "809"],
    ["GARCHND κ=50% tone",    "Rogers-Satchell","22",   "2.6711", "2.1221", "4.4999", "3.9479", "792"],
    ["GARCHND κ=50% log_artg","Parkinson",       "1",   "5.2228", "2.0639", "4.5829", "3.7801", "813"],
    ["GARCHND κ=50% log_artg","Parkinson",       "5",   "2.8647", "2.0817", "4.1814", "3.6072", "809"],
    ["GARCHND κ=50% log_artg","Parkinson",      "22",   "2.5875", "2.1172", "4.0473", "3.6191", "792"],
    ["GARCHND κ=50% log_artg","Rogers-Satchell", "1",  "12.4078", "2.0485", "4.4993", "3.7318", "813"],
    ["GARCHND κ=50% log_artg","Rogers-Satchell", "5",   "2.8427", "2.0666", "4.0714", "3.5971", "809"],
    ["GARCHND κ=50% log_artg","Rogers-Satchell","22",   "2.5565", "2.0995", "4.0380", "3.5954", "792"],
    ["EGARCH-X",              "Parkinson",       "1",   "5.6122", "2.1150", "4.9407", "4.1347", "813"],
    ["EGARCH-X",              "Parkinson",       "5",   "3.0832", "2.1221", "4.4406", "3.8878", "809"],
    ["EGARCH-X",              "Parkinson",      "22",   "2.5222", "2.1166", "3.9003", "3.5732", "792"],
    ["EGARCH-X",              "Rogers-Satchell", "1",  "15.0765", "2.0988", "4.8522", "4.0859", "813"],
    ["EGARCH-X",              "Rogers-Satchell", "5",   "3.1362", "2.1060", "4.3254", "3.8659", "809"],
    ["EGARCH-X",              "Rogers-Satchell","22",   "2.5188", "2.0984", "3.8857", "3.5367", "792"],
]

# DM (two-sided) and Clark-West (one-sided) vs GARCH(1,1) Student-t baseline.
# DM > 0 ⇒ candidate has lower MSE; CW > 0 ⇒ candidate adds predictive content vs nested baseline.
# [Candidate, Proxy, h, DM, DM_p, CW, CW_p]
OOS_DM_CW = [
    ["GARCH-X tone",          "Parkinson",       "1",   "−7.883", "0.0000", "−7.768", "1.0000"],
    ["GARCH-X tone",          "Parkinson",       "5",   "−4.738", "0.0000", "−4.690", "1.0000"],
    ["GARCH-X tone",          "Parkinson",      "22",   "−2.061", "0.0393", "−2.022", "0.9784"],
    ["GARCH-X tone",          "Rogers-Satchell", "1",   "−7.647", "0.0000", "−7.510", "1.0000"],
    ["GARCH-X tone",          "Rogers-Satchell", "5",   "−4.569", "0.0000", "−4.503", "1.0000"],
    ["GARCH-X tone",          "Rogers-Satchell","22",   "−2.048", "0.0405", "−1.996", "0.9770"],
    ["GARCH-X log_artg",      "Parkinson",       "1",   "−6.321", "0.0000", "−5.960", "1.0000"],
    ["GARCH-X log_artg",      "Parkinson",       "5",   "−3.891", "0.0001", "−3.664", "0.9999"],
    ["GARCH-X log_artg",      "Parkinson",      "22",   "−2.203", "0.0276", "−1.929", "0.9731"],
    ["GARCH-X log_artg",      "Rogers-Satchell", "1",   "−5.862", "0.0000", "−5.479", "1.0000"],
    ["GARCH-X log_artg",      "Rogers-Satchell", "5",   "−3.689", "0.0002", "−3.450", "0.9997"],
    ["GARCH-X log_artg",      "Rogers-Satchell","22",   "−2.135", "0.0328", "−1.856", "0.9683"],
    ["GARCHAND tone",         "Parkinson",       "1",  "−12.523", "0.0000","−11.809", "1.0000"],
    ["GARCHAND tone",         "Parkinson",       "5",   "−5.973", "0.0000", "−5.555", "1.0000"],
    ["GARCHAND tone",         "Parkinson",      "22",   "−0.823", "0.4103", "−0.397", "0.6542"],
    ["GARCHAND tone",         "Rogers-Satchell", "1",  "−12.691", "0.0000","−11.987", "1.0000"],
    ["GARCHAND tone",         "Rogers-Satchell", "5",   "−5.956", "0.0000", "−5.540", "1.0000"],
    ["GARCHAND tone",         "Rogers-Satchell","22",   "−0.785", "0.4323", "−0.335", "0.6312"],
    ["GARCHAND log_artg",     "Parkinson",       "1",   "−4.534", "0.0000", "−4.098", "1.0000"],
    ["GARCHAND log_artg",     "Parkinson",       "5",   "−2.687", "0.0072", "−2.457", "0.9930"],
    ["GARCHAND log_artg",     "Parkinson",      "22",   "−0.922", "0.3565", "−0.640", "0.7388"],
    ["GARCHAND log_artg",     "Rogers-Satchell", "1",   "−4.221", "0.0000", "−3.776", "0.9999"],
    ["GARCHAND log_artg",     "Rogers-Satchell", "5",   "−2.521", "0.0117", "−2.285", "0.9888"],
    ["GARCHAND log_artg",     "Rogers-Satchell","22",   "−0.834", "0.4045", "−0.552", "0.7095"],
    ["GARCHND κ=30% tone",    "Parkinson",       "1",   "−4.436", "0.0000", "−4.211", "1.0000"],
    ["GARCHND κ=30% tone",    "Parkinson",       "5",   "−3.798", "0.0001", "−3.723", "0.9999"],
    ["GARCHND κ=30% tone",    "Parkinson",      "22",   "−2.770", "0.0056", "−2.804", "0.9975"],
    ["GARCHND κ=30% tone",    "Rogers-Satchell", "1",   "−4.236", "0.0000", "−3.999", "1.0000"],
    ["GARCHND κ=30% tone",    "Rogers-Satchell", "5",   "−3.644", "0.0003", "−3.558", "0.9998"],
    ["GARCHND κ=30% tone",    "Rogers-Satchell","22",   "−2.751", "0.0059", "−2.772", "0.9972"],
    ["GARCHND κ=30% log_artg","Parkinson",       "1",   "−3.152", "0.0016", "−2.351", "0.9906"],
    ["GARCHND κ=30% log_artg","Parkinson",       "5",   "−1.996", "0.0459", "−1.580", "0.9430"],
    ["GARCHND κ=30% log_artg","Parkinson",      "22",   "−1.376", "0.1690", "−1.059", "0.8552"],
    ["GARCHND κ=30% log_artg","Rogers-Satchell", "1",   "−3.283", "0.0010", "−2.454", "0.9929"],
    ["GARCHND κ=30% log_artg","Rogers-Satchell", "5",   "−2.123", "0.0338", "−1.717", "0.9570"],
    ["GARCHND κ=30% log_artg","Rogers-Satchell","22",   "−1.422", "0.1551", "−1.112", "0.8670"],
    ["GARCHND κ=50% tone",    "Parkinson",       "1",   "+4.102", "0.0000", "+4.792", "0.0000"],
    ["GARCHND κ=50% tone",    "Parkinson",       "5",   "+1.228", "0.2193", "+1.813", "0.0349"],
    ["GARCHND κ=50% tone",    "Parkinson",      "22",   "−2.336", "0.0195", "−1.057", "0.8547"],
    ["GARCHND κ=50% tone",    "Rogers-Satchell", "1",   "+4.118", "0.0000", "+4.800", "0.0000"],
    ["GARCHND κ=50% tone",    "Rogers-Satchell", "5",   "+1.150", "0.2501", "+1.750", "0.0401"],
    ["GARCHND κ=50% tone",    "Rogers-Satchell","22",   "−2.370", "0.0178", "−1.073", "0.8583"],
    ["GARCHND κ=50% log_artg","Parkinson",       "1",   "+6.775", "0.0000", "+7.009", "0.0000"],
    ["GARCHND κ=50% log_artg","Parkinson",       "5",   "+3.300", "0.0010", "+3.342", "0.0004"],
    ["GARCHND κ=50% log_artg","Parkinson",      "22",   "+0.996", "0.3195", "+1.157", "0.1237"],
    ["GARCHND κ=50% log_artg","Rogers-Satchell", "1",   "+7.088", "0.0000", "+7.261", "0.0000"],
    ["GARCHND κ=50% log_artg","Rogers-Satchell", "5",   "+3.300", "0.0010", "+3.341", "0.0004"],
    ["GARCHND κ=50% log_artg","Rogers-Satchell","22",   "+0.983", "0.3254", "+1.144", "0.1264"],
    ["EGARCH-X",              "Parkinson",       "1",   "−1.048", "0.2948", "+0.967", "0.1668"],
    ["EGARCH-X",              "Parkinson",       "5",   "+0.423", "0.6724", "+1.312", "0.0948"],
    ["EGARCH-X",              "Parkinson",      "22",   "+1.608", "0.1078", "+1.839", "0.0330"],
    ["EGARCH-X",              "Rogers-Satchell", "1",   "−0.657", "0.5111", "+1.334", "0.0912"],
    ["EGARCH-X",              "Rogers-Satchell", "5",   "+0.614", "0.5392", "+1.454", "0.0730"],
    ["EGARCH-X",              "Rogers-Satchell","22",   "+1.601", "0.1093", "+1.827", "0.0339"],
]

# Check 2 — Placebo (notebook 06, cell 8)
PLACEBO_TONE = [
    # [Spec, LL, AIC, BIC, gamma, se(gamma), t(gamma)]
    ["original",           "-5951.97", "11913.94", "11943.57", "-0.8262", "0.0602", "-13.72"],
    ["regime-only (γ·d3)", "-5950.18", "11910.37", "11939.99", "-1.1218", "0.0172", "-65.10"],
    ["placebo seed=2026",  "-5955.76", "11921.52", "11951.15", "-0.1501", "0.0283",  "-5.31"],
    ["placebo seed=2027",  "-5949.09", "11908.18", "11937.80", "-0.7548", "0.0233", "-32.41"],
    ["placebo seed=2028",  "-5952.23", "11914.46", "11944.08", "-0.3636", "0.0213", "-17.03"],
    ["placebo seed=2029",  "-5950.81", "11911.63", "11941.25", "-0.8354", "0.0175", "-47.80"],
    ["placebo seed=2030",  "-5952.38", "11914.75", "11944.38", "-0.3935", "0.0610",  "-6.45"],
]

PLACEBO_ARTG = [
    ["original",           "-5951.25", "11912.49", "11942.12", "-1.4116", "0.2184",  "-6.46"],
    ["regime-only (γ·d3)", "-5950.18", "11910.37", "11939.99", "-1.1218", "0.0172", "-65.10"],
    ["placebo seed=2026",  "-5955.15", "11920.29", "11949.92", "-0.8036", "0.1258",  "-6.39"],
    ["placebo seed=2027",  "-5955.34", "11920.67", "11950.30", "-0.2475", "0.1213",  "-2.04"],
    ["placebo seed=2028",  "-5955.10", "11920.19", "11949.82", "-0.3646", "0.1288",  "-2.83"],
    ["placebo seed=2029",  "-5949.72", "11909.44", "11939.06", "-1.0843", "0.0814", "-13.32"],
    ["placebo seed=2030",  "-5954.52", "11919.04", "11948.66", "-0.4900", "0.0936",  "-5.24"],
]

# Check 3 — SE stability across sigmoid sharpness K (notebook 06, cell 10)
STABILITY_TONE = [
    # [K, LL, gamma, SE(gamma), t(gamma), converged]
    ["K=5",      "-5950.65", "-1.0209", "0.1238",    "-8.24", "True"],
    ["K=10",     "-5950.57", "-1.0253", "0.1020",   "-10.05", "True"],
    ["K=20",     "-5950.47", "-1.0966", "0.1445",    "-7.59", "True"],
    ["K=50",     "-5950.59", "-0.9086", "0.0248",   "-36.65", "True"],
    ["hard d3",  "-5954.05", "-0.2857", "0.0001", "-2318.55", "True"],
]

STABILITY_ARTG = [
    ["K=5",      "-5954.08", "-1.1225", "0.3712",    "-3.02", "True"],
    ["K=10",     "-5954.01", "-1.1592", "0.1874",    "-6.19", "True"],
    ["K=20",     "-5954.03", "-0.9972", "0.1635",    "-6.10", "True"],
    ["K=50",     "-5953.87", "-1.2235", "0.0888",   "-13.78", "True"],
    ["hard d3",  "-5955.11", "-0.4223", "0.0001", "-6100.89", "True"],
]

# Check 4 — Cluster count (notebook 06, cell 12)
CLUSTERS_TONE = [
    ["1", "2021-02-10", "2021-04-05", "37"],
    ["2", "2020-03-11", "2020-04-21", "29"],
    ["3", "2025-10-15", "2025-11-13", "22"],
    ["4", "2026-02-03", "2026-02-18", "11"],
    ["5", "2018-12-24", "2019-01-02",  "6"],
    ["6", "2022-11-08", "2022-11-15",  "6"],
    ["7", "2025-04-11", "2025-04-17",  "5"],
    ["8", "2026-03-05", "2026-03-11",  "5"],
    ["9", "2026-03-25", "2026-03-30",  "4"],
    ["10","2021-01-11", "2021-01-13",  "3"],
]

CLUSTERS_ARTG = [
    ["1", "2021-02-04", "2021-03-29", "37"],
    ["2", "2020-03-04", "2020-04-23", "36"],
    ["3", "2025-10-15", "2025-11-12", "21"],
    ["4", "2026-03-05", "2026-03-31", "19"],
    ["5", "2024-09-30", "2024-10-14", "11"],
    ["6", "2026-02-03", "2026-02-18", "11"],
    ["7", "2018-12-24", "2019-01-08", "10"],
    ["8", "2025-04-11", "2025-04-22",  "7"],
    ["9", "2022-11-08", "2022-11-14",  "5"],
    ["10","2021-01-07", "2021-01-12",  "4"],
]

# Check 5 — HAC/MBB SEs (notebook 07, cell 3)
HAC_TONE = [
    # [Method, gamma, SE(gamma), t(gamma)]
    ["iid sandwich", "-1.0966", "0.1445", "-7.59"],
    ["HAC L=25",     "-1.0966", "0.1360", "-8.06"],
    ["HAC L=50",     "-1.0966", "0.1400", "-7.83"],
    ["HAC L=75",     "-1.0966", "0.1413", "-7.76"],
    ["HAC L=100",    "-1.0966", "0.1411", "-7.77"],
    ["MBB L=25",     "-1.0966", "0.1337", "-8.20"],
    ["MBB L=50",     "-1.0966", "0.1416", "-7.74"],
    ["MBB L=75",     "-1.0966", "0.1403", "-7.82"],
    ["MBB L=100",    "-1.0966", "0.1448", "-7.57"],
]

HAC_ARTG = [
    ["iid sandwich", "-0.9972", "0.1635", "-6.10"],
    ["HAC L=25",     "-0.9972", "0.1444", "-6.91"],
    ["HAC L=50",     "-0.9972", "0.1421", "-7.02"],
    ["HAC L=75",     "-0.9972", "0.1453", "-6.86"],
    ["HAC L=100",    "-0.9972", "0.1458", "-6.84"],
    ["MBB L=25",     "-0.9972", "0.1496", "-6.67"],
    ["MBB L=50",     "-0.9972", "0.1469", "-6.79"],
    ["MBB L=75",     "-0.9972", "0.1479", "-6.74"],
    ["MBB L=100",    "-0.9972", "0.1489", "-6.69"],
]

# Check 6 — GDELT coverage (notebook 07, cells 5–6)
GDELT_CONCEPTS = [
    # [Concept, N articles, share]
    ["has_cerium",        "186,346", "59.34%"],
    ["has_rare_earth",     "92,383", "29.42%"],
    ["has_lynas",          "26,066",  "8.30%"],
    ["has_samarium",        "5,005",  "1.59%"],
    ["has_neodymium",       "2,960",  "0.94%"],
    ["has_mp_materials",    "2,891",  "0.92%"],
    ["has_praseodymium",    "2,124",  "0.68%"],
    ["has_arafura",         "1,730",  "0.55%"],
    ["has_europium",        "1,153",  "0.37%"],
    ["has_terbium",         "1,143",  "0.36%"],
    ["has_gadolinium",      "1,051",  "0.33%"],
    ["has_yttrium",         "1,030",  "0.33%"],
    ["has_lanthanum",         "877",  "0.28%"],
    ["has_china_northern",    "702",  "0.22%"],
    ["has_dysprosium",        "576",  "0.18%"],
    ["has_ndfeb",             "505",  "0.16%"],
    ["has_ndpr",              "454",  "0.14%"],
    ["has_remx",              "150",  "0.05%"],
    ["has_jl_mag",             "28",  "0.01%"],
]

GDELT_TIERS = [
    # [Tier, N articles, Art/day, % zero-days, mean tone, std tone]
    ["Broad (≥1 concept, used in model)", "314,048", "78.5",  "0.5%", "−0.280", "3.299"],
    ["Medium (metal OR company)",         "231,708", "57.9",  "0.5%", "−0.453", "3.585"],
    ["Tight (rare_earth AND metal/co.)",   "10,158",  "2.5", "38.5%", "−0.130", "2.133"],
    ["Ultra-tight (metal AND company)",       "559",  "0.14","91.4%", "+0.638", "1.850"],
]


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
HEADER_RE = re.compile(r"^-{60,}\s*\n(.+?)\n-{60,}", re.MULTILINE)


def split_into_blocks(text: str) -> Dict[str, str]:
    blocks: Dict[str, str] = {}
    matches = list(HEADER_RE.finditer(text))
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks[name] = text[start:end].strip()
    return blocks


def parse_parameter_table(block: str) -> Tuple[List[List[str]], List[str]]:
    rows: List[List[str]] = []
    notes: List[str] = []
    lines = block.splitlines()

    i = 0
    while i < len(lines):
        if "estimate" in lines[i] and "t_ratio" in lines[i] and "p_value" in lines[i]:
            i += 1
            break
        i += 1

    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln.strip():
            break
        if ln.lstrip().startswith(("Log-likelihood", "AIC", "BIC")):
            break
        parts = ln.split()
        if len(parts) >= 8:
            name = parts[0]
            estimate = parts[1]
            std_err = parts[2]
            t_ratio = parts[3]
            p_value = parts[4]
            sigs = parts[5:8]
            stars = sig_stars(sigs)
            rows.append([name, estimate, std_err, t_ratio, p_value, stars])
        i += 1

    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("Verdict:"):
            break
        if ln and not ln.startswith("--"):
            notes.append(ln)
        i += 1

    return rows, notes


def sig_stars(sig_flags: List[str]) -> str:
    n_true = sum(1 for f in sig_flags if f.lower() == "true")
    if n_true == 3:
        return "***"
    if n_true == 2:
        return "**"
    if n_true == 1:
        return "*"
    return ""


def parse_garchnd_subblocks(block: str) -> List[Tuple[str, List[List[str]], List[str]]]:
    out: List[Tuple[str, List[List[str]], List[str]]] = []
    sub_header_re = re.compile(r"^----\s*(GARCHND.+?)\s*----\s*$", re.MULTILINE)
    matches = list(sub_header_re.finditer(block))
    for i, m in enumerate(matches):
        sub_title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        sub_text = block[start:end].strip()
        rows, notes = parse_parameter_table(sub_text)
        out.append((sub_title, rows, notes))
    return out


def parse_extension_file(path: Path) -> Tuple[List[List[str]], List[str]]:
    text = path.read_text()
    return parse_parameter_table(text)


# --------------------------------------------------------------------------- #
# PDF building
# --------------------------------------------------------------------------- #
def build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle", parent=styles["Title"],
        fontSize=20, leading=24, spaceAfter=18, alignment=TA_LEFT,
        textColor=colors.HexColor("#1F3A68"),
    ))
    styles.add(ParagraphStyle(
        name="ModelTitle", parent=styles["Heading1"],
        fontSize=14, leading=18, spaceBefore=14, spaceAfter=8,
        textColor=colors.HexColor("#1F3A68"),
    ))
    styles.add(ParagraphStyle(
        name="SubSection", parent=styles["Heading2"],
        fontSize=11, leading=14, spaceBefore=10, spaceAfter=4,
        textColor=colors.HexColor("#444444"),
    ))
    styles.add(ParagraphStyle(
        name="VarianceEq", parent=styles["BodyText"],
        fontName="Helvetica-Bold", fontSize=11, leading=15,
        backColor=colors.HexColor("#F4F4F4"),
        borderColor=colors.HexColor("#CCCCCC"), borderWidth=0.5,
        borderPadding=8, spaceBefore=4, spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="Body", parent=styles["BodyText"],
        fontSize=10, leading=13, alignment=TA_JUSTIFY,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="NoteLine", parent=styles["BodyText"],
        fontName="Courier", fontSize=8.5, leading=11, spaceAfter=2,
        textColor=colors.HexColor("#333333"),
    ))
    styles.add(ParagraphStyle(
        name="SubSubTitle", parent=styles["BodyText"],
        fontSize=10, leading=13, spaceBefore=6, spaceAfter=2,
        textColor=colors.HexColor("#222222"),
        fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        name="Verdict", parent=styles["BodyText"],
        fontSize=9.5, leading=13, spaceBefore=4, spaceAfter=6,
        textColor=colors.HexColor("#8B0000"),
        fontName="Helvetica-Bold",
    ))
    return styles


def _tbl(data, col_widths, header_row=True) -> Table:
    tbl = Table(data, hAlign="LEFT", colWidths=col_widths)
    style = [
        ("FONTSIZE",   (0, 0), (-1, -1), 8.5),
        ("GRID",       (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
            [colors.whitesmoke, colors.HexColor("#FAFAFA")]),
    ]
    if header_row:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A68")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    tbl.setStyle(TableStyle(style))
    return tbl


def parameter_table(rows: List[List[str]]) -> Table:
    header = ["Parameter", "Estimate", "Std. Err.", "t-ratio", "p-value", "Sig."]
    data = [header] + rows
    tbl = Table(
        data, hAlign="LEFT",
        colWidths=[3.2 * cm, 2.6 * cm, 2.6 * cm, 2.4 * cm, 2.4 * cm, 1.6 * cm],
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A68")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ALIGN",      (1, 1), (-2, -1), "RIGHT"),
        ("ALIGN",      (-1, 1), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
            [colors.whitesmoke, colors.HexColor("#FAFAFA")]),
        ("GRID",       (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return tbl


def render_results(story, styles, label: str, rows, notes):
    story.append(Paragraph(label, styles["SubSection"]))
    if not rows:
        story.append(Paragraph("<i>No results found in source file.</i>", styles["Body"]))
        return
    story.append(parameter_table(rows))
    story.append(Spacer(1, 4))
    for ln in notes:
        story.append(Paragraph(
            ln.replace("<", "&lt;").replace(">", "&gt;"),
            styles["NoteLine"],
        ))


def render_garchnd(story, styles, label: str, sub_results):
    story.append(Paragraph(label, styles["SubSection"]))
    if not sub_results:
        story.append(Paragraph("<i>No sub-runs found.</i>", styles["Body"]))
        return
    for sub_title, rows, notes in sub_results:
        story.append(Paragraph(sub_title, styles["SubSubTitle"]))
        story.append(parameter_table(rows))
        story.append(Spacer(1, 3))
        for ln in notes:
            story.append(Paragraph(
                ln.replace("<", "&lt;").replace(">", "&gt;"),
                styles["NoteLine"],
            ))
        story.append(Spacer(1, 6))


# --------------------------------------------------------------------------- #
# Robustness check renderers
# --------------------------------------------------------------------------- #

def render_check1(story, styles):
    story.append(Paragraph(
        "Check 1 — Rolling-window out-of-sample forecasting (h = 1, 5, 22 days)",
        styles["ModelTitle"],
    ))
    story.append(Paragraph(
        "Out-of-sample window: <b>2023-01-03 to 2026-03-31</b> (813 forecast origins). "
        "Parameters are re-estimated <b>daily</b> on a <b>1,260-day rolling sample</b> for "
        "every candidate model in the report (GARCH(1,1) baseline plus 9 augmented specs). "
        "The forecast at origin &tau; is the average of the iterated conditional-variance "
        "path &sigma;<sup>2</sup><sub>&tau;+1|&tau;</sub>,&hellip;,&sigma;<sup>2</sup><sub>&tau;+h|&tau;</sub>; "
        "the target is the corresponding h-day mean of the realised-variance proxy. "
        "Two proxies are reported in parallel: <b>Parkinson</b> (RV_park) and "
        "<b>Rogers&ndash;Satchell</b> (RV_rs), both in %<sup>2</sup>/day. "
        "Losses: HRMSE, QLIKE (Patton 2011, robust to RV-noise), RMSE, MAE. "
        "Tests against the GARCH(1,1) Student-t baseline use Newey&ndash;West LRV at lag h&minus;1: "
        "<b>Diebold&ndash;Mariano</b> (two-sided on squared-loss differential; DM &gt; 0 ⇒ candidate "
        "has lower MSE) and <b>Clark&ndash;West</b> (one-sided, the appropriate test for "
        "nested models since the baseline is &gamma; = 0; CW &gt; 0 ⇒ candidate adds "
        "genuine predictive content). Rows sorted by model &rarr; proxy &rarr; h.",
        styles["Body"],
    ))

    story.append(Paragraph(
        "Loss functions (averaged over the 813 origins)", styles["SubSection"],
    ))
    hdr = ["Model", "Proxy", "h", "HRMSE", "QLIKE", "RMSE", "MAE", "N"]
    cw = [3.6 * cm, 2.6 * cm, 0.7 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm, 1.7 * cm, 0.9 * cm]
    story.append(_tbl([hdr] + OOS_LOSS, cw))
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "Diebold&ndash;Mariano and Clark&ndash;West vs GARCH(1,1) baseline",
        styles["SubSection"],
    ))
    hdr2 = ["Candidate", "Proxy", "h", "DM", "DM p", "CW", "CW p"]
    cw2 = [3.8 * cm, 2.6 * cm, 0.7 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm]
    story.append(_tbl([hdr2] + OOS_DM_CW, cw2))

    story.append(Paragraph(
        "Verdict: a clean separation emerges across the nine augmented candidates. "
        "<b>The simple symmetric and asymmetric news models are dominated by the "
        "baseline at every horizon and under both RV proxies.</b> "
        "GARCH-X (tone, log_artg) and GARCHAND (tone, log_artg) post strongly "
        "negative DM and one-sided Clark&ndash;West p-values &gt; 0.99 at h = 1 and h = 5: "
        "adding tone<sup>2</sup> or its asymmetric counterpart purely as a variance shifter "
        "destroys forecast accuracy, with GARCHAND tone being the worst performer "
        "(DM &asymp; &minus;12 at h = 1). "
        "<b>The GARCHND κ = 30% specifications also lose to the baseline</b> &mdash; "
        "the 30 %-annualised volatility threshold triggers d<sub>3</sub> too often, "
        "so the &gamma;-term injects noise rather than information. "
        "<b>Only the GARCHND κ = 50% pair adds OOS value</b>, and only in the news "
        "regime they were designed for: <i>κ = 50% with log_art_growth</i> beats "
        "the baseline at h = 1 and h = 5 with high significance under both proxies "
        "(DM &asymp; +6.8–7.1 at h = 1; +3.3 at h = 5; CW p &lt; 0.001 at both horizons), "
        "and is at worst neutral at h = 22 (CW p &asymp; 0.12). "
        "<i>κ = 50% with tone</i> beats the baseline decisively at h = 1 "
        "(DM &asymp; +4.1, CW p &lt; 0.001), is borderline at h = 5 (CW p &asymp; 0.04 "
        "but DM not significant), and turns negative at h = 22. "
        "<b>EGARCH-X</b> is roughly tied with the baseline at h = 1 and h = 5 "
        "(insignificant DM, weakly positive CW) and posts a marginal positive CW "
        "edge at h = 22 (p &asymp; 0.03), suggesting log-variance plus a small "
        "leverage term captures something the squared-variance specifications miss "
        "at the monthly horizon. Taken together, the picture is consistent with a "
        "short-lived news shock that is informative only when the previous day was "
        "already in a high-volatility regime &mdash; exactly the d<sub>3</sub> "
        "switch the κ = 50% GARCHND was designed around &mdash; and is otherwise "
        "noise.",
        styles["Verdict"],
    ))


def render_check2(story, styles):
    story.append(Paragraph("Check 2 — Placebo / regime-only test", styles["ModelTitle"]))
    story.append(Paragraph(
        "Two competing specifications are estimated for each GARCHND κ=50% model: "
        "(a) the regime-only model &gamma;&middot;d<sub>3</sub> (no x<sup>2</sup> term), and "
        "(b) five placebos replacing the real x<sub>t-1</sub> with independent random permutations of it. "
        "If the original model does not beat these, the &gamma; estimate reflects regime mean-reversion, not news.",
        styles["Body"],
    ))

    hdr = ["Specification", "LL", "AIC", "BIC", "&gamma;", "SE(&gamma;)", "t(&gamma;)"]
    cw = [4.2 * cm, 2.2 * cm, 2.4 * cm, 2.4 * cm, 1.8 * cm, 2.0 * cm, 2.0 * cm]

    story.append(Paragraph("x = tone", styles["SubSection"]))
    story.append(_tbl([hdr] + PLACEBO_TONE, cw))
    story.append(Paragraph(
        "Verdict: original does NOT beat the regime-only / placebo baselines "
        "(&Delta;LL vs regime-only = &minus;1.79; placebo LL max = &minus;5949.09 vs original &minus;5951.97). "
        "News effect is likely a regime artefact.",
        styles["Verdict"],
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("x = log_art_growth", styles["SubSection"]))
    story.append(_tbl([hdr] + PLACEBO_ARTG, cw))
    story.append(Paragraph(
        "Verdict: original does NOT beat the regime-only / placebo baselines "
        "(&Delta;LL vs regime-only = &minus;1.06; placebo LL max = &minus;5949.72 vs original &minus;5951.25). "
        "News effect is likely a regime artefact.",
        styles["Verdict"],
    ))


def render_check3(story, styles):
    story.append(Paragraph("Check 3 — SE stability across sigmoid sharpness K", styles["ModelTitle"]))
    story.append(Paragraph(
        "Both GARCHND κ=50% models are refitted at K &isin; {5, 10, 20, 50} and with a hard "
        "binary indicator (K&rarr;&infin;). Stability is measured by the coefficient of variation "
        "of &gamma; across K values; CV &lt; 0.2 is considered stable.",
        styles["Body"],
    ))

    hdr = ["K", "LL", "&gamma;", "SE(&gamma;)", "t(&gamma;)", "converged"]
    cw = [2.0 * cm, 2.6 * cm, 2.2 * cm, 2.6 * cm, 2.6 * cm, 2.4 * cm]

    story.append(Paragraph("x = tone  (CV = 0.342 — moderately sensitive to K)", styles["SubSection"]))
    story.append(_tbl([hdr] + STABILITY_TONE, cw))
    story.append(Spacer(1, 8))

    story.append(Paragraph("x = log_art_growth  (CV = 0.295 — moderately sensitive to K)", styles["SubSection"]))
    story.append(_tbl([hdr] + STABILITY_ARTG, cw))
    story.append(Paragraph(
        "Verdict: &gamma;&#770; keeps its sign and roughly its magnitude across "
        "smooth K &isin; {5, 10, 20, 50} for both specifications "
        "(tone: &asymp; &minus;0.91 to &minus;1.10; log_artg: &asymp; &minus;1.00 to &minus;1.22), "
        "with coefficients of variation in the 0.30 range &mdash; neither stable "
        "(&lt; 0.2) nor unstable (&gt; 0.5). The hard-indicator fit, by contrast, "
        "collapses to a degenerate Hessian (condition number 10<sup>9</sup>–10<sup>10</sup>, "
        "SE &asymp; 10<sup>&minus;4</sup>) and yields the implausible t-ratios "
        "in the thousands &mdash; a clear sign that the binary d<sub>3</sub> kills "
        "identification. The smoothed version (K = 20, the default) is the right "
        "estimator; the t-ratios it reports are the ones to trust.",
        styles["Verdict"],
    ))


def render_check4(story, styles):
    story.append(Paragraph("Check 4 — High-volatility episode clusters (κ=50%)", styles["ModelTitle"]))
    story.append(Paragraph(
        "Days where &sigma;<sup>2</sup><sub>t-1</sub> &ge; &kappa;<sub>high</sub> (annualised 50% vol) "
        "are grouped into episodes separated by gaps of at most 5 days. "
        "The longest episode determines the recommended block-bootstrap length and HAC bandwidth.",
        styles["Body"],
    ))

    hdr = ["#", "Start", "End", "Days"]
    cw = [0.8 * cm, 3.0 * cm, 3.0 * cm, 1.8 * cm]

    story.append(Paragraph(
        "x = tone: 125 triggered days (4.52%), 17 distinct episodes; "
        "recommended block length &ge; 37 days",
        styles["SubSection"],
    ))
    story.append(_tbl([hdr] + CLUSTERS_TONE, cw))
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "x = log_art_growth: 137 triggered days (4.95%), 21 distinct episodes; "
        "recommended block length &ge; 37 days",
        styles["SubSection"],
    ))
    story.append(_tbl([hdr] + CLUSTERS_ARTG, cw))


def render_check5(story, styles):
    story.append(Paragraph("Check 5 — HAC and Moving-Block-Bootstrap SEs on &gamma; (Notebook 07)", styles["ModelTitle"]))
    story.append(Paragraph(
        "Sandwich (HAC) and moving-block-bootstrap (MBB, B=2000) standard errors on &gamma; "
        "for both GARCHND κ=50% specifications, at block lengths L &isin; {25, 50, 75, 100} days "
        "(informed by the episode clusters in Check 4).",
        styles["Body"],
    ))

    hdr = ["SE method", "&gamma;", "SE(&gamma;)", "t(&gamma;)"]
    cw = [3.2 * cm, 2.4 * cm, 2.8 * cm, 2.8 * cm]

    story.append(Paragraph("x = tone", styles["SubSection"]))
    story.append(_tbl([hdr] + HAC_TONE, cw))
    story.append(Spacer(1, 8))

    story.append(Paragraph("x = log_art_growth", styles["SubSection"]))
    story.append(_tbl([hdr] + HAC_ARTG, cw))
    story.append(Paragraph(
        "Verdict: HAC and moving-block-bootstrap corrections leave the t-ratios "
        "essentially unchanged from the iid sandwich (tone: t &asymp; &minus;7.6 to &minus;8.2 "
        "across all block lengths; log_artg: t &asymp; &minus;6.1 to &minus;7.0). "
        "If anything, the HAC and MBB SEs are slightly tighter than iid at moderate L, "
        "because the news shocks deliver a partly mean-reverting score that under-weights "
        "the long-run variance estimate. The in-sample significance of &gamma; in the "
        "κ = 50% GARCHND specifications is robust to plausible serial-correlation corrections, "
        "consistent with the favourable rolling-window OOS evidence from Check 1.",
        styles["Verdict"],
    ))


def render_check6(story, styles):
    story.append(Paragraph("Check 6 — GDELT coverage of REMX / rare earths (Notebook 07)", styles["ModelTitle"]))
    story.append(Paragraph(
        "The GDELT article cache (314,048 articles, 2015-04-01 to 2026-04-01) was built with a "
        "permissive 19-concept filter (any single match passes). This check quantifies how many "
        "articles are genuinely related to the REMX economy across four progressively tighter filter tiers.",
        styles["Body"],
    ))

    story.append(Paragraph("Articles per concept (full cache)", styles["SubSection"]))
    hdr = ["Concept", "N articles", "Share"]
    cw = [6.0 * cm, 3.2 * cm, 2.4 * cm]
    story.append(_tbl([hdr] + GDELT_CONCEPTS, cw))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Filter tiers", styles["SubSection"]))
    hdr2 = ["Tier", "N articles", "Art/day", "% zero-days", "Mean tone", "Std tone"]
    cw2 = [5.4 * cm, 2.2 * cm, 1.8 * cm, 2.2 * cm, 2.0 * cm, 1.8 * cm]
    story.append(_tbl([hdr2] + GDELT_TIERS, cw2))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    gp_g_blocks = split_into_blocks(FILES["gp_gaussian"].read_text())
    gp_s_blocks = split_into_blocks(FILES["gp_student"].read_text())

    ext_g_rows, ext_g_notes = parse_extension_file(FILES["ext_gaussian"])
    ext_s_rows, ext_s_notes = parse_extension_file(FILES["ext_student"])

    styles = build_styles()
    doc = SimpleDocTemplate(
        str(OUT_PDF), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="Volatility models — REMX",
    )
    story: list = []

    # Cover
    story.append(Paragraph(
        "Volatility models for REMX:<br/>"
        "GARCH, GARCH-X, GARCHAND, GARCHND and EGARCH-X",
        styles["ReportTitle"],
    ))
    story.append(Paragraph(
        "Summary of all models estimated in the notebooks "
        "<b>03-GP-models-gaussian</b>, <b>03-GP-models-student</b> and "
        "<b>04-extension-models</b>. Each section contains the variance "
        "expression, a brief description of what the model is meant to "
        "demonstrate, and the main estimation results (Gaussian first, "
        "Student-t second). The <i>Sig.</i> column summarises significance: "
        "*** = 1%, ** = 5%, * = 10%. "
        "A final part covers the six robustness checks from notebooks "
        "<b>06-robustness-checks</b> and <b>07-extra-robustness-checks</b>.",
        styles["Body"],
    ))
    story.append(Spacer(1, 10))

    # One section per model
    for model in MODELS:
        story.append(Paragraph(model["title"], styles["ModelTitle"]))
        story.append(Paragraph("<b>Conditional variance:</b>", styles["Body"]))
        story.append(Paragraph(model["variance"], styles["VarianceEq"]))
        story.append(Paragraph(
            "<b>What we want to demonstrate:</b> " + model["explanation"],
            styles["Body"],
        ))

        if model.get("is_extension"):
            render_results(story, styles, "Gaussian", ext_g_rows, ext_g_notes)
            render_results(story, styles, "Student-t", ext_s_rows, ext_s_notes)

        elif model.get("is_garchnd"):
            sub_g = parse_garchnd_subblocks(gp_g_blocks.get(model["gauss_block"], ""))
            sub_s = parse_garchnd_subblocks(gp_s_blocks.get(model["stud_block"], ""))
            render_garchnd(story, styles, "Gaussian", sub_g)
            render_garchnd(story, styles, "Student-t", sub_s)

        else:
            g_rows, g_notes = parse_parameter_table(gp_g_blocks.get(model["gauss_block"], ""))
            s_rows, s_notes = parse_parameter_table(gp_s_blocks.get(model["stud_block"], ""))
            render_results(story, styles, "Gaussian", g_rows, g_notes)
            render_results(story, styles, "Student-t", s_rows, s_notes)

        story.append(PageBreak())

    # Robustness checks part
    story.append(Paragraph(
        "Part II — Robustness Checks (Notebooks 06 &amp; 07)",
        styles["ReportTitle"],
    ))
    story.append(Paragraph(
        "Six diagnostic checks. Checks 1–4 are from notebook 06; Checks 5–6 from "
        "notebook 07. Check 1 was extended to a rolling-window scheme (813 origins, "
        "daily refit on 1,260 days, both RV proxies, DM and Clark&ndash;West tests) and "
        "applied to <b>all nine augmented candidates plus the GARCH(1,1) baseline</b>, "
        "not only the GARCHND κ=50% pair. The headline result is sharper than the "
        "earlier static-OOS reading: only the κ=50% GARCHND specifications add "
        "OOS value &mdash; the log_artg version dominates at h=1 and h=5 under both "
        "proxies (DM and CW p &lt; 0.001), the tone version at h=1 only, and both "
        "fade or invert at h=22. The other augmented models (simple GARCH-X, GARCHAND, "
        "κ=30% GARCHND) lose to the baseline at every horizon, and EGARCH-X is "
        "roughly tied with the baseline. Checks 2–6 then characterise the in-sample "
        "&gamma; in the surviving κ=50% pair: it is partly redundant with a regime "
        "intercept (Check 2), moderately sensitive to the sigmoid sharpness (Check 3, "
        "CV &asymp; 0.3 across smooth K), driven by ~17–21 high-vol episodes "
        "concentrated in 2020-Q1, 2021-Q1 and 2025-Q4 (Check 4), but robust to HAC "
        "and moving-block-bootstrap corrections (Check 5). Check 6 documents that "
        "the GDELT sentiment series is built from a permissive filter dominated by "
        "the &lsquo;cerium&rsquo; chemistry token, which puts an obvious noise ceiling on the "
        "magnitude of any sentiment effect &mdash; consistent with the conclusion that "
        "the OOS predictive content sits with the volatility-regime switch, not with "
        "raw news intensity.",
        styles["Body"],
    ))
    story.append(Spacer(1, 6))

    render_check1(story, styles)
    story.append(PageBreak())
    render_check2(story, styles)
    story.append(PageBreak())
    render_check3(story, styles)
    story.append(PageBreak())
    render_check4(story, styles)
    story.append(PageBreak())
    render_check5(story, styles)
    story.append(PageBreak())
    render_check6(story, styles)

    doc.build(story)
    print(f"PDF generated: {OUT_PDF}")


if __name__ == "__main__":
    main()
