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

# Check 1 — OOS forecasting (notebook 06, cells 5–6)
OOS_LOSS = [
    # [Model, h, QLIKE, MSE]
    ["GARCH(1,1) Student-t (base)",    "1",  "2.4225", "48.0561"],
    ["GARCH(1,1) Student-t (base)",    "5",  "2.4561", "38.7524"],
    ["GARCH(1,1) Student-t (base)",   "22",  "2.3900", "29.7071"],
    ["GARCHND tone κ=50%",             "1",  "2.4397", "49.2963"],
    ["GARCHND tone κ=50%",             "5",  "2.4792", "41.8515"],
    ["GARCHND tone κ=50%",            "22",  "2.4417", "39.1974"],
    ["GARCHND log_artg κ=50%",         "1",  "2.4167", "43.2401"],
    ["GARCHND log_artg κ=50%",         "5",  "2.4561", "35.6411"],
    ["GARCHND log_artg κ=50%",        "22",  "2.4229", "33.9900"],
]

OOS_DM = [
    # [Candidate, h, DM-QLIKE, p, DM-MSE, p]
    ["GARCHND tone κ=50%",      "1",  "−6.262", "0.0000", "−3.150", "0.0016"],
    ["GARCHND tone κ=50%",      "5",  "−4.882", "0.0000", "−4.013", "0.0001"],
    ["GARCHND tone κ=50%",     "22",  "−6.267", "0.0000", "−3.409", "0.0007"],
    ["GARCHND log_artg κ=50%",  "1",  "+1.287", "0.1982", "+5.153", "0.0000"],
    ["GARCHND log_artg κ=50%",  "5",  "+0.003", "0.9975", "+1.831", "0.0671"],
    ["GARCHND log_artg κ=50%", "22",  "−2.496", "0.0125", "−1.969", "0.0490"],
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
    ["K=5",      "-5952.63", "-0.3626", "0.0737",  "-4.92", "True"],
    ["K=10",     "-5952.42", "-0.3769", "0.0740",  "-5.09", "True"],
    ["K=20",     "-5951.97", "-0.8262", "0.0602", "-13.72", "True"],
    ["K=50",     "-5952.16", "-0.3880", "0.0008", "-468.3", "True"],
    ["hard d3",  "-5953.24", "-0.3191", "0.0989",  "-3.23", "True"],
]

STABILITY_ARTG = [
    ["K=5",      "-5951.98", "-1.3781", "0.2739",  "-5.03", "True"],
    ["K=10",     "-5952.01", "-1.0979", "0.2493",  "-4.40", "True"],
    ["K=20",     "-5951.25", "-1.4116", "0.2184",  "-6.46", "True"],
    ["K=50",     "-5951.42", "-1.1276", "0.0405", "-27.81", "True"],
    ["hard d3",  "-5951.52", "-1.0767", "0.0029", "-370.8", "True"],
]

# Check 4 — Cluster count (notebook 06, cell 12)
CLUSTERS_TONE = [
    ["1", "2021-01-25", "2021-04-06", "50"],
    ["2", "2026-02-03", "2026-03-31", "40"],
    ["3", "2020-03-04", "2020-04-22", "35"],
    ["4", "2025-10-14", "2025-11-14", "24"],
    ["5", "2025-04-11", "2025-04-28", "11"],
    ["6", "2018-12-24", "2019-01-08", "10"],
    ["7", "2024-09-30", "2024-10-11", "10"],
    ["8", "2022-05-11", "2022-05-23",  "9"],
    ["9", "2022-11-08", "2022-11-18",  "9"],
    ["10","2022-06-03", "2022-06-13",  "7"],
]

CLUSTERS_ARTG = [
    ["1", "2021-02-04", "2021-04-06", "42"],
    ["2", "2020-03-04", "2020-04-29", "40"],
    ["3", "2025-10-15", "2025-11-12", "21"],
    ["4", "2026-03-05", "2026-03-31", "19"],
    ["5", "2021-01-07", "2021-01-25", "12"],
    ["6", "2026-02-03", "2026-02-19", "12"],
    ["7", "2024-09-30", "2024-10-14", "11"],
    ["8", "2018-12-24", "2019-01-08", "10"],
    ["9", "2022-05-11", "2022-05-23",  "9"],
    ["10","2022-06-03", "2022-06-15",  "9"],
]

# Check 5 — HAC/MBB SEs (notebook 07, cell 3)
HAC_TONE = [
    # [Method, gamma, SE(gamma), t(gamma)]
    ["iid sandwich", "-0.8262", "0.0602", "-13.72"],
    ["HAC L=25",     "-0.8262", "0.0586", "-14.09"],
    ["HAC L=50",     "-0.8262", "0.0548", "-15.07"],
    ["HAC L=75",     "-0.8262", "0.0565", "-14.62"],
    ["HAC L=100",    "-0.8262", "0.0578", "-14.30"],
    ["MBB L=25",     "-0.8262", "0.0583", "-14.18"],
    ["MBB L=50",     "-0.8262", "0.0563", "-14.68"],
    ["MBB L=75",     "-0.8262", "0.0575", "-14.36"],
    ["MBB L=100",    "-0.8262", "0.0596", "-13.87"],
]

HAC_ARTG = [
    ["iid sandwich", "-1.4116", "0.2184", "-6.46"],
    ["HAC L=25",     "-1.4116", "0.2321", "-6.08"],
    ["HAC L=50",     "-1.4116", "0.2424", "-5.82"],
    ["HAC L=75",     "-1.4116", "0.2469", "-5.72"],
    ["HAC L=100",    "-1.4116", "0.2473", "-5.71"],
    ["MBB L=25",     "-1.4116", "0.2360", "-5.98"],
    ["MBB L=50",     "-1.4116", "0.2455", "-5.75"],
    ["MBB L=75",     "-1.4116", "0.2471", "-5.71"],
    ["MBB L=100",    "-1.4116", "0.2503", "-5.64"],
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
    story.append(Paragraph("Check 1 — Out-of-sample forecasting (h = 1, 5, 22 days)", styles["ModelTitle"]))
    story.append(Paragraph(
        "In-sample fit through 2025-03-31 (2,514 obs). OOS window: 2025-04-01 to "
        "2026-03-31 (251 obs). Baseline is GARCH(1,1) Student-t. "
        "DM &gt; 0 means the candidate beats the baseline; p-values are two-sided.",
        styles["Body"],
    ))

    story.append(Paragraph("Loss functions (mean over OOS)", styles["SubSection"]))
    hdr = ["Model", "h", "QLIKE", "MSE"]
    cw = [7.5 * cm, 1.2 * cm, 2.8 * cm, 2.8 * cm]
    story.append(_tbl([hdr] + OOS_LOSS, cw))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Diebold-Mariano test (baseline vs candidate)", styles["SubSection"]))
    hdr2 = ["Candidate", "h", "DM-QLIKE", "p", "DM-MSE", "p"]
    cw2 = [5.5 * cm, 1.0 * cm, 2.2 * cm, 2.0 * cm, 2.2 * cm, 1.4 * cm]
    story.append(_tbl([hdr2] + OOS_DM, cw2))
    story.append(Paragraph(
        "Verdict: neither GARCHND specification improves OOS forecasting. "
        "GARCHND tone κ=50% is dominated by the baseline across all horizons and loss functions. "
        "GARCHND log_artg κ=50% shows marginal MSE gains at h=1 but no QLIKE gain.",
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

    story.append(Paragraph("x = tone  (CV = 0.412 — sensitive to K)", styles["SubSection"]))
    story.append(_tbl([hdr] + STABILITY_TONE, cw))
    story.append(Spacer(1, 8))

    story.append(Paragraph("x = log_art_growth  (CV = 0.119 — stable)", styles["SubSection"]))
    story.append(_tbl([hdr] + STABILITY_ARTG, cw))
    story.append(Paragraph(
        "Verdict: the tone specification is sensitive to the sigmoid shape; "
        "log_art_growth is stable in sign and magnitude but not in t-ratio (t ranges from &minus;4.4 to &minus;370.8).",
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
        "x = tone: 193 triggered days (6.98%), 21 distinct episodes; "
        "recommended block length &ge; 50 days",
        styles["SubSection"],
    ))
    story.append(_tbl([hdr] + CLUSTERS_TONE, cw))
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "x = log_art_growth: 179 triggered days (6.47%), 20 distinct episodes; "
        "recommended block length &ge; 42 days",
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
        "Verdict: HAC and MBB corrections do NOT inflate the standard errors materially. "
        "The t-ratios remain strongly significant (tone: around &minus;14; log_artg: around &minus;5.7). "
        "This is explained by the iid nature of x<sup>2</sup><sub>t-1</sub> within the regime, "
        "which absorbs serial correlation in the score. The conclusion from Checks 1&ndash;2 "
        "(regime artefact, no OOS gain) stands unaffected.",
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
        "Six diagnostic checks applied to the GARCHND κ=50% Student-t models "
        "(the best in-sample specifications). Checks 1–4 are from notebook 06; "
        "Checks 5–6 from notebook 07. All checks point to the same conclusion: "
        "the in-sample significance of &gamma; is a regime artefact with no "
        "out-of-sample value, and the sentiment variable was measuring global "
        "chemistry noise rather than REMX-specific information.",
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
