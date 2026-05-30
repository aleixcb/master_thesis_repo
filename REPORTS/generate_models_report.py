"""
PDF report generator for the GARCH/EGARCH volatility models.

Reads the results captured in the REPORTS folder and builds a structured
PDF with one section per model. Each section contains the variance
expression, a short motivation, and the main results (Gaussian first,
Student-t second).

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
# Model metadata: variance equation + motivation (one entry per model section).
# Variance equations use HTML <sub>/<sup> tags so ReportLab's Paragraph
# renders subscripts/superscripts correctly regardless of the font.
# Subsection keys must match the block names extracted from the GP txt files.
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
        # GARCHND has 4 sub-blocks in the captured text (tone/log_art_growth x kappa_low/kappa_high)
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
# Parsing helpers
# --------------------------------------------------------------------------- #
HEADER_RE = re.compile(r"^-{60,}\s*\n(.+?)\n-{60,}", re.MULTILINE)


def split_into_blocks(text: str) -> Dict[str, str]:
    """
    Split a captured txt file into named blocks using its ascii header rule:
        ------------------------------------------------------------
        BLOCK NAME
        ------------------------------------------------------------
    Returns dict {block_name: block_text_below_the_header}.
    """
    blocks: Dict[str, str] = {}
    matches = list(HEADER_RE.finditer(text))
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks[name] = text[start:end].strip()
    return blocks


def parse_parameter_table(block: str) -> Tuple[List[List[str]], List[str]]:
    """
    Extract the per-parameter significance table inside a block.

    Returns:
        rows: list of [name, estimate, std_err, t_ratio, p_value, sig]
        notes: list of trailing text lines (LL/AIC/BIC/extra notes)
    """
    rows: List[List[str]] = []
    notes: List[str] = []
    lines = block.splitlines()

    # Locate the parameter table by its header line
    i = 0
    while i < len(lines):
        if "estimate" in lines[i] and "t_ratio" in lines[i] and "p_value" in lines[i]:
            i += 1
            break
        i += 1

    # Read data rows until a blank line or a "Log-likelihood" footer
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln.strip():
            break
        if ln.lstrip().startswith(("Log-likelihood", "AIC", "BIC")):
            break
        parts = ln.split()
        # Expected >= 8 columns: name est std_err t p sig10 sig5 sig1
        if len(parts) >= 8:
            name = parts[0]
            estimate = parts[1]
            std_err = parts[2]
            t_ratio = parts[3]
            p_value = parts[4]
            sigs = parts[5:8]  # 10%, 5%, 1%
            stars = sig_stars(sigs)
            rows.append([name, estimate, std_err, t_ratio, p_value, stars])
        i += 1

    # Collect remaining footer lines (LL/AIC/BIC/extras) up to the "Verdict:" section
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("Verdict:"):
            break
        if ln and not ln.startswith("--"):
            notes.append(ln)
        i += 1

    return rows, notes


def sig_stars(sig_flags: List[str]) -> str:
    """Convert ['True','True','True'] significance flags to a *** marker."""
    n_true = sum(1 for f in sig_flags if f.lower() == "true")
    if n_true == 3:
        return "***"
    if n_true == 2:
        return "**"
    if n_true == 1:
        return "*"
    return ""


def parse_garchnd_subblocks(block: str) -> List[Tuple[str, List[List[str]], List[str]]]:
    """
    The GARCHND block in the txt contains 4 sub-runs separated by
    `---- GARCHND  x = ... kappa = ... % annual ----`.
    Returns a list of (subtitle, rows, notes).
    """
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
    """The 04-extension files only contain one big block — no headers."""
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
    # Variance equation: use a regular (non-monospace) font so HTML
    # <sub>/<sup> tags render with proper baseline shifts and so all
    # Greek letters/entities display consistently.
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
    return styles


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
    """Append a Gaussian/Student-t subsection to the story."""
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
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    # Read files ------------------------------------------------------------
    gp_g_blocks = split_into_blocks(FILES["gp_gaussian"].read_text())
    gp_s_blocks = split_into_blocks(FILES["gp_student"].read_text())

    ext_g_rows, ext_g_notes = parse_extension_file(FILES["ext_gaussian"])
    ext_s_rows, ext_s_notes = parse_extension_file(FILES["ext_student"])

    # Build PDF -------------------------------------------------------------
    styles = build_styles()
    doc = SimpleDocTemplate(
        str(OUT_PDF), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="Volatility models — REMX",
    )
    story: list = []

    # --- Cover -------------------------------------------------------------
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
        "*** = 1%, ** = 5%, * = 10%.",
        styles["Body"],
    ))
    story.append(Spacer(1, 10))

    # --- One section per model --------------------------------------------
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

    doc.build(story)
    print(f"PDF generated: {OUT_PDF}")


if __name__ == "__main__":
    main()
