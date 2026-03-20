from pathlib import Path

import pypdfium2 as pdfium
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output" / "pdf"
TMP_DIR = ROOT / "tmp" / "pdfs"
PDF_PATH = OUTPUT_DIR / "auto-civil-lab-summary.pdf"
PNG_PATH = TMP_DIR / "auto-civil-lab-summary-page-1.png"


TITLE = "AutoCivil-Lab"
SUBTITLE = "Evidence-only app summary from repository files"

WHAT_IT_IS = (
    "AutoCivil-Lab is a Python ML research application for concrete compressive-strength "
    "prediction. It combines data prep, training, governed experiment search, engineering "
    "validation, uncertainty estimation, inverse mix design, and a Flask dashboard around "
    "shared artifacts in outputs/."
)

WHO_ITS_FOR = [
    "Primary persona: Not found in repo.",
    "Inferred from README, config, and module scope: a civil/materials engineering ML "
    "practitioner or researcher working on concrete strength modeling and governed experiments.",
]

WHAT_IT_DOES = [
    "Normalizes workbook/UCI concrete data into data/concrete_data.csv.",
    "Builds engineered mix, binder, replacement-ratio, and age features.",
    "Trains baseline and tuned regressors across multiple model families.",
    "Runs governed scout -> confirm -> keep/revert experiment cycles.",
    "Supports proposal generation through Ollama, OpenAI, or deterministic fallback logic.",
    "Applies engineering validation, uncertainty calibration, reporting, and plotting.",
    "Searches inverse concrete mix designs and serves artifacts through a Flask dashboard.",
]

HOW_IT_WORKS = [
    "Config and strategy: config.yaml defines data paths, metrics, bounds, search spaces, "
    "and llm settings; research_brief.md supplies human strategy and acceptance thresholds.",
    "Data layer: generate_data.py ingests the workbook/UCI source and writes "
    "data/concrete_data.csv; feature_engineering.py adds derived mix features.",
    "Modeling layer: train.py handles split, cross-validation, scoring, persistence, and "
    "baseline artifacts; validator.py adds engineering-rule checks.",
    "Research loop: proposal_engine.py and llm_backend.py generate candidates, research_lab.py "
    "holds the editable research surface, and research_loop.py runs scout/confirm/ratchet flow.",
    "Artifacts and delivery: uncertainty.py, report.py, and design_tool.py write JSON/CSV/PNG "
    "outputs; dashboard.py reads outputs/ and serves them over Flask; render.yaml starts "
    "gunicorn dashboard:app.",
]

HOW_TO_RUN = [
    "pip install -r requirements.txt",
    "Use the default local workbook in config.yaml or point data.local_file.path at your file.",
    "python generate_data.py",
    "python train.py",
    "python research_loop.py --with-report",
    "Optional UI: python dashboard.py  (Windows shortcut: launch_auto_research.bat)",
]

EVIDENCE = (
    "Evidence: README.md, PROJECT_INDEX.md, docs/autoresearch_workflow.md, config.yaml, "
    "dashboard.py, render.yaml, launch_auto_research.bat."
)


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def wrap_text(text: str, width: float, font_name: str, font_size: float) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if pdfmetrics.stringWidth(trial, font_name, font_size) <= width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_wrapped_lines(
    c: canvas.Canvas,
    lines: list[str],
    x: float,
    y: float,
    font_name: str,
    font_size: float,
    leading: float,
    color=colors.HexColor("#1f2937"),
) -> float:
    c.setFont(font_name, font_size)
    c.setFillColor(color)
    cursor = y
    for line in lines:
        c.drawString(x, cursor, line)
        cursor -= leading
    return cursor


def draw_paragraph(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    font_name: str = "Helvetica",
    font_size: float = 8.5,
    leading: float = 11.0,
) -> float:
    return draw_wrapped_lines(
        c,
        wrap_text(text, width, font_name, font_size),
        x,
        y,
        font_name,
        font_size,
        leading,
    )


def draw_bullets(
    c: canvas.Canvas,
    items: list[str],
    x: float,
    y: float,
    width: float,
    font_name: str = "Helvetica",
    font_size: float = 8.15,
    leading: float = 10.2,
) -> float:
    bullet_indent = 10
    text_width = width - bullet_indent
    cursor = y
    c.setFont(font_name, font_size)
    c.setFillColor(colors.HexColor("#1f2937"))

    for item in items:
        wrapped = wrap_text(item, text_width, font_name, font_size)
        for index, line in enumerate(wrapped):
            prefix = "- " if index == 0 else "  "
            c.drawString(x, cursor, prefix + line)
            cursor -= leading
        cursor -= 2
    return cursor


def draw_box(
    c: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    kind: str,
    content,
) -> None:
    border = colors.HexColor("#cbd5e1")
    fill = colors.HexColor("#f8fafc")
    accent = colors.HexColor("#0f766e")
    title_color = colors.HexColor("#0f172a")

    c.setFillColor(fill)
    c.setStrokeColor(border)
    c.roundRect(x, y, width, height, 12, fill=1, stroke=1)
    c.setFillColor(accent)
    c.roundRect(x, y + height - 22, width, 22, 12, fill=1, stroke=0)
    c.setFillColor(accent)
    c.rect(x, y + height - 11, width, 11, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(x + 12, y + height - 14.5, title.upper())

    inner_x = x + 12
    inner_y = y + height - 34
    inner_width = width - 24

    if kind == "paragraph":
        draw_paragraph(c, content, inner_x, inner_y, inner_width)
    else:
        draw_bullets(c, content, inner_x, inner_y, inner_width)

    c.setFillColor(title_color)


def generate_pdf() -> None:
    ensure_dirs()
    page_width, page_height = landscape(letter)
    c = canvas.Canvas(str(PDF_PATH), pagesize=(page_width, page_height))
    c.setTitle("AutoCivil-Lab Summary")
    c.setAuthor("OpenAI Codex")
    c.setSubject("Repository summary")

    margin = 0.42 * inch
    gap = 0.22 * inch
    banner_height = 0.9 * inch
    column_width = (page_width - (2 * margin) - gap) / 2

    bg = colors.HexColor("#eef6f6")
    c.setFillColor(bg)
    c.rect(0, 0, page_width, page_height, fill=1, stroke=0)

    banner_color = colors.HexColor("#0f172a")
    c.setFillColor(banner_color)
    c.roundRect(margin, page_height - margin - banner_height, page_width - 2 * margin, banner_height, 18, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(margin + 18, page_height - margin - 28, TITLE)
    c.setFont("Helvetica", 10)
    c.drawString(margin + 18, page_height - margin - 46, SUBTITLE)
    c.setFont("Helvetica", 8.25)
    c.drawRightString(page_width - margin - 18, page_height - margin - 46, "Repo snapshot date: 2026-03-15")

    left_x = margin
    right_x = margin + column_width + gap

    body_top = page_height - margin - banner_height - 0.16 * inch

    left_boxes = [
        ("What It Is", "paragraph", WHAT_IT_IS, 1.22 * inch),
        ("Who It's For", "bullets", WHO_ITS_FOR, 1.18 * inch),
        ("What It Does", "bullets", WHAT_IT_DOES, 3.17 * inch),
    ]
    right_boxes = [
        ("How It Works", "bullets", HOW_IT_WORKS, 3.42 * inch),
        ("How To Run", "bullets", HOW_TO_RUN, 2.10 * inch),
    ]

    cursor = body_top
    for title, kind, content, height in left_boxes:
        y = cursor - height
        draw_box(c, left_x, y, column_width, height, title, kind, content)
        cursor = y - 0.12 * inch

    cursor = body_top
    for title, kind, content, height in right_boxes:
        y = cursor - height
        draw_box(c, right_x, y, column_width, height, title, kind, content)
        cursor = y - 0.12 * inch

    footer_height = 0.54 * inch
    footer_y = margin
    footer_fill = colors.HexColor("#ffffff")
    footer_border = colors.HexColor("#dbe4ea")
    c.setFillColor(footer_fill)
    c.setStrokeColor(footer_border)
    c.roundRect(margin, footer_y, page_width - 2 * margin, footer_height, 12, fill=1, stroke=1)
    c.setFillColor(colors.HexColor("#475569"))
    c.setFont("Helvetica", 7.6)
    footer_lines = wrap_text(EVIDENCE, page_width - (2 * margin) - 24, "Helvetica", 7.6)
    draw_wrapped_lines(c, footer_lines, margin + 12, footer_y + footer_height - 16, "Helvetica", 7.6, 9.1, colors.HexColor("#475569"))

    c.showPage()
    c.save()


def render_preview() -> None:
    pdf = pdfium.PdfDocument(str(PDF_PATH))
    page = pdf[0]
    bitmap = page.render(scale=2.5)
    image = bitmap.to_pil()
    image.save(PNG_PATH)
    page.close()
    pdf.close()


if __name__ == "__main__":
    generate_pdf()
    render_preview()
    print(PDF_PATH)
    print(PNG_PATH)
