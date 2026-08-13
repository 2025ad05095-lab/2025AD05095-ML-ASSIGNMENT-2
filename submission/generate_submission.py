"""Generate the single PDF required for ML Assignment 2 submission."""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
GITHUB_URL = "https://github.com/2025ad05095-lab/2025AD05095-ML-ASSIGNMENT-2"
APP_URL = "https://2025ad05095-ml-assignment-2-n2n9yb6qu5cuqyxacft9r9.streamlit.app/"


def clean_inline(text: str) -> str:
    text = text.replace("**", "").replace("`", "")
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def markdown_story(markdown: str, styles) -> list:
    story = []
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or line.startswith("```"):
            index += 1
            continue
        if line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [clean_inline(cell.strip()) for cell in lines[index].strip().strip("|").split("|")]
                if not all(set(cell) <= {"-", ":"} for cell in cells):
                    rows.append([Paragraph(cell, styles["TableText"]) for cell in cells])
                index += 1
            table = Table(rows, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDEBE9")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#78909C")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.extend([table, Spacer(1, 8)])
            continue
        if line.startswith("# "):
            story.append(Paragraph(clean_inline(line[2:]), styles["Title"]))
        elif line.startswith("## "):
            story.append(Paragraph(clean_inline(line[3:]), styles["Heading2"]))
        elif line.startswith("- "):
            rendered = clean_inline(line[2:])
            if "GitHub repository:" in rendered:
                rendered = f'GitHub repository: <link href="{GITHUB_URL}" color="blue">{GITHUB_URL}</link>'
            elif "Live Streamlit application:" in rendered:
                rendered = f'Live Streamlit application: <link href="{APP_URL}" color="blue">{APP_URL}</link>'
            story.append(Paragraph(f"• {rendered}", styles["BodyText"]))
        elif line.startswith("|") or line.startswith("`"):
            pass
        else:
            story.append(Paragraph(clean_inline(line), styles["BodyText"]))
        story.append(Spacer(1, 4))
        index += 1
    return story


def generate(screenshot: Path | None) -> Path:
    has_screenshot = screenshot is not None and screenshot.is_file()
    filename = "2025AD05095_ML_Assignment_2_Submission.pdf" if has_screenshot else "2025AD05095_ML_Assignment_2_Submission_DRAFT.pdf"
    output = ROOT / "submission" / filename
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=22, leading=28, spaceAfter=18))
    styles.add(ParagraphStyle(name="TableText", parent=styles["BodyText"], fontSize=7.2, leading=9))
    document = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42, title="ML Assignment 2 Submission - 2025AD05095")
    story = [
        Spacer(1, 0.6 * inch),
        Paragraph("Machine Learning Assignment 2", styles["CoverTitle"]),
        Paragraph("Classification Models and Streamlit Deployment", styles["Heading2"]),
        Spacer(1, 0.25 * inch),
        Paragraph("Student ID: 2025AD05095", styles["BodyText"]),
        Paragraph("Programme: M.Tech (AIML/DSE)", styles["BodyText"]),
        Paragraph("Course: Machine Learning", styles["BodyText"]),
        Spacer(1, 0.35 * inch),
        Paragraph("1. GitHub Repository Link", styles["Heading2"]),
        Paragraph(f'<link href="{GITHUB_URL}" color="blue">{GITHUB_URL}</link>', styles["BodyText"]),
        Spacer(1, 0.2 * inch),
        Paragraph("2. Live Streamlit App Link", styles["Heading2"]),
        Paragraph(f'<link href="{APP_URL}" color="blue">{APP_URL}</link>', styles["BodyText"]),
        PageBreak(),
        Paragraph("3. BITS Virtual Lab Execution Screenshot", styles["Heading2"]),
    ]
    if has_screenshot:
        image = Image(str(screenshot))
        image._restrictSize(7.0 * inch, 8.8 * inch)
        story.extend([Spacer(1, 0.15 * inch), image])
    else:
        warning = Table([[Paragraph("REQUIRED BEFORE SUBMISSION<br/><br/>Add one screenshot showing assignment execution on the authenticated BITS Virtual Lab, then regenerate this PDF.", styles["BodyText"])]], colWidths=[6.8 * inch], rowHeights=[3.0 * inch])
        warning.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 2, colors.red), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF4F4"))]))
        story.extend([Spacer(1, 0.4 * inch), warning])
    story.extend([PageBreak(), Paragraph("4. GitHub README Content", styles["Heading2"]), Spacer(1, 8)])
    story.extend(markdown_story((ROOT / "README.md").read_text(encoding="utf-8"), styles))
    document.build(story)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot", type=Path, help="BITS Virtual Lab execution screenshot")
    arguments = parser.parse_args()
    result = generate(arguments.screenshot)
    print(result)
