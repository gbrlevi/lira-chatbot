"""Turns the policy PDF into Markdown and section-sized chunks.

Tables are extracted separately and placed back at their original position, because plain
text extraction detaches each weekday from its opening hours.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from pdfplumber.page import Page

SECTION_FONT_SIZE = 14
SUBSECTION_FONT_SIZE = 12
# Measured on the PDF: lines of a paragraph are 5.3pt apart, list items 6.4pt and separate
# paragraphs at least 8.5pt, so anything above this value starts a new paragraph.
PARAGRAPH_GAP = 7.5

_PAGE_FURNITURE = re.compile(
    r"^(Empório da Música Manual de Políticas e Procedimentos"
    r"|Página \d+"
    r"|Empório da Música — Sua música começa aqui\."
    r"|Última atualização: .*)$"
)
_NEW_BLOCK = re.compile(r"^(•|\d+\.\s)")
_LIST_ITEM = re.compile(r"^(- |\d+\.\s)")
_SECTION_HEADING = re.compile(r"^## (\d+)\. (.+)$")
_SUBSECTION_HEADING = re.compile(r"^### (\d+\.\d+) (.+)$")


@dataclass(frozen=True)
class PolicyChunk:
    section: str
    title: str
    text: str

    @property
    def searchable_text(self) -> str:
        """Includes the title, so a query can match words that only appear in the heading."""
        return f"Seção {self.section} — {self.title}\n{self.text}"


def _table_to_markdown(rows: list[list[str | None]]) -> str:
    cells = [[" ".join((cell or "").split()) for cell in row] for row in rows]
    header, *body = cells
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + " --- |" * len(header),
        *("| " + " | ".join(row) + " |" for row in body),
    ]
    return "\n".join(lines)


def _is_bold(line: dict) -> bool:
    return "Bold" in line["chars"][0]["fontname"]


def _heading_prefix(line: dict) -> str | None:
    # Sizes come back as floats such as 13.9999995.
    size = round(line["chars"][0]["size"])
    if not _is_bold(line):
        return None
    if size == SECTION_FONT_SIZE:
        return "## "
    if size == SUBSECTION_FONT_SIZE:
        return "### "
    return None


def _inside(line: dict, bbox: tuple[float, float, float, float]) -> bool:
    _, top, _, bottom = bbox
    return top <= (line["top"] + line["bottom"]) / 2 <= bottom


def _page_items(page: Page) -> list[tuple[float, str, dict | str]]:
    tables = page.find_tables()
    items: list[tuple[float, str, dict | str]] = [
        (table.bbox[1], "table", _table_to_markdown(table.extract())) for table in tables
    ]
    for line in page.extract_text_lines():
        if _PAGE_FURNITURE.match(line["text"].strip()):
            continue
        if any(_inside(line, table.bbox) for table in tables):
            continue
        items.append((line["top"], "line", line))
    return sorted(items, key=lambda item: item[0])


def extract_markdown(pdf_path: Path) -> str:
    """Extract the manual as Markdown, skipping the cover (its data repeats in section 1.2)."""
    blocks: list[str] = []
    previous_bottom: float | None = None

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[1:]:
            previous_bottom = None  # no paragraph in the manual continues across pages
            for _, kind, content in _page_items(page):
                if kind == "table":
                    blocks.append(content)
                    previous_bottom = None
                    continue

                text = content["text"].strip()
                prefix = _heading_prefix(content)
                if prefix:
                    blocks.append(prefix + text)
                    previous_bottom = None
                    continue

                starts_block = (
                    _NEW_BLOCK.match(text)
                    or _is_bold(content)
                    or (
                        previous_bottom is not None
                        and content["top"] - previous_bottom > PARAGRAPH_GAP
                    )
                )
                last_is_text = bool(blocks) and not blocks[-1].startswith(("#", "|"))
                if starts_block or not last_is_text or previous_bottom is None:
                    blocks.append(text.replace("• ", "- ", 1) if text.startswith("•") else text)
                else:
                    blocks[-1] = f"{blocks[-1]} {text}"
                previous_bottom = content["bottom"]

    return _join_blocks(blocks)


def _join_blocks(blocks: list[str]) -> str:
    """List items stay on consecutive lines; everything else is separated by a blank line."""
    output = ""
    for index, block in enumerate(blocks):
        if index:
            both_list = block.startswith(("- ", "1.", "2.", "3.", "4.", "5.")) and blocks[
                index - 1
            ].startswith(("- ", "1.", "2.", "3.", "4.", "5."))
            output += "\n" if both_list else "\n\n"
        output += block
    return output + "\n"


def chunk_markdown(markdown: str) -> list[PolicyChunk]:
    """One chunk per numbered subsection, plus one for text directly under each section."""
    chunks: list[PolicyChunk] = []
    section_number = ""
    section_title = ""
    current_id = ""
    current_title = ""
    body: list[str] = []

    def flush() -> None:
        text = "\n".join(body).strip()
        if current_id and text:
            chunks.append(PolicyChunk(section=current_id, title=current_title, text=text))
        body.clear()

    for line in markdown.splitlines():
        section = _SECTION_HEADING.match(line)
        subsection = _SUBSECTION_HEADING.match(line)
        if section:
            flush()
            section_number, section_title = section.groups()
            current_id, current_title = section_number, section_title
        elif subsection:
            flush()
            current_id = subsection.group(1)
            current_title = f"{section_title}: {subsection.group(2)}"
        else:
            body.append(line)
    flush()
    return chunks
