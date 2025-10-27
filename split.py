#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    from PyPDF2 import PdfReader, PdfWriter  # type: ignore
except ImportError as exc:  # pragma: no cover
    print(
        "未找到 PyPDF2，请运行 `python3 -m pip install PyPDF2` 后重试。",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


INPUT_PDF = Path("tmp/book.pdf")
OUTPUT_DIR = Path("tmp/split")
TARGET_LEVELS = {0, 1, 2}  # chapter -> 0, section -> 1, subsection -> 2
TOC_PATH = Path("tmp/book.toc")


def _flatten_outline(entries: Sequence, level: int = 0) -> Iterable[Tuple[int, object]]:
    for item in entries:
        if isinstance(item, list):
            yield from _flatten_outline(item, level + 1)
        else:
            yield level, item


def _collect_from_outline(reader: PdfReader) -> List[Tuple[int, int, str]]:
    outline = getattr(reader, "outline", None)
    if not outline:
        return []

    collected: List[Tuple[int, int, str]] = []
    for level, destination in _flatten_outline(outline):
        try:
            start = reader.get_destination_page_number(destination)
        except Exception:
            continue
        title = getattr(destination, "title", "") or "untitled"
        collected.append((level, start, title))
    return collected


TOC_KIND_LEVEL: Dict[str, int] = {
    "chapter": 0,
    "section": 1,
    "subsection": 2,
}


def _collect_from_toc(total_pages: int) -> List[Tuple[int, int, str]]:
    if not TOC_PATH.exists():
        return []

    entries: List[Tuple[int, int, str]] = []
    pattern = re.compile(r"\\contentsline {(chapter|section|subsection)}")
    with TOC_PATH.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            match = pattern.search(raw_line)
            if not match:
                continue
            kind = match.group(1)
            level = TOC_KIND_LEVEL[kind]
            line = raw_line.strip()
            try:
                before_page, rest = line.rsplit("}{", 1)
                page_text, _ = rest.split("}", 1)
                page = int(page_text) - 1  # convert to 0-based
            except ValueError:
                continue

            if page < 0 or page >= total_pages:
                continue

            try:
                _, remainder = before_page.split("{\\numberline ", 1)
            except ValueError:
                continue
            if remainder.startswith("{"):
                remainder = remainder[1:]
            number_part, _, title_part = remainder.partition("}")
            number = number_part.strip()
            title = title_part.strip()
            merged_title = f"{number} {title}".strip()
            if not merged_title:
                merged_title = number or title or "subsection"
            entries.append((level, page, merged_title))
    return entries


def _sanitize_for_filename(text: str) -> str:
    clean = re.sub(r"[\\/*?:\"<>|]", "_", text.strip())
    clean = re.sub(r"\s+", " ", clean)
    return clean or "subsection"


def _compute_ranges(
    entries: List[Tuple[int, int, str]], total_pages: int
) -> List[Tuple[int, int, int, str]]:
    if not entries:
        return []

    sorted_entries = sorted(entries, key=lambda item: (item[1], item[0]))
    ranges: List[Tuple[int, int, int, str]] = []
    for index, (level, start, title) in enumerate(sorted_entries):
        if start >= total_pages:
            continue
        end = total_pages
        for next_level, next_start, _ in sorted_entries[index + 1 :]:
            if next_start <= start:
                continue
            if next_level <= level:
                end = next_start
                break
        if end <= start:
            end = min(total_pages, start + 1)
        ranges.append((level, start, end, title))
    return ranges


def _level_name(level: int) -> str:
    if level == 0:
        return "chapter"
    if level == 1:
        return "section"
    if level == 2:
        return "subsection"
    return f"level{level}"


def _select_output_ranges(
    ranges: List[Tuple[int, int, int, str]],
) -> List[Tuple[int, int, int, str]]:
    if not ranges:
        return []

    primary = [item for item in ranges if item[0] in TARGET_LEVELS]
    if not primary:
        fallback = list(ranges)
        fallback.sort(key=lambda item: (item[1], item[0]))
        return fallback

    primary.sort(key=lambda item: (item[1], item[0]))
    return primary


def split_pdf() -> None:
    if not INPUT_PDF.exists():
        raise FileNotFoundError(f"未找到输入文件：{INPUT_PDF}")

    with INPUT_PDF.open("rb") as pdf_stream:
        reader = PdfReader(pdf_stream)
        total_pages = len(reader.pages)
        entries = _collect_from_outline(reader)
        if not entries:
            entries = _collect_from_toc(total_pages)
        if not entries:
            raise RuntimeError("无法找到任何小节信息，确认 PDF 启用了书签或目录。")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for existing in OUTPUT_DIR.glob("*.pdf"):
            existing.unlink()

        ranges = _compute_ranges(entries, total_pages)
        filtered = _select_output_ranges(ranges)

        for seq, (level, start, end, title) in enumerate(filtered, start=1):
            writer = PdfWriter()
            for page_number in range(start, end):
                writer.add_page(reader.pages[page_number])
            level_label = _level_name(level)
            filename = f"{seq:03d}-{level_label}-{_sanitize_for_filename(title)}.pdf"
            output_path = OUTPUT_DIR / filename
            with output_path.open("wb") as pdf_out:
                writer.write(pdf_out)
            print(f"导出 {filename}：第 {start + 1} - {end} 页")


if __name__ == "__main__":
    split_pdf()
