"""Conservative text cleaning and token-bounded, section-aware Markdown chunks."""

from dataclasses import dataclass
from hashlib import sha256
import re
import unicodedata


PIPELINE_VERSION = "v2-nfc-sections-1"


class PreprocessingError(ValueError):
    pass


@dataclass(frozen=True)
class CleanedText:
    text: str
    removed_lines: int
    duplicate_blocks: int


@dataclass(frozen=True)
class TextChunk:
    content: str
    metadata: dict
    source_page: int


def clean_text(raw: str, *, boilerplate_lines=()) -> CleanedText:
    # NFC preserves math symbols, unlike compatibility normalization (NFKC).
    text = unicodedata.normalize("NFC", raw).replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate({ord(c): None for c in "\ufeff\u200b\u00ad"}).replace("\xa0", " ")
    pages = text.split("\f")
    output, removed, duplicates = [], 0, 0
    explicit = {line.strip() for line in boilerplate_lines}
    for page in pages:
        lines = page.splitlines()
        nonempty = [i for i, line in enumerate(lines) if line.strip()]
        kept = []
        fenced = None
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(("```", "~~~")):
                marker = stripped[:3]
                fenced = None if fenced == marker else marker
                kept.append(line)
                continue
            if fenced:
                kept.append(line)
                continue
            edge = bool(nonempty) and index in (nonempty[0], nonempty[-1])
            page_number = len(pages) > 1 and edge and re.fullmatch(r"(?:第\s*\d+\s*页(?:\s*[/／]\s*共?\s*\d+\s*页?)?|\d+)", stripped)
            if stripped and (stripped in explicit or page_number):
                removed += 1
                continue
            indentation = line[:len(line) - len(line.lstrip(" \t"))]
            kept.append(indentation + re.sub(r"[ \t]+", " ", line.lstrip(" \t")).rstrip())
        # Only adjacent, identical prose paragraphs are deduplicated. Repeated
        # examples in another section, formulas, tables and code remain intact.
        protected = {}
        def protect(match):
            key = f"\x00CODE{len(protected)}\x00"
            protected[key] = match[0]
            return "\n\n" + key + "\n\n"
        page_text = re.sub(r"(?m)^[ \t]*(`{3,}|~{3,})[^\n]*\n[\s\S]*?^[ \t]*\1[ \t]*$", protect, "\n".join(kept).strip())
        paragraphs = re.split(r"\n\s*\n", page_text)
        deduped = []
        for paragraph in paragraphs:
            atomic = any(mark in paragraph for mark in ("```", "~~~", "$$", "\\[", "|"))
            if (deduped and paragraph == deduped[-1] and not atomic
                    and not paragraph.lstrip().startswith("#")):
                duplicates += 1
            elif paragraph:
                deduped.append(paragraph)
        page_text = "\n\n".join(deduped)
        for key, value in protected.items():
            page_text = page_text.replace(key, value)
        output.append(page_text)
    return CleanedText("\f".join(output).strip(), removed, duplicates)


def _blocks(page):
    lines = page.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            yield "heading", len(heading[1]), heading[2]
            index += 1
            continue
        start = index
        marker = line[:3] if line.startswith(("```", "~~~")) else None
        math_end = "$$" if line == "$$" else ("\\]" if line == "\\[" else None)
        if marker or math_end:
            index += 1
            while index < len(lines) and not (lines[index].strip().startswith(marker) if marker else lines[index].strip() == math_end):
                index += 1
            if index == len(lines):
                raise PreprocessingError("代码块或公式块缺少结束标记。")
            index += 1
            yield "atomic", 0, "\n".join(lines[start:index])
            continue
        table = "|" in line
        while index < len(lines) and lines[index].strip():
            if index > start and re.match(r"^#{1,6}\s", lines[index]):
                break
            index += 1
        block = "\n".join(lines[start:index])
        atomic = table or "$$" in block or "\\[" in block or bool(re.match(r"^(例题|题目|练习题)[:：\s]", block))
        yield "atomic" if atomic else "prose", 0, block


def split_text(title, raw, count_tokens, *, max_tokens=448, overlap_tokens=48, boilerplate_lines=()):
    """Count the final content, including heading breadcrumbs and special tokens.

    Atomic formula/table/example blocks that exceed the budget are rejected,
    never silently truncated. Only the last complete prose block can overlap.
    """
    if max_tokens < 8 or not 0 <= overlap_tokens < max_tokens:
        raise PreprocessingError("无效的切分 token 上限或重叠配置。")
    cleaned = clean_text(raw, boilerplate_lines=boilerplate_lines)
    if not cleaned.text:
        raise PreprocessingError("清洗后正文为空。")
    chunks = []
    section = []
    raw_hash = sha256(raw.encode()).hexdigest()
    for page_number, page in enumerate(cleaned.text.split("\f"), 1):
        pending = []
        def prefix():
            return "\n".join([f"# {title}"] + [f"{'#' * (level + 1)} {name}" for level, name in section]) + "\n\n"
        def emit():
            if not pending:
                return
            content = prefix() + "\n\n".join(value for _, value in pending)
            chunks.append(TextChunk(content, {
                "pipeline": PIPELINE_VERSION, "section_path": [name for _, name in section],
                "token_count": count_tokens(content), "max_tokens": max_tokens,
                "raw_sha256": raw_hash, "cleaned_sha256": sha256(cleaned.text.encode()).hexdigest(),
                "removed_lines": cleaned.removed_lines, "duplicate_blocks": cleaned.duplicate_blocks,
            }, page_number))
        for kind, level, block in _blocks(page):
            if kind == "heading":
                emit()
                pending = []
                section = [(old_level, name) for old_level, name in section if old_level < level]
                section.append((level, block))
                continue
            if count_tokens(prefix() + block) > max_tokens:
                if kind == "atomic":
                    raise PreprocessingError("单个公式、表格或例题超过 token 上限，请按完整语义拆分源资料。")
                # Split ordinary prose at sentence boundaries first, then by
                # character for a single long sentence. Every character survives.
                parts, part = [], ""
                units = re.findall(r".*?[。！？!?；;\n]|.+$", block)
                for unit in units:
                    if part and count_tokens(prefix() + part + unit) > max_tokens:
                        parts.append(part)
                        part = ""
                    if count_tokens(prefix() + unit) <= max_tokens:
                        part += unit
                        continue
                    for char in unit:
                        if count_tokens(prefix() + part + char) > max_tokens:
                            if not part:
                                raise PreprocessingError("标题过长，无法容纳正文。")
                            parts.append(part)
                            part = ""
                        part += char
                    # Prefer whole sentences whenever another sentence does not fit.
                    if part and count_tokens(prefix() + part) > max_tokens - overlap_tokens:
                        parts.append(part)
                        part = ""
                if part:
                    parts.append(part)
            else:
                parts = [block]
            for part in parts:
                if pending and count_tokens(prefix() + "\n\n".join([v for _, v in pending] + [part])) > max_tokens:
                    tail = pending[-1]
                    emit()
                    pending = []
                    if (tail[0] == "prose" and count_tokens(tail[1]) <= overlap_tokens
                            and count_tokens(prefix() + tail[1] + "\n\n" + part) <= max_tokens):
                        pending.append(tail)
                pending.append((kind, part))
        emit()
    if not chunks:
        raise PreprocessingError("正文只有标题，没有可入库内容。")
    return chunks
