#!/usr/bin/env python3
"""Build a linked PDF archive from a readpst -e style folder tree.

Supported input files:
  - .eml: mail (attachments and images are ignored)
  - .ics: calendar items
  - .vcf: contacts

The PDF contains:
  - a folder-only archive table of contents;
  - one chapter index per folder, with links to its direct items;
  - one variable-height PDF page per item;
  - internal breadcrumb/index links, a local-folder link, and a footer link
    back to the first page.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import os
import quopri
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from email import policy
from email.parser import BytesHeaderParser, BytesParser
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Iterator, Sequence

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfgen.canvas import Canvas
except ImportError as exc:  # pragma: no cover - dependency check
    print(
        "Missing ReportLab. On Arch Linux install it with:\n"
        "  sudo pacman -S python-reportlab",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

SUPPORTED_EXTENSIONS = {".eml", ".ics", ".vcf"}
A4_WIDTH, A4_HEIGHT = A4
PDF_SAFE_MAX_HEIGHT = 5000 * mm  # just below the traditional 14,400 pt limit

INK = colors.HexColor("#18212B")
MUTED = colors.HexColor("#64717D")
LINK = colors.HexColor("#165D9C")
RULE = colors.HexColor("#D7DDE3")
PANEL = colors.HexColor("#F4F6F8")
PANEL_DARK = colors.HexColor("#E8EDF2")
WHITE = colors.white

# ReportLab cannot embed the common CFF-based Noto CJK TTC files. Standard
# PDF CID fonts provide a viewer-supported fallback for CJK text while the
# selected TTF remains embedded for Latin, Greek, Cyrillic, and similar text.
_FONT_FALLBACKS: dict[str, tuple[str | None, str | None]] = {}


def is_hangul(char: str) -> bool:
    code = ord(char)
    return (
        0x1100 <= code <= 0x11FF
        or 0x3130 <= code <= 0x318F
        or 0xA960 <= code <= 0xA97F
        or 0xAC00 <= code <= 0xD7AF
        or 0xD7B0 <= code <= 0xD7FF
    )


def is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x2E80 <= code <= 0x303F
        or 0x3040 <= code <= 0x30FF
        or 0x3100 <= code <= 0x31FF
        or 0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0xFF65 <= code <= 0xFF9F
        or 0x20000 <= code <= 0x2FA1F
    )


def resolved_font(base_font: str, char: str) -> str:
    cjk_font, korean_font = _FONT_FALLBACKS.get(base_font, (None, None))
    if is_hangul(char) and korean_font:
        return korean_font
    if is_cjk(char) and cjk_font:
        return cjk_font
    return base_font


def font_runs(text: str, base_font: str) -> list[tuple[str, str]]:
    if not text:
        return [(base_font, "")]
    runs: list[tuple[str, str]] = []
    current_font = resolved_font(base_font, text[0])
    current: list[str] = []
    for char in text:
        char_font = resolved_font(base_font, char)
        if char_font != current_font and current:
            runs.append((current_font, "".join(current)))
            current = []
            current_font = char_font
        current.append(char)
    runs.append((current_font, "".join(current)))
    return runs


def natural_key(value: str) -> tuple[object, ...]:
    return tuple(int(p) if p.isdigit() else p.casefold() for p in re.split(r"(\d+)", value))


def clean_text(value: object | None) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch if ch in "\n\t" or ord(ch) >= 32 else " " for ch in text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def one_line(value: object | None) -> str:
    return re.sub(r"\s+", " ", clean_text(value)).strip()


def truncate(value: str, limit: int) -> str:
    value = one_line(value)
    if len(value) <= limit:
        return value
    return value[: max(1, limit - 1)].rstrip() + "..."


def destination(prefix: str, key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8", "surrogatepass")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def read_text_best(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig", errors="replace")
    for encoding in ("utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def format_mail_date(value: str) -> str:
    value = one_line(value)
    if not value:
        return ""
    try:
        dt = parsedate_to_datetime(value)
        if dt is None:
            return value
        if dt.tzinfo is None:
            return dt.strftime("%Y-%m-%d %H:%M")
        return dt.strftime("%Y-%m-%d %H:%M %z")
    except (TypeError, ValueError, OverflowError):
        return value


class HTMLToText(HTMLParser):
    BLOCK_START = {
        "address",
        "article",
        "aside",
        "blockquote",
        "div",
        "dl",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tr",
        "ul",
    }
    SKIP = {"script", "style", "svg", "canvas", "noscript"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0
        self.anchor_stack: list[tuple[str, int]] = []

    def newline(self) -> None:
        if not self.parts or not self.parts[-1].endswith("\n"):
            self.parts.append("\n")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.SKIP:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in self.BLOCK_START or tag == "br":
            self.newline()
        if tag == "li":
            self.newline()
            self.parts.append("- ")
        if tag == "a":
            href = dict(attrs).get("href") or ""
            self.anchor_stack.append((href, len("".join(self.parts))))

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP:
            if self.skip_depth:
                self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag == "a" and self.anchor_stack:
            href, _ = self.anchor_stack.pop()
            if href and not href.lower().startswith(("cid:", "data:")):
                tail = "".join(self.parts[-6:])
                if href not in tail:
                    self.parts.append(f" <{href}>")
        if tag in self.BLOCK_START or tag in {"li", "td", "th"}:
            self.newline()

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data)

    def get_text(self) -> str:
        text = html.unescape("".join(self.parts))
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        return clean_text(text)


def html_to_text(source: str) -> str:
    parser = HTMLToText()
    try:
        parser.feed(source)
        parser.close()
        return parser.get_text()
    except Exception:
        return clean_text(re.sub(r"<[^>]+>", " ", source))


def rtf_to_text(source: str) -> str:
    """Small, dependency-free RTF text extractor for mail bodies."""
    data = source
    data = re.sub(
        r"\\'([0-9a-fA-F]{2})",
        lambda m: bytes([int(m.group(1), 16)]).decode("cp1252", errors="replace"),
        data,
    )

    def unicode_repl(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if number < 0:
            number += 65536
        try:
            return chr(number)
        except ValueError:
            return "\ufffd"

    data = re.sub(r"\\u(-?\d+)\??", unicode_repl, data)
    data = re.sub(r"\\(?:par|line)\b ?", "\n", data)
    data = re.sub(r"\\tab\b ?", "\t", data)
    data = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", data)
    data = data.replace("\\{", "{").replace("\\}", "}").replace("\\\\", "\\")
    data = data.replace("{", "").replace("}", "")
    return clean_text(data)


def decode_mime_text(part) -> str:
    try:
        content = part.get_content()
        if isinstance(content, str):
            return content
        if isinstance(content, bytes):
            raw = content
        else:
            return str(content)
    except Exception:
        raw = part.get_payload(decode=True)
        if raw is None:
            payload = part.get_payload()
            return payload if isinstance(payload, str) else ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return raw.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return raw.decode("utf-8", errors="replace")


def extract_mail_part(part) -> tuple[str, int, list[str]]:
    """Return body text, preference score, and calendar payloads."""
    disposition = part.get_content_disposition()
    if disposition == "attachment" or part.get_filename():
        return "", -1, []

    content_type = part.get_content_type().lower()
    if content_type.startswith("image/") or content_type == "message/rfc822":
        return "", -1, []

    if part.is_multipart():
        children = list(part.iter_parts())
        calendars: list[str] = []
        subtype = part.get_content_subtype().lower()
        results: list[tuple[str, int]] = []
        for child in children:
            body, score, child_cal = extract_mail_part(child)
            calendars.extend(child_cal)
            if body:
                results.append((body, score))

        if not results:
            return "", -1, calendars
        if subtype in {"alternative", "related"}:
            body, score = max(results, key=lambda value: value[1])
            return body, score, calendars
        bodies = [body for body, _ in results]
        return clean_text("\n\n".join(bodies)), max(score for _, score in results), calendars

    raw = decode_mime_text(part)
    if content_type == "text/calendar":
        return "", -1, [raw]
    if content_type == "text/plain":
        return clean_text(raw), 30, []
    if content_type == "text/html":
        return html_to_text(raw), 20, []
    if content_type in {"text/rtf", "application/rtf"}:
        return rtf_to_text(raw), 10, []
    if content_type.startswith("text/"):
        return clean_text(raw), 5, []
    return "", -1, []


def unfold_content_lines(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    result: list[str] = []
    for line in text.split("\n"):
        if line.startswith((" ", "\t")) and result:
            result[-1] += line[1:]
        else:
            result.append(line)
    return result


def split_property(line: str) -> tuple[str, dict[str, str], str] | None:
    if ":" not in line:
        return None
    left, value = line.split(":", 1)
    pieces = left.split(";")
    name = pieces[0].split(".")[-1].upper()
    params: dict[str, str] = {}
    for piece in pieces[1:]:
        if "=" in piece:
            key, val = piece.split("=", 1)
            params[key.upper()] = val.strip('"')
    if params.get("ENCODING", "").upper() == "QUOTED-PRINTABLE":
        value = quopri.decodestring(value).decode(params.get("CHARSET", "utf-8"), errors="replace")
    return name, params, value


def unescape_ical_text(value: str) -> str:
    return (
        value.replace("\\n", "\n")
        .replace("\\N", "\n")
        .replace("\\,", ",")
        .replace("\\;", ";")
        .replace("\\\\", "\\")
    )


@dataclass
class CalendarComponent:
    kind: str
    properties: dict[str, list[tuple[dict[str, str], str]]] = field(default_factory=dict)

    def values(self, name: str) -> list[tuple[dict[str, str], str]]:
        return self.properties.get(name.upper(), [])

    def first(self, name: str) -> tuple[dict[str, str], str] | None:
        values = self.values(name)
        return values[0] if values else None


def parse_calendar(text: str) -> list[CalendarComponent]:
    wanted = {"VEVENT", "VTODO", "VJOURNAL", "VFREEBUSY"}
    stack: list[str] = []
    current: CalendarComponent | None = None
    result: list[CalendarComponent] = []
    for line in unfold_content_lines(text):
        parsed = split_property(line)
        if parsed is None:
            continue
        name, params, value = parsed
        if name == "BEGIN":
            component = value.upper()
            stack.append(component)
            if component in wanted:
                current = CalendarComponent(component)
            continue
        if name == "END":
            component = value.upper()
            if current is not None and component == current.kind:
                result.append(current)
                current = None
            if stack:
                stack.pop()
            continue
        if current is not None:
            current.properties.setdefault(name, []).append((params, unescape_ical_text(value)))
    return result


def format_ical_datetime(params: dict[str, str], value: str) -> str:
    raw = value.strip()
    tzid = params.get("TZID", "")
    is_date = params.get("VALUE", "").upper() == "DATE" or re.fullmatch(r"\d{8}", raw)
    formats = ["%Y%m%d"] if is_date else ["%Y%m%dT%H%M%SZ", "%Y%m%dT%H%M%S", "%Y%m%dT%H%M"]
    for fmt in formats:
        try:
            dt = datetime.strptime(raw, fmt)
            if is_date:
                return dt.strftime("%Y-%m-%d")
            suffix = " UTC" if raw.endswith("Z") else (f" ({tzid})" if tzid else "")
            return dt.strftime("%Y-%m-%d %H:%M") + suffix
        except ValueError:
            pass
    return raw + (f" ({tzid})" if tzid else "")


def calendar_summary(components: Sequence[CalendarComponent]) -> tuple[str, str]:
    if not components:
        return "Calendar item", ""
    component = components[0]
    summary_entry = component.first("SUMMARY")
    title = one_line(summary_entry[1]) if summary_entry else component.kind.title()
    start_entry = component.first("DTSTART") or component.first("DUE")
    date = format_ical_datetime(*start_entry) if start_entry else ""
    return title or "Calendar item", date


def component_lines(component: CalendarComponent) -> list[tuple[str, str]]:
    labels = [
        ("SUMMARY", "Summary"),
        ("DTSTART", "Start"),
        ("DTEND", "End"),
        ("DUE", "Due"),
        ("LOCATION", "Location"),
        ("STATUS", "Status"),
        ("ORGANIZER", "Organizer"),
        ("RRULE", "Recurrence"),
        ("URL", "URL"),
        ("UID", "UID"),
    ]
    output: list[tuple[str, str]] = []
    for key, label in labels:
        entry = component.first(key)
        if not entry:
            continue
        params, value = entry
        if key in {"DTSTART", "DTEND", "DUE"}:
            value = format_ical_datetime(params, value)
        output.append((label, clean_text(value)))
    attendees = [clean_text(value) for _, value in component.values("ATTENDEE") if clean_text(value)]
    if attendees:
        output.append(("Attendees", "\n".join(f"- {value}" for value in attendees)))
    description = component.first("DESCRIPTION")
    if description and clean_text(description[1]):
        output.append(("Description", clean_text(description[1])))
    return output


def parse_vcards(text: str) -> list[dict[str, list[tuple[dict[str, str], str]]]]:
    cards: list[dict[str, list[tuple[dict[str, str], str]]]] = []
    current: dict[str, list[tuple[dict[str, str], str]]] | None = None
    for line in unfold_content_lines(text):
        parsed = split_property(line)
        if parsed is None:
            continue
        name, params, value = parsed
        if name == "BEGIN" and value.upper() == "VCARD":
            current = {}
            continue
        if name == "END" and value.upper() == "VCARD":
            if current is not None:
                cards.append(current)
            current = None
            continue
        if current is not None:
            current.setdefault(name, []).append((params, unescape_ical_text(value)))
    return cards


def first_card_value(card: dict[str, list[tuple[dict[str, str], str]]], key: str) -> str:
    values = card.get(key, [])
    return clean_text(values[0][1]) if values else ""


def vcard_name(card: dict[str, list[tuple[dict[str, str], str]]]) -> str:
    name = first_card_value(card, "FN")
    if name:
        return one_line(name)
    n = first_card_value(card, "N")
    if n:
        parts = n.split(";")
        ordered = (parts[1:2] + parts[0:1] + parts[2:]) if len(parts) > 1 else parts
        return one_line(" ".join(part for part in ordered if part))
    return "Contact"


def vcard_lines(card: dict[str, list[tuple[dict[str, str], str]]]) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    for key, label in (("FN", "Name"), ("ORG", "Organization"), ("TITLE", "Title"), ("BDAY", "Birthday")):
        value = first_card_value(card, key)
        if value:
            output.append((label, value.replace(";", ", ")))
    for key, label in (("EMAIL", "Email"), ("TEL", "Phone"), ("ADR", "Address"), ("URL", "URL")):
        values = [clean_text(value).replace(";", ", ") for _, value in card.get(key, []) if clean_text(value)]
        if values:
            output.append((label, "\n".join(values)))
    note = first_card_value(card, "NOTE")
    if note:
        output.append(("Notes", note))
    return output


@dataclass
class ItemStub:
    path: Path
    rel_path: str
    kind: str
    title: str
    index_meta: str
    dest: str
    error: str = ""


@dataclass
class FolderNode:
    path: Path
    rel_parts: tuple[str, ...]
    parent: FolderNode | None = None
    children: list[FolderNode] = field(default_factory=list)
    items: list[ItemStub] = field(default_factory=list)
    dest: str = ""

    @property
    def rel_key(self) -> str:
        return "/".join(self.rel_parts) if self.rel_parts else "."

    @property
    def display_name(self) -> str:
        return self.rel_parts[-1] if self.rel_parts else self.path.name

    @property
    def display_path(self) -> str:
        return "/".join(self.rel_parts) if self.rel_parts else self.path.name

    @property
    def depth(self) -> int:
        return len(self.rel_parts)

    def recursive_count(self) -> int:
        return len(self.items) + sum(child.recursive_count() for child in self.children)


@dataclass
class ItemContent:
    title: str
    kind_label: str
    metadata: list[tuple[str, str]]
    sections: list[tuple[str, str]]
    warnings: list[str] = field(default_factory=list)


def inspect_stub(path: Path, root: Path) -> ItemStub:
    rel = path.relative_to(root).as_posix()
    ext = path.suffix.lower()
    kind = ext.lstrip(".").upper()
    title = path.stem
    meta = kind
    error = ""
    try:
        if ext == ".eml":
            with path.open("rb") as stream:
                message = BytesHeaderParser(policy=policy.default).parse(stream)
            title = one_line(message.get("Subject")) or "(No subject)"
            date = format_mail_date(str(message.get("Date", "")))
            sender = one_line(message.get("From"))
            meta = " - ".join(value for value in (date, sender) if value) or "Mail"
        elif ext == ".ics":
            components = parse_calendar(read_text_best(path))
            title, date = calendar_summary(components)
            meta = " - ".join(value for value in (date, "Calendar") if value)
        elif ext == ".vcf":
            cards = parse_vcards(read_text_best(path))
            title = vcard_name(cards[0]) if cards else path.stem
            email_value = first_card_value(cards[0], "EMAIL") if cards else ""
            meta = " - ".join(value for value in (email_value, "Contact") if value)
    except Exception as exc:  # include damaged files rather than aborting the archive
        error = f"{type(exc).__name__}: {exc}"
        meta = f"Unreadable {kind}"
    return ItemStub(path, rel, kind, title, meta, destination("item", rel), error)


def build_tree(root: Path, include_hidden: bool, follow_symlinks: bool) -> tuple[FolderNode, int]:
    root_node = FolderNode(root, (), None, dest=destination("folder", "."))
    nodes: dict[tuple[str, ...], FolderNode] = {(): root_node}
    item_count = 0

    for current_dir, dir_names, file_names in os.walk(root, followlinks=follow_symlinks):
        current = Path(current_dir)
        rel_parts = current.relative_to(root).parts
        if not include_hidden:
            dir_names[:] = [name for name in dir_names if not name.startswith(".")]
        if not follow_symlinks:
            dir_names[:] = [name for name in dir_names if not (current / name).is_symlink()]

        if rel_parts not in nodes:
            parent_parts = rel_parts[:-1]
            parent = nodes[parent_parts]
            node = FolderNode(current, rel_parts, parent, dest=destination("folder", "/".join(rel_parts)))
            parent.children.append(node)
            nodes[rel_parts] = node
        node = nodes[rel_parts]

        for file_name in file_names:
            if not include_hidden and file_name.startswith("."):
                continue
            file_path = current / file_name
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            node.items.append(inspect_stub(file_path, root))
            item_count += 1

        for dir_name in dir_names:
            child_parts = rel_parts + (dir_name,)
            if child_parts not in nodes:
                child_path = current / dir_name
                child = FolderNode(
                    child_path,
                    child_parts,
                    node,
                    dest=destination("folder", "/".join(child_parts)),
                )
                node.children.append(child)
                nodes[child_parts] = child

    def prune_and_sort(node: FolderNode) -> bool:
        node.children = [child for child in node.children if prune_and_sort(child)]
        node.children.sort(key=lambda child: natural_key(child.display_name))
        node.items.sort(key=lambda item: natural_key(Path(item.rel_path).name))
        return bool(node.items or node.children or node is root_node)

    prune_and_sort(root_node)
    return root_node, item_count


def iter_folders(node: FolderNode) -> Iterator[FolderNode]:
    yield node
    for child in node.children:
        yield from iter_folders(child)


def load_item(stub: ItemStub) -> ItemContent:
    if stub.error:
        return ItemContent(
            title=stub.title,
            kind_label=stub.kind,
            metadata=[("File", stub.rel_path), ("Error", stub.error)],
            sections=[("Unreadable item", "The source file could not be parsed. It is still represented in the archive index.")],
        )

    ext = stub.path.suffix.lower()
    try:
        if ext == ".eml":
            with stub.path.open("rb") as stream:
                message = BytesParser(policy=policy.default).parse(stream)
            body, _, calendars = extract_mail_part(message)
            title = one_line(message.get("Subject")) or "(No subject)"
            metadata = [
                ("From", one_line(message.get("From"))),
                ("To", one_line(message.get("To"))),
                ("Cc", one_line(message.get("Cc"))),
                ("Date", format_mail_date(str(message.get("Date", "")))),
                ("Message-ID", one_line(message.get("Message-ID"))),
                ("File", stub.rel_path),
            ]
            metadata = [(label, value) for label, value in metadata if value]
            sections: list[tuple[str, str]] = []
            for index, calendar_payload in enumerate(calendars, 1):
                components = parse_calendar(calendar_payload)
                if components:
                    lines: list[str] = []
                    for component_index, component in enumerate(components, 1):
                        heading = one_line(component.first("SUMMARY")[1]) if component.first("SUMMARY") else component.kind.title()
                        if len(components) > 1:
                            lines.append(f"{component_index}. {heading}")
                        for label, value in component_lines(component):
                            lines.append(f"{label}: {value}")
                        lines.append("")
                    sections.append(("Calendar invitation" if index == 1 else f"Calendar invitation {index}", clean_text("\n".join(lines))))
            sections.append(("Message", body or "[No textual message body]"))
            return ItemContent(title, "MAIL", metadata, sections)

        if ext == ".ics":
            components = parse_calendar(read_text_best(stub.path))
            title, _ = calendar_summary(components)
            metadata = [("File", stub.rel_path), ("Components", str(len(components)))]
            sections = []
            if not components:
                sections.append(("Calendar", "[No supported VEVENT, VTODO, VJOURNAL, or VFREEBUSY component found]"))
            for index, component in enumerate(components, 1):
                heading_entry = component.first("SUMMARY")
                heading = one_line(heading_entry[1]) if heading_entry else component.kind.title()
                if len(components) > 1:
                    heading = f"{index}. {heading}"
                text = "\n".join(f"{label}: {value}" for label, value in component_lines(component))
                sections.append((heading, text or f"[{component.kind}]"))
            return ItemContent(title, "CALENDAR", metadata, sections)

        if ext == ".vcf":
            cards = parse_vcards(read_text_best(stub.path))
            title = vcard_name(cards[0]) if cards else stub.title
            metadata = [("File", stub.rel_path), ("Contacts", str(len(cards)))]
            sections = []
            if not cards:
                sections.append(("Contact", "[No VCARD component found]"))
            for index, card in enumerate(cards, 1):
                heading = vcard_name(card)
                if len(cards) > 1:
                    heading = f"{index}. {heading}"
                text = "\n".join(f"{label}: {value}" for label, value in vcard_lines(card))
                sections.append((heading, text or "[Empty contact]"))
            return ItemContent(title, "CONTACT", metadata, sections)

        return ItemContent(stub.title, stub.kind, [("File", stub.rel_path)], [("Item", "[Unsupported file type]")])
    except Exception as exc:
        return ItemContent(
            stub.title,
            stub.kind,
            [("File", stub.rel_path), ("Error", f"{type(exc).__name__}: {exc}")],
            [("Unreadable item", "The source file could not be fully parsed.")],
        )


def find_font(explicit: str | None, pattern: str, candidates: Sequence[str]) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Font not found: {path}")
        return path
    if shutil.which("fc-match"):
        try:
            process = subprocess.run(
                ["fc-match", "-f", "%{file}\n", pattern],
                check=True,
                capture_output=True,
                text=True,
            )
            for line in process.stdout.splitlines():
                path = Path(line.strip())
                if path.is_file() and path.suffix.lower() in {".ttf", ".otf"}:
                    return path
        except (OSError, subprocess.SubprocessError):
            pass
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            return path
    return None


@dataclass(frozen=True)
class Fonts:
    regular: str
    bold: str
    mono: str


def register_fonts(args: argparse.Namespace) -> Fonts:
    regular_path = find_font(
        args.font,
        "Noto Sans:style=Regular",
        (
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ),
    )
    bold_path = find_font(
        args.bold_font,
        "Noto Sans:style=Bold",
        (
            "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ),
    )
    mono_path = find_font(
        args.mono_font,
        "Noto Sans Mono:style=Regular",
        (
            "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
        ),
    )
    if regular_path:
        pdfmetrics.registerFont(TTFont("ArchiveSans", str(regular_path)))
        regular = "ArchiveSans"
    else:
        regular = "Helvetica"
    if bold_path:
        pdfmetrics.registerFont(TTFont("ArchiveSansBold", str(bold_path)))
        bold = "ArchiveSansBold"
    else:
        bold = regular if regular != "Helvetica" else "Helvetica-Bold"
    if mono_path:
        pdfmetrics.registerFont(TTFont("ArchiveMono", str(mono_path)))
        mono = "ArchiveMono"
    else:
        mono = regular if regular != "Helvetica" else "Courier"

    cjk_regular: str | None = None
    cjk_bold: str | None = None
    korean_regular: str | None = None
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))
        cjk_regular = "HeiseiMin-W3"
    except Exception:
        pass
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))
        cjk_bold = "HeiseiKakuGo-W5"
    except Exception:
        cjk_bold = cjk_regular
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
        korean_regular = "HYSMyeongJo-Medium"
    except Exception:
        pass

    _FONT_FALLBACKS.clear()
    _FONT_FALLBACKS[regular] = (cjk_regular, korean_regular)
    _FONT_FALLBACKS[bold] = (cjk_bold or cjk_regular, korean_regular)
    _FONT_FALLBACKS[mono] = (cjk_regular, korean_regular)
    return Fonts(regular, bold, mono)


def text_width(text: str, font: str, size: float) -> float:
    total = 0.0
    for run_font, run_text in font_runs(text, font):
        try:
            total += pdfmetrics.stringWidth(run_text, run_font, size)
        except Exception:
            safe = run_text.encode("ascii", "replace").decode("ascii")
            total += pdfmetrics.stringWidth(safe, font, size)
    return total


def split_long_token(token: str, font: str, size: float, width: float) -> list[str]:
    pieces: list[str] = []
    current = ""
    current_width = 0.0
    for char in token:
        char_width = text_width(char, font, size)
        if current and current_width + char_width > width:
            pieces.append(current)
            current = char
            current_width = char_width
        else:
            current += char
            current_width += char_width
    if current or not pieces:
        pieces.append(current)
    return pieces


def wrap_line(text: str, font: str, size: float, width: float) -> list[str]:
    text = text.expandtabs(4)
    if text == "":
        return [""]
    if text_width(text, font, size) <= width:
        return [text]

    tokens = re.findall(r"\s+|\S+", text)
    lines: list[str] = []
    current = ""
    for token in tokens:
        candidate = current + token
        if current and text_width(candidate, font, size) <= width:
            current = candidate
            continue
        if not current and text_width(token, font, size) <= width:
            current = token
            continue
        if current:
            lines.append(current.rstrip())
            current = ""
        if token.isspace():
            continue
        pieces = split_long_token(token, font, size, width)
        lines.extend(pieces[:-1])
        current = pieces[-1]
    if current or not lines:
        lines.append(current.rstrip())
    return lines


def wrap_text(text: str, font: str, size: float, width: float) -> list[str]:
    result: list[str] = []
    for raw_line in clean_text(text).split("\n") if clean_text(text) else [""]:
        result.extend(wrap_line(raw_line, font, size, width))
    return result


@dataclass
class IndexEntry:
    title: str
    meta: str
    dest: str
    indent: float = 0.0
    folder: bool = False


@dataclass
class PreparedIndexEntry:
    entry: IndexEntry
    title_lines: list[str]
    meta_lines: list[str]
    height: float


@dataclass
class BodyLine:
    text: str
    font: str
    size: float
    leading: float
    color: object
    indent: float = 0.0


@dataclass
class ItemLayout:
    title_lines: list[str]
    meta_rows: list[tuple[str, list[str]]]
    body_lines: list[BodyLine]
    body_font_size: float
    required_height: float
    truncated: bool


class ArchiveRenderer:
    def __init__(
        self,
        output: Path,
        root: FolderNode,
        title: str,
        fonts: Fonts,
        max_item_height: float,
        filesystem_links: bool,
        body_font_size: float,
    ) -> None:
        self.output = output
        self.root = root
        self.title = title
        self.fonts = fonts
        self.max_item_height = min(max_item_height, PDF_SAFE_MAX_HEIGHT)
        self.filesystem_links = filesystem_links
        self.body_font_size = body_font_size
        self.page_number = 0
        self.canvas = Canvas(str(output), pagesize=A4, pageCompression=1, pdfVersion=(1, 7))
        self.canvas.setTitle(title)
        self.canvas.setSubject("Linked archive generated from EML, ICS, and VCF files")
        self.canvas.setCreator("eml_archive_to_pdf.py")

    def begin_page(
        self,
        size: tuple[float, float],
        dest: str | None = None,
        outline_title: str | None = None,
        outline_level: int = 0,
    ) -> tuple[float, float]:
        self.canvas.setPageSize(size)
        self.page_number += 1
        if dest:
            self.canvas.bookmarkPage(dest)
        if dest and outline_title:
            try:
                self.canvas.addOutlineEntry(truncate(outline_title, 120), dest, level=outline_level, closed=False)
            except Exception:
                pass
        return size

    def end_page(self) -> None:
        self.canvas.showPage()

    def draw_text(self, text: str, x: float, y: float, font: str, size: float) -> float:
        current_x = x
        for run_font, run_text in font_runs(text, font):
            self.canvas.setFont(run_font, size)
            self.canvas.drawString(current_x, y, run_text)
            current_x += pdfmetrics.stringWidth(run_text, run_font, size)
        return current_x

    def draw_link_text(
        self,
        text: str,
        x: float,
        y: float,
        font: str,
        size: float,
        dest: str | None = None,
        url: str | None = None,
        color=LINK,
    ) -> float:
        self.canvas.setFillColor(color)
        self.draw_text(text, x, y, font, size)
        width = text_width(text, font, size)
        rect = (x, y - 2, x + width, y + size + 1)
        if dest:
            self.canvas.linkRect("", dest, Rect=rect, relative=0, thickness=0)
        elif url:
            self.canvas.linkURL(url, rect, relative=0, thickness=0)
        return x + width

    def draw_footer(self, page_width: float) -> None:
        y = 18
        self.canvas.setStrokeColor(RULE)
        self.canvas.setLineWidth(0.5)
        self.canvas.line(30, 32, page_width - 30, 32)
        self.draw_link_text("Archive contents", 36, y, self.fonts.regular, 8.5, dest="archive_toc")
        page_text = f"Page {self.page_number}"
        self.canvas.setFont(self.fonts.regular, 8)
        self.canvas.setFillColor(MUTED)
        self.canvas.drawRightString(page_width - 36, y, page_text)

    def folder_chain(self, folder: FolderNode) -> list[FolderNode]:
        chain: list[FolderNode] = []
        current: FolderNode | None = folder
        while current is not None:
            chain.append(current)
            current = current.parent
        chain.reverse()
        return chain

    def draw_header(self, folder: FolderNode, page_width: float, page_height: float, item_page: bool) -> None:
        header_bottom = page_height - 34
        self.canvas.setFillColor(PANEL)
        self.canvas.rect(0, header_bottom, page_width, 34, stroke=0, fill=1)
        self.canvas.setStrokeColor(RULE)
        self.canvas.line(0, header_bottom, page_width, header_bottom)

        x = 36.0
        y = page_height - 21.5
        right_reserved = 195 if item_page else 120
        max_x = page_width - right_reserved
        size = 8.5
        chain = self.folder_chain(folder)
        for index, node in enumerate(chain):
            label = self.root.path.name if node is self.root else node.display_name
            if index:
                separator = " / "
                separator_width = text_width(separator, self.fonts.regular, size)
                if x + separator_width > max_x:
                    break
                self.canvas.setFillColor(MUTED)
                self.canvas.setFont(self.fonts.regular, size)
                self.canvas.drawString(x, y, separator)
                x += separator_width
            available = max_x - x
            if available <= 18:
                break
            shown = label
            while shown and text_width(shown, self.fonts.regular, size) > available:
                shown = shown[:-1]
            if shown != label and len(shown) > 3:
                shown = shown[:-3] + "..."
            if not shown:
                break
            x = self.draw_link_text(shown, x, y, self.fonts.regular, size, dest=node.dest)

        right_x = page_width - 36
        if item_page:
            label = "Folder index"
            width = text_width(label, self.fonts.regular, 8.5)
            self.draw_link_text(label, right_x - width, y, self.fonts.regular, 8.5, dest=folder.dest)
            right_x -= width + 16
            if self.filesystem_links:
                open_label = "Open folder"
                open_width = text_width(open_label, self.fonts.regular, 8.5)
                try:
                    uri = folder.path.resolve().as_uri()
                    self.draw_link_text(open_label, right_x - open_width, y, self.fonts.regular, 8.5, url=uri)
                except ValueError:
                    pass
        else:
            label = "Archive contents"
            width = text_width(label, self.fonts.regular, 8.5)
            self.draw_link_text(label, right_x - width, y, self.fonts.regular, 8.5, dest="archive_toc")

    def prepare_index_entries(
        self,
        entries: Sequence[IndexEntry],
        title_width: float,
    ) -> list[PreparedIndexEntry]:
        prepared: list[PreparedIndexEntry] = []
        for entry in entries:
            width = max(80.0, title_width - entry.indent)
            font = self.fonts.bold if entry.folder else self.fonts.regular
            title_lines = wrap_text(entry.title, font, 10.2, width)
            meta_lines = wrap_text(entry.meta, self.fonts.regular, 8.2, width) if entry.meta else []
            height = max(18.0, len(title_lines) * 12.2 + len(meta_lines) * 10.0 + 7.0)
            prepared.append(PreparedIndexEntry(entry, title_lines, meta_lines, height))
        return prepared

    def paginate_index(self, prepared: Sequence[PreparedIndexEntry], available: float) -> list[list[PreparedIndexEntry]]:
        pages: list[list[PreparedIndexEntry]] = []
        current: list[PreparedIndexEntry] = []
        used = 0.0
        for entry in prepared:
            if current and used + entry.height > available:
                pages.append(current)
                current = []
                used = 0.0
            current.append(entry)
            used += entry.height
        pages.append(current)
        return pages

    def render_index_pages(
        self,
        page_title: str,
        subtitle: str,
        entries: Sequence[IndexEntry],
        first_dest: str,
        outline_title: str,
        outline_level: int,
        folder: FolderNode | None,
        is_toc: bool = False,
    ) -> None:
        left = 42.0
        right = A4_WIDTH - 42.0
        title_width = right - left - 10
        prepared = self.prepare_index_entries(entries, title_width)
        available = A4_HEIGHT - 172
        chunks = self.paginate_index(prepared, available)

        for page_index, chunk in enumerate(chunks):
            dest = first_dest if page_index == 0 else None
            outline = outline_title if page_index == 0 else None
            self.begin_page(A4, dest, outline, outline_level)
            if folder is not None:
                self.draw_header(folder, A4_WIDTH, A4_HEIGHT, item_page=False)
                top = A4_HEIGHT - 64
            else:
                self.canvas.setFillColor(PANEL)
                self.canvas.rect(0, A4_HEIGHT - 34, A4_WIDTH, 34, stroke=0, fill=1)
                self.canvas.setFont(self.fonts.regular, 8.5)
                self.canvas.setFillColor(MUTED)
                self.draw_text(self.title, 36, A4_HEIGHT - 21.5, self.fonts.regular, 8.5)
                top = A4_HEIGHT - 64

            shown_title = page_title if page_index == 0 else f"{page_title} - continued"
            self.canvas.setFont(self.fonts.bold, 20)
            self.canvas.setFillColor(INK)
            self.draw_text(shown_title, left, top, self.fonts.bold, 20)
            self.canvas.setFont(self.fonts.regular, 9)
            self.canvas.setFillColor(MUTED)
            self.draw_text(subtitle, left, top - 20, self.fonts.regular, 9)
            y = top - 48

            if not chunk:
                self.canvas.setFont(self.fonts.regular, 10)
                self.canvas.setFillColor(MUTED)
                self.canvas.drawString(left, y, "No supported items in this folder.")

            for prepared_entry in chunk:
                entry = prepared_entry.entry
                x = left + entry.indent
                entry_top = y
                if entry.folder:
                    self.canvas.setFillColor(PANEL)
                    self.canvas.roundRect(x - 7, y - prepared_entry.height + 4, right - x + 7, prepared_entry.height - 1, 3, stroke=0, fill=1)
                font = self.fonts.bold if entry.folder else self.fonts.regular
                self.canvas.setFont(font, 10.2)
                self.canvas.setFillColor(LINK)
                line_y = y - 11
                for line in prepared_entry.title_lines:
                    self.draw_text(line, x, line_y, font, 10.2)
                    line_y -= 12.2
                self.canvas.setFont(self.fonts.regular, 8.2)
                self.canvas.setFillColor(MUTED)
                for line in prepared_entry.meta_lines:
                    self.draw_text(line, x, line_y, self.fonts.regular, 8.2)
                    line_y -= 10.0
                self.canvas.linkRect(
                    "",
                    entry.dest,
                    Rect=(x - 3, entry_top - prepared_entry.height + 2, right, entry_top + 2),
                    relative=0,
                    thickness=0,
                )
                y -= prepared_entry.height

            self.draw_footer(A4_WIDTH)
            self.end_page()

    def toc_entries(self) -> list[IndexEntry]:
        entries: list[IndexEntry] = []
        for folder in iter_folders(self.root):
            count = folder.recursive_count()
            direct = len(folder.items)
            label = self.root.path.name if folder is self.root else folder.display_name
            meta = f"{direct} direct item{'s' if direct != 1 else ''}; {count} including subfolders"
            entries.append(IndexEntry(label, meta, folder.dest, indent=folder.depth * 16.0, folder=True))
        return entries

    def render_toc(self) -> None:
        total = self.root.recursive_count()
        folder_count = sum(1 for _ in iter_folders(self.root))
        subtitle = f"{folder_count} folders - {total} items - folder links only"
        self.render_index_pages(
            self.title,
            subtitle,
            self.toc_entries(),
            "archive_toc",
            self.title,
            0,
            folder=None,
            is_toc=True,
        )

    def folder_index_entries(self, folder: FolderNode) -> list[IndexEntry]:
        entries: list[IndexEntry] = []
        for child in folder.children:
            count = child.recursive_count()
            entries.append(
                IndexEntry(
                    f"Folder: {child.display_name}",
                    f"{count} item{'s' if count != 1 else ''} including subfolders",
                    child.dest,
                    folder=True,
                )
            )
        for item in folder.items:
            prefix = {"EML": "Mail", "ICS": "Calendar", "VCF": "Contact"}.get(item.kind, item.kind)
            entries.append(IndexEntry(item.title, f"{prefix} - {item.index_meta}", item.dest))
        return entries

    def render_folder_index(self, folder: FolderNode) -> None:
        title = self.root.path.name if folder is self.root else folder.display_path
        direct = len(folder.items)
        recursive = folder.recursive_count()
        subtitle = f"{direct} direct item{'s' if direct != 1 else ''}; {recursive} including subfolders"
        self.render_index_pages(
            title,
            subtitle,
            self.folder_index_entries(folder),
            folder.dest,
            title,
            folder.depth + 1,
            folder=folder,
        )

    def build_item_layout(self, content: ItemContent, body_size: float, max_height: float) -> ItemLayout:
        page_width = A4_WIDTH
        left = 42.0
        right = page_width - 42.0
        content_width = right - left
        title_lines = wrap_text(content.title, self.fonts.bold, 17.0, content_width)
        meta_label_width = 78.0
        meta_rows: list[tuple[str, list[str]]] = []
        for label, value in content.metadata:
            lines = wrap_text(value, self.fonts.regular, 8.6, content_width - meta_label_width - 14)
            meta_rows.append((label, lines))

        body_lines: list[BodyLine] = []
        for section_index, (heading, section_text) in enumerate(content.sections):
            if section_index:
                body_lines.append(BodyLine("", self.fonts.regular, body_size, body_size * 0.75, INK))
            for line in wrap_text(heading, self.fonts.bold, body_size + 1.1, content_width):
                body_lines.append(BodyLine(line, self.fonts.bold, body_size + 1.1, (body_size + 1.1) * 1.35, INK))
            for line in wrap_text(section_text, self.fonts.regular, body_size, content_width):
                body_lines.append(BodyLine(line, self.fonts.regular, body_size, body_size * 1.36, INK))
        for warning in content.warnings:
            for line in wrap_text(warning, self.fonts.regular, max(6.0, body_size - 1), content_width):
                body_lines.append(BodyLine(line, self.fonts.regular, max(6.0, body_size - 1), body_size * 1.25, MUTED))

        title_height = len(title_lines) * 21.0
        meta_height = 12.0 + sum(max(1, len(lines)) * 11.0 + 4.0 for _, lines in meta_rows)
        body_height = sum(line.leading for line in body_lines)
        required = 34 + 28 + title_height + 15 + meta_height + 22 + body_height + 52
        return ItemLayout(title_lines, meta_rows, body_lines, body_size, required, False)

    def fit_item_layout(self, content: ItemContent) -> ItemLayout:
        body_size = self.body_font_size
        layout = self.build_item_layout(content, body_size, self.max_item_height)
        while layout.required_height > self.max_item_height and body_size > 5.5:
            body_size = max(5.5, body_size - 0.5)
            layout = self.build_item_layout(content, body_size, self.max_item_height)
        if layout.required_height <= self.max_item_height:
            return layout

        fixed = layout.required_height - sum(line.leading for line in layout.body_lines)
        available = max(20.0, self.max_item_height - fixed - 18.0)
        kept: list[BodyLine] = []
        used = 0.0
        for line in layout.body_lines:
            if used + line.leading > available:
                break
            kept.append(line)
            used += line.leading
        notice = "[Content truncated: this single item exceeded the configured maximum PDF page height.]"
        notice_lines = wrap_text(notice, self.fonts.bold, 6.0, A4_WIDTH - 84)
        for line in notice_lines:
            notice_line = BodyLine(line, self.fonts.bold, 6.0, 8.0, MUTED)
            while kept and used + notice_line.leading > available:
                removed = kept.pop()
                used -= removed.leading
            kept.append(notice_line)
            used += notice_line.leading
        layout.body_lines = kept
        layout.required_height = self.max_item_height
        layout.truncated = True
        return layout

    def render_item(self, folder: FolderNode, stub: ItemStub) -> None:
        content = load_item(stub)
        layout = self.fit_item_layout(content)
        page_height = max(A4_HEIGHT, min(layout.required_height, self.max_item_height))
        self.begin_page(
            (A4_WIDTH, page_height),
            stub.dest,
            content.title,
            folder.depth + 2,
        )
        self.draw_header(folder, A4_WIDTH, page_height, item_page=True)

        left = 42.0
        right = A4_WIDTH - 42.0
        width = right - left
        y = page_height - 68

        self.canvas.setFont(self.fonts.bold, 17.0)
        self.canvas.setFillColor(INK)
        for line in layout.title_lines:
            self.draw_text(line, left, y, self.fonts.bold, 17.0)
            y -= 21.0

        badge_text = content.kind_label
        badge_width = text_width(badge_text, self.fonts.bold, 7.5) + 14
        self.canvas.setFillColor(PANEL_DARK)
        self.canvas.roundRect(left, y - 1, badge_width, 15, 4, stroke=0, fill=1)
        self.canvas.setFont(self.fonts.bold, 7.5)
        self.canvas.setFillColor(MUTED)
        self.canvas.drawString(left + 7, y + 3.5, badge_text)
        y -= 25

        meta_height = 12.0 + sum(max(1, len(lines)) * 11.0 + 4.0 for _, lines in layout.meta_rows)
        self.canvas.setFillColor(PANEL)
        self.canvas.roundRect(left, y - meta_height, width, meta_height, 5, stroke=0, fill=1)
        meta_y = y - 17
        for label, lines in layout.meta_rows:
            self.canvas.setFont(self.fonts.bold, 8.6)
            self.canvas.setFillColor(MUTED)
            self.canvas.drawString(left + 10, meta_y, label)
            self.canvas.setFont(self.fonts.regular, 8.6)
            self.canvas.setFillColor(INK)
            value_y = meta_y
            for value_line in lines or [""]:
                self.draw_text(value_line, left + 88, value_y, self.fonts.regular, 8.6)
                value_y -= 11.0
            row_height = max(1, len(lines)) * 11.0 + 4.0
            meta_y -= row_height
        y -= meta_height + 20

        for line in layout.body_lines:
            self.canvas.setFont(line.font, line.size)
            self.canvas.setFillColor(line.color)
            self.draw_text(line.text, left + line.indent, y, line.font, line.size)
            y -= line.leading

        if layout.truncated:
            print(f"warning: truncated over-height item: {stub.rel_path}", file=sys.stderr)
        self.draw_footer(A4_WIDTH)
        self.end_page()

    def render_tree(self, folder: FolderNode) -> None:
        self.render_folder_index(folder)
        for item in folder.items:
            self.render_item(folder, item)
        for child in folder.children:
            self.render_tree(child)

    def render(self) -> None:
        self.render_toc()
        self.render_tree(self.root)
        self.canvas.save()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a readpst -e folder tree into one linked PDF archive.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("source", type=Path, help="root folder containing .eml/.ics/.vcf files")
    parser.add_argument("output", type=Path, help="output PDF")
    parser.add_argument("--title", help="PDF title; defaults to the source folder name")
    parser.add_argument("--font", help="regular TTF/OTF font path")
    parser.add_argument("--bold-font", help="bold TTF/OTF font path")
    parser.add_argument("--mono-font", help="monospace TTF/OTF font path")
    parser.add_argument("--body-font-size", type=float, default=9.2, help="starting body font size in points")
    parser.add_argument(
        "--max-page-height-mm",
        type=float,
        default=5000.0,
        help="maximum height for a one-item page; very long items shrink, then truncate",
    )
    parser.add_argument("--include-hidden", action="store_true", help="include hidden files and folders")
    parser.add_argument("--follow-symlinks", action="store_true", help="follow directory symlinks")
    parser.add_argument(
        "--no-filesystem-links",
        action="store_true",
        help="do not embed file:// links that open the source folder",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        print(f"error: source is not a directory: {source}", file=sys.stderr)
        return 2
    if output.suffix.lower() != ".pdf":
        print("error: output must end in .pdf", file=sys.stderr)
        return 2
    if args.body_font_size < 5.5 or args.body_font_size > 18:
        print("error: --body-font-size must be between 5.5 and 18", file=sys.stderr)
        return 2
    max_height = args.max_page_height_mm * mm
    if max_height < A4_HEIGHT:
        print("error: --max-page-height-mm cannot be smaller than A4 height (297 mm)", file=sys.stderr)
        return 2

    root, item_count = build_tree(source, args.include_hidden, args.follow_symlinks)
    if item_count == 0:
        print("error: no .eml, .ics, or .vcf files found", file=sys.stderr)
        return 1
    fonts = register_fonts(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    title = args.title or source.name

    temp_handle = tempfile.NamedTemporaryFile(
        prefix=output.stem + ".",
        suffix=".pdf",
        dir=output.parent,
        delete=False,
    )
    temp_path = Path(temp_handle.name)
    temp_handle.close()
    try:
        renderer = ArchiveRenderer(
            temp_path,
            root,
            title,
            fonts,
            max_height,
            filesystem_links=not args.no_filesystem_links,
            body_font_size=args.body_font_size,
        )
        renderer.render()
        temp_path.replace(output)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    folder_count = sum(1 for _ in iter_folders(root))
    print(f"created: {output}")
    print(f"folders: {folder_count}")
    print(f"items: {item_count}")
    print(f"pages: {renderer.page_number}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
