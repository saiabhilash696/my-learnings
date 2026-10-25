#!/usr/bin/env python3
"""
md2ipynb.py - Convert between Markdown (.md) and Jupyter notebooks (.ipynb).

The direction is picked from the input file's extension:

    guide.md     ->  guide.ipynb      (Markdown to notebook)
    guide.ipynb  ->  guide.md         (notebook to Markdown)

Markdown -> notebook rules
--------------------------
* Fenced ```python blocks (also py / python3 / ipython)  -> code cells
* Fenced ```bash / sh / shell blocks                     -> code cells, but only
  when they are install commands (pip / conda); other shell commands such as
  `mlflow ui` (which never returns) stay as Markdown so the notebook still
  "Run All"s cleanly. Override with --shell code|markdown.
* Everything else (prose, tables, ```text, ```json ...)  -> Markdown cells
* Output-style blocks stay in Markdown by default, because they are examples,
  not real execution results.

Notebook -> Markdown rules
--------------------------
* Markdown cells are written out as they are.
* Code cells become fenced ```python blocks (the notebook's language is used).
* Code cells made only of `%pip ...` / `!cmd` lines become ```bash blocks with
  the magic prefix removed (the exact reverse of the rule above).
* Saved outputs are written under a **Output:** label as ```text blocks, so a
  notebook that was run keeps its results. Images are saved to a
  `<name>_files/` folder and linked. Use --no-outputs to drop all outputs.
* Raw cells are copied verbatim.

No third-party dependencies. If `nbformat` happens to be installed, notebooks
written by the Markdown -> notebook direction are validated with it.

Usage
-----
    python md2ipynb.py guide.md                  # -> guide.ipynb
    python md2ipynb.py guide.md -o out.ipynb
    python md2ipynb.py guide.md --split-headings 2   # new Markdown cell at every # and ##
    python md2ipynb.py guide.ipynb               # -> guide.md
    python md2ipynb.py guide.ipynb -o out.md --no-outputs
    python md2ipynb.py docs/ -o notebooks/       # every .md in a folder -> .ipynb
    python md2ipynb.py notebooks/ --to md        # every .ipynb in a folder -> .md
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import uuid
from pathlib import Path

PYTHON_LANGS = {"python", "py", "python3", "ipython", "py3"}
SHELL_LANGS = {"bash", "sh", "shell", "zsh", "console"}
INSTALL_CMD = re.compile(r"^\s*(pip3?|python3?\s+-m\s+pip|conda|mamba)\s+", re.I)

FENCE_OPEN = re.compile(r"^( {0,3})(`{3,}|~{3,})\s*(.*?)\s*$")
HEADING = re.compile(r"^(#{1,6})\s+\S")


# --------------------------------------------------------------------------- #
# Cell helpers
# --------------------------------------------------------------------------- #
def _source(text: str) -> list[str]:
    """Notebook 'source' format: list of lines, each ending in \\n except the last."""
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "id": _new_id(), "metadata": {}, "source": _source(text)}


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _new_id(),
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


# --------------------------------------------------------------------------- #
# Shell handling
# --------------------------------------------------------------------------- #
def _logical_commands(lines: list[str]) -> list[list[str]]:
    """Group physical lines into logical commands (joins backslash continuations)."""
    cmds, cur = [], []
    for ln in lines:
        if not cur and not ln.strip():
            continue
        cur.append(ln)
        if not ln.rstrip().endswith("\\"):
            cmds.append(cur)
            cur = []
    if cur:
        cmds.append(cur)
    return cmds


def is_install_only(lines: list[str]) -> bool:
    cmds = _logical_commands(lines)
    return bool(cmds) and all(INSTALL_CMD.match(c[0]) or c[0].lstrip().startswith("#") for c in cmds)


def shell_to_notebook_code(lines: list[str]) -> str:
    """Prefix commands with ! (or %pip for pip installs) so Jupyter can run them."""
    out = []
    for cmd in _logical_commands(lines):
        first = cmd[0]
        stripped = first.lstrip()
        if stripped.startswith("#"):
            out.extend(cmd)
            continue
        m = re.match(r"^(\s*)pip3?\s+(.*)$", first)
        if m:  # %pip installs into the *running* kernel's environment
            cmd = [f"{m.group(1)}%pip {m.group(2)}"] + cmd[1:]
        else:
            cmd = [f"!{first.lstrip()}"] + cmd[1:]
        out.extend(cmd)
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Core converter
# --------------------------------------------------------------------------- #
def parse_markdown(text: str, shell: str = "auto", split_headings: int = 0) -> list[dict]:
    """Turn markdown text into a list of notebook cells."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    cells: list[dict] = []
    md_buf: list[str] = []

    def flush_md() -> None:
        body = "\n".join(md_buf).strip("\n")
        md_buf.clear()
        if body.strip():
            cells.append(md_cell(body))

    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        m = FENCE_OPEN.match(line)

        # ---- fenced block? ------------------------------------------------ #
        if m and not (m.group(2)[0] == "`" and "`" in m.group(3)):
            indent, fence, info = len(m.group(1)), m.group(2), m.group(3)
            lang = re.split(r"[\s{},]", info.strip("{} "), maxsplit=1)[0].lower() if info else ""
            closer = re.compile(rf"^ {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}\s*$")

            j = i + 1
            body: list[str] = []
            while j < n and not closer.match(lines[j]):
                ln = lines[j]
                body.append(ln[indent:] if ln[:indent].strip() == "" else ln.lstrip())
                j += 1
            closed = j < n
            block_end = j + 1 if closed else j  # index just past the block

            as_code: str | None = None
            if closed and lang in PYTHON_LANGS:
                as_code = "\n".join(body).strip("\n")
            elif closed and lang in SHELL_LANGS and shell != "markdown":
                if shell == "code" or is_install_only(body):
                    as_code = shell_to_notebook_code(body).strip("\n")

            if as_code is not None and as_code.strip():
                flush_md()
                cells.append(code_cell(as_code))
            else:  # keep the whole fenced block verbatim inside Markdown
                md_buf.extend(lines[i:block_end])
            i = block_end
            continue

        # ---- optional split at headings ----------------------------------- #
        h = HEADING.match(line)
        if h and split_headings and len(h.group(1)) <= split_headings:
            flush_md()

        md_buf.append(line)
        i += 1

    flush_md()
    return cells


def build_notebook(cells: list[dict], kernel: str = "python3") -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": kernel},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def convert_md_to_ipynb(src: Path, dst: Path, shell: str, split_headings: int) -> str:
    text = src.read_text(encoding="utf-8")
    cells = parse_markdown(text, shell=shell, split_headings=split_headings)
    nb = build_notebook(cells)

    try:  # optional validation
        import nbformat

        nbformat.validate(nbformat.from_dict(nb))
    except ImportError:
        pass

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    n_code = sum(c["cell_type"] == "code" for c in cells)
    return f"{len(cells)} cells: {n_code} code, {len(cells) - n_code} markdown"


# --------------------------------------------------------------------------- #
# Notebook -> Markdown
# --------------------------------------------------------------------------- #
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
PIP_MAGIC = re.compile(r"^%pip\s+")


def _text(value) -> str:
    """Notebook text fields are either a string or a list of lines."""
    return "".join(value) if isinstance(value, list) else (value or "")


def _fence(body: str, lang: str = "") -> str:
    """Fence `body`, using a longer fence if the body itself contains backticks."""
    longest = max((len(r) for r in re.findall(r"`+", body)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}{lang}\n{body}\n{ticks}"


def notebook_code_to_shell(src: str) -> str | None:
    """If a cell is only `%pip ...` / `!cmd` lines (plus comments), return it as plain
    shell text. Otherwise return None (it is real Python and stays a python block)."""
    out: list[str] = []
    real = False
    for cmd in _logical_commands(src.split("\n")):
        first = cmd[0]
        stripped = first.lstrip()
        indent = first[: len(first) - len(stripped)]
        if stripped.startswith("#"):
            out.extend(cmd)
        elif PIP_MAGIC.match(stripped):
            out.extend([f"{indent}pip {PIP_MAGIC.sub('', stripped)}"] + cmd[1:])
            real = True
        elif stripped.startswith("!"):
            out.extend([f"{indent}{stripped[1:]}"] + cmd[1:])
            real = True
        else:
            return None
    return "\n".join(out) if real else None


def outputs_to_markdown(outputs: list[dict], cell_no: int, files_dir: str,
                        assets: dict[str, bytes]) -> list[str]:
    """Render a code cell's saved outputs as Markdown blocks.
    Images are stored in `assets` (filename -> bytes) and linked from `files_dir`."""
    blocks: list[str] = []
    stream_text: list[str] = []

    def flush_stream() -> None:
        text = "".join(stream_text).rstrip("\n")
        stream_text.clear()
        if text.strip():
            blocks.append("**Output:**\n\n" + _fence(text, "text"))

    for k, out in enumerate(outputs, start=1):
        kind = out.get("output_type")
        if kind == "stream":
            stream_text.append(ANSI.sub("", _text(out.get("text"))))
            continue
        flush_stream()

        if kind in ("execute_result", "display_data"):
            data = out.get("data", {})
            image = next((m for m in ("image/png", "image/jpeg") if m in data), None)
            if image:
                ext = "png" if image == "image/png" else "jpg"
                name = f"output_{cell_no}_{k}.{ext}"
                raw = _text(data[image]).replace("\n", "")
                assets[name] = base64.b64decode(raw)
                blocks.append(f"![output]({files_dir}/{name})")
            elif "text/markdown" in data:
                blocks.append(_text(data["text/markdown"]).strip("\n"))
            elif "text/plain" in data:
                text = ANSI.sub("", _text(data["text/plain"])).rstrip("\n")
                if text.strip():
                    blocks.append("**Output:**\n\n" + _fence(text, "text"))
        elif kind == "error":
            tb = ANSI.sub("", "\n".join(out.get("traceback", [])))
            text = tb or f"{out.get('ename', 'Error')}: {out.get('evalue', '')}"
            blocks.append("**Error:**\n\n" + _fence(text, "text"))

    flush_stream()
    return blocks


def notebook_to_markdown(nb: dict, include_outputs: bool = True,
                         files_dir: str = "files") -> tuple[str, dict[str, bytes]]:
    """Turn a notebook (as a dict) into Markdown text plus any extracted image files."""
    if nb.get("nbformat", 4) < 4 or "cells" not in nb:
        raise ValueError("only nbformat 4 notebooks are supported "
                         "(open and re-save it in a recent Jupyter to upgrade)")

    meta = nb.get("metadata", {})
    lang = (meta.get("language_info", {}).get("name")
            or meta.get("kernelspec", {}).get("language") or "python").lower()

    assets: dict[str, bytes] = {}
    blocks: list[str] = []
    for cell_no, cell in enumerate(nb["cells"], start=1):
        ctype = cell.get("cell_type")
        src = _text(cell.get("source")).strip("\n")

        if ctype in ("markdown", "raw"):
            if src.strip():
                blocks.append(src)
        elif ctype == "code":
            if src.strip():
                shell = notebook_code_to_shell(src) if lang == "python" else None
                blocks.append(_fence(shell, "bash") if shell is not None else _fence(src, lang))
            if include_outputs:
                blocks.extend(outputs_to_markdown(cell.get("outputs", []), cell_no, files_dir, assets))

    return "\n\n".join(blocks).rstrip("\n") + "\n", assets


def convert_ipynb_to_md(src: Path, dst: Path, include_outputs: bool = True) -> str:
    nb = json.loads(src.read_text(encoding="utf-8"))
    files_dir = f"{dst.stem}_files"
    text, assets = notebook_to_markdown(nb, include_outputs, files_dir)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    if assets:
        out_dir = dst.parent / files_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, data in assets.items():
            (out_dir / name).write_bytes(data)

    cells = nb["cells"]
    n_code = sum(c.get("cell_type") == "code" for c in cells)
    extra = f", {len(assets)} image(s) -> {dst.parent / files_dir}" if assets else ""
    return f"{len(cells)} cells: {n_code} code, {len(cells) - n_code} other{extra}"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Convert Markdown to a Jupyter notebook, or a notebook to Markdown.")
    ap.add_argument("input", type=Path, help="a .md or .ipynb file, or a folder of them")
    ap.add_argument("-o", "--output", type=Path,
                    help="output file (file input) or output folder (folder input)")
    ap.add_argument("--to", choices=["auto", "ipynb", "md"], default="auto",
                    help="target format. auto (default): picked from the input file's extension "
                         "(.ipynb -> md, anything else -> ipynb). For a folder, auto means ipynb; "
                         "pass --to md to convert the notebooks in it")
    ap.add_argument("--shell", choices=["auto", "code", "markdown"], default="auto",
                    help="[md -> ipynb] how to treat ```bash blocks: auto = only pip/conda "
                         "installs become code cells (default); code = all; markdown = none")
    ap.add_argument("--split-headings", type=int, default=0, metavar="N",
                    help="[md -> ipynb] start a new Markdown cell at every heading of level <= N "
                         "(default: off)")
    ap.add_argument("--no-outputs", action="store_true",
                    help="[ipynb -> md] leave the notebook's saved outputs out of the Markdown")
    args = ap.parse_args(argv)

    if not args.input.exists():
        ap.error(f"{args.input} does not exist")

    # ---- decide direction ------------------------------------------------- #
    if args.to != "auto":
        target = args.to
    elif args.input.is_dir():
        target = "ipynb"
    else:
        target = "md" if args.input.suffix.lower() == ".ipynb" else "ipynb"

    if args.input.is_file():
        is_nb = args.input.suffix.lower() == ".ipynb"
        if target == "md" and not is_nb:
            ap.error("--to md needs a .ipynb input")
        if target == "ipynb" and is_nb:
            ap.error("--to ipynb needs a Markdown input, but this file is already a notebook")

    # ---- collect jobs ------------------------------------------------------ #
    src_suffix, dst_suffix = (".ipynb", ".md") if target == "md" else (".md", ".ipynb")
    jobs: list[tuple[Path, Path]] = []
    if args.input.is_dir():
        out_dir = args.output or args.input
        for f in sorted(args.input.rglob(f"*{src_suffix}")):
            if ".ipynb_checkpoints" in f.parts:
                continue
            jobs.append((f, out_dir / f.relative_to(args.input).with_suffix(dst_suffix)))
    else:
        jobs.append((args.input, args.output or args.input.with_suffix(dst_suffix)))

    if not jobs:
        print(f"No {src_suffix} files found.", file=sys.stderr)
        return 1

    failed = 0
    for src, dst in jobs:
        try:
            if target == "md":
                summary = convert_ipynb_to_md(src, dst, include_outputs=not args.no_outputs)
            else:
                summary = convert_md_to_ipynb(src, dst, args.shell, args.split_headings)
        except (ValueError, OSError) as exc:  # bad JSON, unsupported nbformat, unreadable file
            print(f"{src}: skipped - {exc}", file=sys.stderr)
            failed += 1
            continue
        print(f"{src} -> {dst}  ({summary})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
