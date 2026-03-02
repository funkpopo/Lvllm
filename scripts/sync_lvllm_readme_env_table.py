#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Sync LVLLM environment-variable tables in README files from vllm/envs.py."""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

START_MARKER = "<!-- BEGIN_LVLLM_ENV_TABLE -->"
END_MARKER = "<!-- END_LVLLM_ENV_TABLE -->"


def _load_envs_module(repo_root: Path):
    envs_path = repo_root / "vllm" / "envs.py"
    spec = importlib.util.spec_from_file_location("lvllm_envs", envs_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {envs_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _render_rows(rows: list[object], *, lang: str) -> str:
    rendered: list[str] = []
    for row in rows:
        if lang == "en":
            category = _escape_cell(row.category_en)
            description = _escape_cell(row.description_en)
            notes = _escape_cell(row.notes_en)
        else:
            category = _escape_cell(row.category_zh)
            description = _escape_cell(row.description_zh)
            notes = _escape_cell(row.notes_zh)

        rendered.append(
            f"| `{row.name}` | {category} | `{row.default}` | "
            f"{description} | {notes} |"
        )
    return "\n".join(rendered)


def _replace_table_block(content: str, rows_markdown: str) -> str:
    pattern = re.compile(
        rf"{re.escape(START_MARKER)}.*?{re.escape(END_MARKER)}",
        re.DOTALL,
    )

    replacement = f"{START_MARKER}\n{rows_markdown}\n{END_MARKER}"
    new_content, count = pattern.subn(replacement, content, count=1)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one LVLLM table marker block, found {count}."
        )
    return new_content


def _sync_file(readme_path: Path, rows_markdown: str, *, check: bool) -> bool:
    original = readme_path.read_text(encoding="utf-8")
    updated = _replace_table_block(original, rows_markdown)
    changed = updated != original

    if changed and not check:
        readme_path.write_text(updated, encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize LVLLM environment-variable rows in README.md and "
            "README_cn.md using vllm/envs.py as the source of truth."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write files; return non-zero if updates are required.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    envs_module = _load_envs_module(repo_root)

    rows_by_name = {row.name: row for row in envs_module.LVLLM_README_DOC_ROWS}
    ordered_rows = [rows_by_name[name] for name in envs_module.LVLLM_README_TABLE_ORDER]

    en_rows = _render_rows(ordered_rows, lang="en")
    zh_rows = _render_rows(ordered_rows, lang="zh")

    changed_en = _sync_file(repo_root / "README.md", en_rows, check=args.check)
    changed_zh = _sync_file(repo_root / "README_cn.md", zh_rows, check=args.check)

    if args.check and (changed_en or changed_zh):
        print("LVLLM README env table is out of sync. Run sync_lvllm_readme_env_table.py")
        return 1

    if not args.check:
        print("Synchronized LVLLM README environment-variable tables.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

