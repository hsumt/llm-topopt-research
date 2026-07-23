"""
Bundle every .py and .txt file under llm-topopt-research/project into a small
number of plain-text files for ChatGPT upload.

Expected location:
    llm-topopt-research/bundle_project_for_chatgpt.py

Run from the repository root:
    python bundle_project_for_chatgpt.py

Optional:
    python bundle_project_for_chatgpt.py --project-dir project --max-chars 750000

Output:
    _chatgpt_bundle/
        PROJECT_PART_001.txt
        PROJECT_PART_002.txt
        ...
        MANIFEST.txt

Only files ending in .py or .txt under the selected project directory are
included. Relative paths and file boundaries are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


INCLUDED_SUFFIXES = {".py", ".txt"}
OUTPUT_DIRECTORY_NAME = "_chatgpt_bundle"


@dataclass(frozen=True)
class SourceFile:
    path: Path
    relative_path: str
    text: str
    sha256: str
    size_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bundle project/**/*.py and project/**/*.txt for ChatGPT."
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("project"),
        help=(
            "Path to the project directory. Default: ./project, assuming the "
            "script is run from the llm-topopt-research repository root."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(OUTPUT_DIRECTORY_NAME),
        help=f"Output directory. Default: ./{OUTPUT_DIRECTORY_NAME}",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=750_000,
        help="Approximate maximum characters per bundle part. Default: 750000.",
    )
    return parser.parse_args()


def read_text_safely(path: Path) -> str:
    raw = path.read_bytes()

    if b"\x00" in raw:
        raise ValueError(f"Probable binary file despite text suffix: {path}")

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8-sig")


def collect_source_files(project_dir: Path, output_dir: Path) -> list[SourceFile]:
    files: list[SourceFile] = []
    output_dir = output_dir.resolve()

    for path in sorted(project_dir.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in INCLUDED_SUFFIXES:
            continue

        resolved = path.resolve()

        # Prevent a bundle placed inside project/ from including itself.
        if output_dir == resolved or output_dir in resolved.parents:
            continue

        try:
            text = read_text_safely(path)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError(f"Could not include {path}: {exc}") from exc

        relative_path = path.relative_to(project_dir).as_posix()
        raw_utf8 = text.encode("utf-8")

        files.append(
            SourceFile(
                path=path,
                relative_path=relative_path,
                text=text,
                sha256=hashlib.sha256(raw_utf8).hexdigest(),
                size_bytes=len(raw_utf8),
            )
        )

    return files


def render_file_block(source: SourceFile) -> str:
    trailing_newline = "" if source.text.endswith("\n") else "\n"

    return (
        f"\n{'=' * 88}\n"
        f"BEGIN FILE: project/{source.relative_path}\n"
        f"SHA256: {source.sha256}\n"
        f"SIZE_BYTES: {source.size_bytes}\n"
        f"{'=' * 88}\n"
        f"{source.text}{trailing_newline}"
        f"{'-' * 88}\n"
        f"END FILE: project/{source.relative_path}\n"
        f"{'-' * 88}\n"
    )


def split_into_parts(
    files: Iterable[SourceFile],
    max_chars: int,
) -> list[list[SourceFile]]:
    if max_chars <= 10_000:
        raise ValueError("--max-chars must be greater than 10000")

    parts: list[list[SourceFile]] = []
    current_part: list[SourceFile] = []
    current_chars = 0

    for source in files:
        block_chars = len(render_file_block(source))

        if current_part and current_chars + block_chars > max_chars:
            parts.append(current_part)
            current_part = []
            current_chars = 0

        current_part.append(source)
        current_chars += block_chars

    if current_part:
        parts.append(current_part)

    return parts


def write_parts(
    output_dir: Path,
    project_dir: Path,
    parts: list[list[SourceFile]],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for index, part in enumerate(parts, start=1):
        output_path = output_dir / f"PROJECT_PART_{index:03d}.txt"

        header = (
            "CHATGPT PROJECT SOURCE SNAPSHOT\n"
            f"Source directory: {project_dir.resolve()}\n"
            "Included suffixes: .py, .txt\n"
            f"Part: {index} of {len(parts)}\n"
            f"Files in this part: {len(part)}\n"
            "\n"
            "Each original file is enclosed by BEGIN FILE / END FILE markers.\n"
            "Paths are relative to the repository's project directory.\n"
        )

        body = "".join(render_file_block(source) for source in part)
        output_path.write_text(header + body, encoding="utf-8")
        written.append(output_path)

    return written


def write_manifest(
    output_dir: Path,
    project_dir: Path,
    files: list[SourceFile],
    parts: list[list[SourceFile]],
) -> Path:
    manifest_path = output_dir / "MANIFEST.txt"

    lines = [
        "CHATGPT PROJECT SOURCE BUNDLE MANIFEST",
        f"Source directory: {project_dir.resolve()}",
        "Included suffixes: .py, .txt",
        f"Source files included: {len(files)}",
        f"Bundle parts: {len(parts)}",
        "",
        "UPLOAD",
        "Upload MANIFEST.txt and every PROJECT_PART_*.txt file together.",
        "Treat the set as one complete snapshot of llm-topopt-research/project.",
        "",
        "FILES BY PART",
    ]

    for index, part in enumerate(parts, start=1):
        lines.append("")
        lines.append(f"[PROJECT_PART_{index:03d}.txt]")
        for source in part:
            lines.append(
                f"project/{source.relative_path} | "
                f"{source.size_bytes} bytes | "
                f"sha256={source.sha256}"
            )

    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path


def main() -> int:
    args = parse_args()

    project_dir = args.project_dir.resolve()
    output_dir = args.output.resolve()

    if not project_dir.exists() or not project_dir.is_dir():
        raise SystemExit(
            f"Project directory not found: {project_dir}\n"
            "Run this script from llm-topopt-research or pass "
            "--project-dir /path/to/llm-topopt-research/project"
        )

    files = collect_source_files(project_dir, output_dir)

    if not files:
        raise SystemExit(
            f"No .py or .txt files found under {project_dir}"
        )

    parts = split_into_parts(files, args.max_chars)
    written_parts = write_parts(output_dir, project_dir, parts)
    manifest_path = write_manifest(
        output_dir, project_dir, files, parts
    )

    total_bytes = sum(source.size_bytes for source in files)

    print("\nBundle complete")
    print(f"  Project directory: {project_dir}")
    print(f"  Included types:    .py, .txt")
    print(f"  Files included:    {len(files)}")
    print(f"  Source bytes:      {total_bytes}")
    print(f"  Bundle parts:      {len(written_parts)}")
    print(f"  Output directory:  {output_dir}")
    print(f"  Manifest:          {manifest_path.name}")

    for path in written_parts:
        print(f"  Part:              {path.name}")

    print("\nUpload MANIFEST.txt and all PROJECT_PART_*.txt files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())