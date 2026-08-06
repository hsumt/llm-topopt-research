"""Safely clear generated topology-optimization runtime artifacts.

Current output location:

    artifacts/

Run:

    /dolfinx-env/bin/python -m project.tools.clear_outputs

The script preserves the output root, immediate case directories, and
.gitkeep files. It refuses to clean a directory not named ``_05_OUT``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from project.paths import ARTIFACT_ROOT

# Keep these marker/configuration files wherever they appear.
PRESERVE_FILENAMES = {".gitkeep"}

# Create .gitkeep in every immediate child folder after cleaning.
CREATE_GITKEEP = True


def locate_output_directory() -> Path:
    """Return the generated-artifact root or fail clearly."""
    output_root = ARTIFACT_ROOT.resolve()

    if output_root.is_dir():
        return output_root

    raise FileNotFoundError(
        "Could not locate the generated-artifact directory:\n"
        f"  {output_root}\n\n"
        "The directory has not been created or the repository structure "
        "is not the expected one."
    )


def remove_item(path: Path) -> tuple[int, int]:
    """
    Remove one file, symlink, or directory.

    Returns:
        (files_removed, directories_removed)
    """
    if path.is_symlink() or path.is_file():
        path.unlink()
        return 1, 0

    if path.is_dir():
        file_count = sum(
            1 for item in path.rglob("*")
            if item.is_file() or item.is_symlink()
        )
        directory_count = sum(1 for item in path.rglob("*") if item.is_dir()) + 1
        shutil.rmtree(path)
        return file_count, directory_count

    return 0, 0


def clear_directory_contents(
    directory: Path,
    preserve_filenames: set[str],
) -> tuple[int, int]:
    """Delete all contents of a directory except explicitly preserved files."""
    files_removed = 0
    directories_removed = 0

    for item in list(directory.iterdir()):
        if item.name in preserve_filenames and item.is_file():
            continue

        removed_files, removed_directories = remove_item(item)
        files_removed += removed_files
        directories_removed += removed_directories

    return files_removed, directories_removed


def main() -> None:
    output_root = locate_output_directory()

    # Hard safety guard: never recursively clean an unexpectedly named path.
    if output_root.name != "artifacts":
        raise RuntimeError(
            f"Safety check failed: refusing to clean unexpected path {output_root}"
        )

    print(f"Cleaning generated outputs under:\n  {output_root}\n")

    files_removed = 0
    directories_removed = 0
    preserved_case_folders: list[str] = []

    # Snapshot immediate children before deleting anything.
    immediate_children = list(output_root.iterdir())

    for child in immediate_children:
        if child.is_dir() and not child.is_symlink():
            preserved_case_folders.append(child.name)

            removed_files, removed_directories = clear_directory_contents(
                child,
                PRESERVE_FILENAMES,
            )
            files_removed += removed_files
            directories_removed += removed_directories

            if CREATE_GITKEEP:
                (child / ".gitkeep").touch(exist_ok=True)

            print(f"  Cleared: {child.name}/")
        elif child.name not in PRESERVE_FILENAMES:
            removed_files, removed_directories = remove_item(child)
            files_removed += removed_files
            directories_removed += removed_directories
            print(f"  Removed loose item: {child.name}")

    print("\nCleanup complete.")
    print(f"  Files removed:       {files_removed}")
    print(f"  Directories removed: {directories_removed}")
    print(f"  Case folders kept:   {len(preserved_case_folders)}")

    if preserved_case_folders:
        print("  Preserved folders:")
        for name in sorted(preserved_case_folders):
            print(f"    - {name}/")


if __name__ == "__main__":
    main()