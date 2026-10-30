#!/usr/bin/env python3
"""Bundle a Git project into a ZIP, respecting .gitignore (even for tracked files).

Run from anywhere:
    python scripts/bundle_project.py
    python scripts/bundle_project.py --output /tmp/project_context.zip

Requirements: Python 3.10+ and Git. No third-party dependencies.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def git(root: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode not in (0, 1) or (result.returncode == 1 and args[0] != "check-ignore"):
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def bundle(root: Path, output: Path) -> int:
    root = Path(os.path.realpath(root))
    output = output.resolve()
    # Includes tracked and untracked non-ignored files; excludes Git internals.
    candidates = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    names = sorted(set(name for name in candidates.split(b"\0") if name))

    # Unlike `git ls-files`, this also filters *tracked* files matching current
    # .gitignore rules, useful when creating a shareable project snapshot.
    ignored = (
        set(
            name
            for name in git(
                root,
                "check-ignore",
                "--no-index",
                "--stdin",
                "-z",
                input_bytes=b"\0".join(names) + b"\0" if names else b"",
            ).split(b"\0")
            if name
        )
        if names
        else set()
    )

    count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for raw_name in names:
            if raw_name in ignored:
                continue
            relative = Path(os.fsdecode(raw_name))
            file_path = root / relative
            # Skip symlinks (including links to files outside the repository),
            # missing files, directories, and the archive itself.
            if file_path.is_symlink() or not file_path.is_file():
                continue
            if file_path.resolve() == output:
                continue
            archive.write(file_path, arcname=relative.as_posix())
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="ZIP output path (default: <project-root>/project_context.zip)",
    )
    args = parser.parse_args()
    try:
        root_string = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
        root = Path(root_string)
        output = args.output or root / "project_context.zip"
        count = bundle(root, output)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Created {output.resolve()} with {count} files")
    print("Review the ZIP before sharing; .gitignore is not a secret scanner.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
