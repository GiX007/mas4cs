"""
Project statistics utility.

Reports .py file count, line counts, and project size — excluding venv.
Run directly: python scripts/project_stats.py
"""

from pathlib import Path


def count_py_files(root: str = ".") -> int:
    """
    Count all .py files in the project, excluding venv.

    Args: root - starting directory (default: current)
    Returns: number of .py files
    """
    return sum(1 for f in Path(root).rglob("*.py") if "venv" not in f.parts)


def count_total_lines(root: str = ".") -> int:
    """
    Count total lines across all .py files, excluding venv.

    Args: root - starting directory (default: current)
    Returns: total line count
    """
    total = 0
    for f in Path(root).rglob("*.py"):
        if "venv" not in f.parts:
            total += len(f.read_text(encoding="utf-8", errors="ignore").splitlines())
    return total


def count_non_blank_lines(root: str = ".") -> int:
    """
    Count non-blank lines across all .py files, excluding venv and hf_cache.

    Args: root - starting directory (default: current)
    Returns: total non-blank line count
    """
    total = 0
    for f in Path(root).rglob("*.py"):
        if "venv" not in f.parts and "hf_cache" not in f.parts:
            all_lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
            total += sum(1 for line in all_lines if line.strip())
    return total


def lines_per_file(root: str = ".") -> dict[str, int]:
    """
    Count lines per .py file, excluding venv.

    Args: root - starting directory (default: current)
    Returns: dict of {filename: line_count}, sorted descending
    """
    result = {}
    for f in Path(root).rglob("*.py"):
        if "venv" not in f.parts:
            result[str(f)] = len(f.read_text(encoding="utf-8", errors="ignore").splitlines())
    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))


def project_size_mb(root: str = ".") -> float:
    """
    Calculate total project size in MB, excluding venv.

    Args: root - starting directory (default: current)
    Returns: size in MB rounded to 2 decimals
    """
    total_bytes = sum(
        f.stat().st_size for f in Path(root).rglob("*")
        if f.is_file() and "venv" not in f.parts and "hf_cache" not in f.parts
    )
    return round(total_bytes / (1024 * 1024), 2)


if __name__ == "__main__":
    print(f"Python files: {count_py_files()}")
    print(f"Total lines: {count_total_lines()}")
    print(f"Non-blank lines: {count_non_blank_lines()}")
    print(f"Project size: {project_size_mb()} MB")
    print("\nLines per file:")
    for path, lines in lines_per_file().items():
        print(f"  {lines:>5}  {path}")


# ALTERNATIVE: Quick terminal estimates (Python scripts for official stats)
# NOTE: PowerShell's Get-Content effectively counts non-blank lines only, due to how it handles blank lines and UTF-8 encoded files.
# The small remaining difference (~10-20 lines) is from UTF-8 special characters.

# Total lines (excluding venv):
# (Get-ChildItem -Recurse -Filter *.py | Where-Object { $_.FullName -notmatch '\\venv\\' } | Get-Content | Measure-Object -Line).Lines

# Lines per file (excluding venv):
# Get-ChildItem -Recurse -Filter *.py | Where-Object { $_.FullName -notmatch '\\venv\\' } | ForEach-Object { "$($_.Name): $((Get-Content $_.FullName | Measure-Object -Line).Lines)" }

# File count (excluding venv):
# (Get-ChildItem -Recurse -Filter *.py | Where-Object { $_.FullName -notmatch '\\venv\\' }).Count
