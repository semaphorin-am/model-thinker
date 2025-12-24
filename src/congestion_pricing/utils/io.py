"""File I/O utilities."""

import hashlib
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Parsed YAML content as dictionary
    """
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Save data to a YAML file.

    Args:
        data: Dictionary to save
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_checksum(path: str | Path, algorithm: str = "sha256") -> str:
    """Compute checksum of a file.

    Args:
        path: Path to file
        algorithm: Hash algorithm (default sha256)

    Returns:
        Hex digest of file hash
    """
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksum(path: str | Path, expected: str, algorithm: str = "sha256") -> bool:
    """Verify file checksum.

    Args:
        path: Path to file
        expected: Expected hash value
        algorithm: Hash algorithm

    Returns:
        True if checksum matches
    """
    actual = compute_checksum(path, algorithm)
    return actual == expected


def load_checksums(path: str | Path) -> dict[str, str]:
    """Load checksums from a file.

    Args:
        path: Path to checksums file (format: "hash  filename")

    Returns:
        Dictionary mapping filename to hash
    """
    checksums = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split(None, 1)
                if len(parts) == 2:
                    checksums[parts[1]] = parts[0]
    return checksums


def save_checksums(checksums: dict[str, str], path: str | Path) -> None:
    """Save checksums to a file.

    Args:
        checksums: Dictionary mapping filename to hash
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for filename, checksum in sorted(checksums.items()):
            f.write(f"{checksum}  {filename}\n")
