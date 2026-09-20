"""Restore missing analysis inputs from single or multipart gzip archives."""

import gzip
import io
import json
import shutil
from pathlib import Path
from uuid import uuid4

from .paths import PROJECT_ROOT


class _JoinedParts(io.RawIOBase):
    """Read ordered archive parts as one continuous stream without joining on disk."""

    def __init__(self, paths):
        super().__init__()
        self._paths = iter(paths)
        self._current = None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self._current is None:
                path = next(self._paths, None)
                if path is None:
                    return 0
                self._current = path.open("rb")
            count = self._current.readinto(buffer)
            if count:
                return count
            self._current.close()
            self._current = None

    def close(self):
        if self._current is not None:
            self._current.close()
        super().close()


def _inside(directory, relative):
    """Keep manifest paths inside their designated data directory."""
    directory = directory.resolve()
    path = (directory / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(directory):
        raise ValueError(f"Archive path escapes its data directory: {relative}")
    return path


def _restore_entry(root, archive_dir, entry):
    destination = _inside(root, entry["path"])
    if not destination.is_relative_to(root / "data"):
        raise ValueError(f"Bundled inputs must be under data/: {entry['path']}")
    part_paths = [_inside(archive_dir, part["path"]) for part in entry["parts"]]
    if not part_paths:
        raise ValueError(f"No archive parts listed for {entry['path']}")
    for path in part_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing data archive: {path}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.partial")
    try:
        with (
            _JoinedParts(part_paths) as parts,
            io.BufferedReader(parts) as joined,
            gzip.GzipFile(fileobj=joined, mode="rb") as decoded,
            temporary.open("xb") as output,
        ):
            shutil.copyfileobj(decoded, output, length=1024 * 1024)
        if destination.exists():
            raise FileExistsError(
                f"An input appeared during restoration; keeping it unchanged: {destination}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def restore_data(*, root=None, files=None):
    """Unpack selected repository-relative inputs, leaving existing files in place."""
    root = Path(root or PROJECT_ROOT).resolve()
    archive_dir = root / "data" / "archives"
    manifest = json.loads((archive_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1 or manifest.get("compression") != "gzip":
        raise ValueError("Unsupported data archive format.")
    entries = {entry["path"]: entry for entry in manifest["files"]}
    selected = list(entries) if files is None else list(files)
    unknown = set(selected) - entries.keys()
    if unknown:
        raise ValueError(f"No bundled archive for: {sorted(unknown)}")

    report = []
    for relative in selected:
        entry = entries[relative]
        destination = _inside(root, relative)
        if destination.exists():
            status = "already present"
        else:
            _restore_entry(root, archive_dir, entry)
            status = "restored"
            print(f"Restored {relative}", flush=True)
        report.append({"path": relative, "status": status})
    return report
