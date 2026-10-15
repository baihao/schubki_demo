from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from hloc import pairs_from_exhaustive


def write_exhaustive_pairs(
    output: Path,
    image_list: Union[Path, List[str]],
    *,
    ref_list: Optional[Union[Path, List[str]]] = None,
) -> Path:
    """
    Wrap ``hloc.pairs_from_exhaustive.main``.

    - Mapping: ``ref_list=None`` → all pairs among ``image_list``.
    - Localization: ``image_list=[query]``, ``ref_list=registered_refs``.
    """
    pairs_from_exhaustive.main(Path(output), image_list=image_list, ref_list=ref_list)
    return Path(output)


def write_pairs_from_file(source: Path, destination: Path) -> Path:
    """Copy or symlink an existing pairs file (e.g. from ``rig_pairs``)."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(Path(source).read_text(encoding="utf-8"), encoding="utf-8")
    return destination


def read_pairs(path: Path) -> List[tuple[str, str]]:
    pairs: List[tuple[str, str]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Invalid pair line in {path}: {line!r}")
        pairs.append((parts[0], parts[1]))
    return pairs


def unique_image_names_from_pairs(
    pairs_path: Path,
    *,
    sort: bool = True,
) -> List[str]:
    """Collect unique hloc image names referenced in a pairs file."""
    seen: set[str] = set()
    for a, b in read_pairs(pairs_path):
        seen.add(a)
        seen.add(b)
    names = list(seen)
    if sort:
        names.sort()
    return names
