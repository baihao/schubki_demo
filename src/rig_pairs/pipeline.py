from __future__ import annotations

from pathlib import Path

from .config import DefaultPairsOutput, KeyframeConfig, PairsConfig
from .keyframes import load_synced_stereo, select_keyframes, write_keyframes_json
from .pairs import build_pair_lists, write_pair_files


def generate_pairs_for_test_data(
    output_dir: Path | None = None,
    keyframe_config: KeyframeConfig | None = None,
    pairs_config: PairsConfig | None = None,
) -> tuple[list, object, dict[str, Path]]:
    """
    Sync ``test_data`` stereo → INS keyframes → hloc ``pairs-*.txt`` + ``keyframes.json``.
    """
    kc = keyframe_config or KeyframeConfig()
    pc = pairs_config or PairsConfig()
    out = Path(output_dir or DefaultPairsOutput().output_dir)

    synced = load_synced_stereo(kc.left_dir, kc.right_dir)
    keyframes = select_keyframes(synced, kc)
    lists = build_pair_lists(keyframes, pc, kc)

    write_keyframes_json(out / "keyframes.json", keyframes)
    paths = write_pair_files(out, lists)
    return keyframes, lists, paths
