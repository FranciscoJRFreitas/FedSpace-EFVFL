# utils/connectivity.py
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

def load_connectivity(rel_or_abs_path: str) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """
    Parameters
    ----------
    path : str
        Path to the pickled list-of-lists you described:
        length = 96   # 15-min slots in 24 h
        slot[i] == [satellite_id_1, satellite_id_2, ...].

    Returns
    -------
    schedule : dict[int, list[int]]
        slot_index → list of *internal* client indices (0 … N-1).
    id2idx : dict[int, int]
        satellite_id → internal client index.  Useful elsewhere.
    """
    p = Path(rel_or_abs_path)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()

    if not p.exists():
        raise FileNotFoundError(f"Connectivity file not found: {p}")

    with p.open("rb") as f:
        raw: List[List[int]] = pickle.load(f)

    unique_ids = sorted({sid for slot in raw for sid in slot})#[:4]   # keep only first 4
    id2idx = {sid: idx for idx, sid in enumerate(unique_ids)}

    # filter out satellites we dropped
    schedule = {
        i: [id2idx[sid] for sid in slot if sid in id2idx]
        for i, slot in enumerate(raw)
    }
    return schedule, id2idx
