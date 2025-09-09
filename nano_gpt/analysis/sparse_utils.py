# nano_gpt/analysis/sparse_utils.py
import torch
from typing import Iterable, List, Tuple, Optional, Literal

Side = Literal["input", "output"]

def load_sparse_checkpoint(path: str):
    """
    Returns the checkpoint dict saved by SparseEmbeddingModel.save_sparse_checkpoint.
    Keys: 't','V','D','Z_proj','U_proj', optionally 'tokens','meta'
    """
    ckpt = torch.load(path, map_location="cpu")
    # sanity
    for k in ["t","V","D","Z_proj","U_proj"]:
        if k not in ckpt:
            raise ValueError(f"Missing key '{k}' in checkpoint")
    return ckpt

def words_active_at_coord(
    ckpt: dict,
    coord: int,
    side: Side = "input",
    sort_desc: bool = True,
    k: Optional[int] = None,
    min_abs: float = 0.0,
) -> List[Tuple[str, float, int]]:
    """
    For a given coordinate i, list tokens with nonzero value at that coordinate,
    sorted by value (desc by default).

    Returns list of tuples: (token_str, value, token_id)
    """
    V, D = int(ckpt["V"]), int(ckpt["D"])
    if not (0 <= coord < D):
        raise IndexError(f"coord {coord} out of range [0,{D})")

    mat = ckpt["Z_proj"] if side == "input" else ckpt["U_proj"]  # (V, D)
    col = mat[:, coord]  # (V,)

    # active = nonzero after projection (exact zeros)
    nz = (col != 0)
    ids = torch.nonzero(nz, as_tuple=False).flatten().tolist()

    toks = ckpt.get("tokens", None)
    out: List[Tuple[str, float, int]] = []
    for tid in ids:
        val = float(col[tid].item())
        if abs(val) < min_abs:
            continue
        name = toks[tid] if toks is not None and tid < len(toks) else str(tid)
        out.append((name, val, tid))

    out.sort(key=lambda x: x[1], reverse=sort_desc)
    if k is not None:
        out = out[:k]
    return out

def top_coords_for_word(
    ckpt: dict,
    token_id: int,
    side: Side = "input",
    k: Optional[int] = None,
) -> List[Tuple[int, float]]:
    """
    For a given token id, returns list of (coord_index, value) where projected vector is nonzero,
    sorted by |value| desc. If k given, returns top-k.
    """
    V, D = int(ckpt["V"]), int(ckpt["D"])
    if not (0 <= token_id < V):
        raise IndexError(f"token_id {token_id} out of range [0,{V})")

    row = (ckpt["Z_proj"] if side == "input" else ckpt["U_proj"])[token_id]  # (D,)
    ids = torch.nonzero(row != 0, as_tuple=False).flatten().tolist()
    items = [(i, float(row[i].item())) for i in ids]
    items.sort(key=lambda x: abs(x[1]), reverse=True)
    if k is not None:
        items = items[:k]
    return items
