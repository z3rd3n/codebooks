"""Combinatorial index codecs shared by Algorithms 1-4 of the paper.

All four protocol algorithms decode a single integer into a sorted index
combination using the same combinatorial number system:

    index = sum_i C(n_total - 1 - n_i, k - i)      for sorted n_0 < ... < n_{k-1}

* Algorithm 1: i_{1,2}  -> L beams out of N1*N2          (R15/R16/R18 Type II)
* Algorithm 2: beta_1   -> 4 restricted groups of O1*O2  (subset restriction)
* Algorithm 3: i_{1,6,l}-> M_v-1 taps out of N3-1 (or the 2*M_v window, N3>19)
* Algorithm 4: i_{1,2}  -> L ports out of P/2            (R17; corrected errata)

The paper's Algorithm 4 contains copy-paste typos (``C(x*, 4-k)`` and
``s_{k-1}``); the corrected form is the same generic codec.
"""

from __future__ import annotations

from math import comb


def combo_to_index(indices: list[int] | tuple[int, ...], n_total: int) -> int:
    """Encode a sorted combination into its 3GPP combinatorial index."""
    idx = sorted(indices)
    k = len(idx)
    if any(not 0 <= n < n_total for n in idx):
        raise ValueError(f"indices {idx} out of range [0, {n_total})")
    if len(set(idx)) != k:
        raise ValueError("indices must be distinct")
    return sum(comb(n_total - 1 - n, k - i) for i, n in enumerate(idx))


def index_to_combo(index: int, n_total: int, k: int) -> list[int]:
    """Decode a combinatorial index into its sorted combination.

    Generic form of the greedy loop in Algorithms 1-4: at step i, find the
    largest x* with index - s >= C(x*, k - i), then n_i = n_total - 1 - x*.
    """
    if not 0 <= index < comb(n_total, k):
        raise ValueError(f"index {index} out of range [0, C({n_total},{k}))")
    out = []
    s = 0
    for i in range(k):
        for x_star in range(n_total - 1 - i, k - 2 - i, -1):
            if index - s >= comb(x_star, k - i):
                break
        s += comb(x_star, k - i)
        out.append(n_total - 1 - x_star)
    return out


# ---------------------------------------------------------------------------
# Algorithm 1: spatial beam combination i_{1,2}
# ---------------------------------------------------------------------------

def encode_beam_combination(n1: list[int], n2: list[int], N1: int, N2: int) -> int:
    """(n1^(i), n2^(i)) -> i_{1,2}, with the flat index n = N1*n2 + n1."""
    flat = [N1 * b + a for a, b in zip(n1, n2, strict=True)]
    return combo_to_index(flat, N1 * N2)


def decode_beam_combination(i12: int, N1: int, N2: int, L: int) -> tuple[list[int], list[int]]:
    """i_{1,2} -> ({n1^(i)}, {n2^(i)}) per Algorithm 1."""
    flat = index_to_combo(i12, N1 * N2, L)
    n1 = [n % N1 for n in flat]
    n2 = [n // N1 for n in flat]
    return n1, n2


# ---------------------------------------------------------------------------
# Algorithm 2: codebook subset restriction group indicator beta_1
# ---------------------------------------------------------------------------

def encode_restriction_groups(r1: list[int], r2: list[int], O1: int, O2: int) -> int:
    """(r1^(k), r2^(k)) for k=0..3 -> beta_1, with g = O1*r2 + r1 (eq. a58/a59)."""
    g = [O1 * b + a for a, b in zip(r1, r2, strict=True)]
    if len(g) != 4:
        raise ValueError("exactly 4 restricted vector groups are configured")
    return combo_to_index(g, O1 * O2)


def decode_restriction_groups(beta1: int, O1: int, O2: int) -> tuple[list[int], list[int], list[int]]:
    """beta_1 -> (g^(k), r1^(k), r2^(k)) per Algorithm 2."""
    g = index_to_combo(beta1, O1 * O2, 4)
    r1 = [x % O1 for x in g]
    r2 = [x // O1 for x in g]
    return g, r1, r2


# ---------------------------------------------------------------------------
# Algorithm 3: delay-tap selection i_{1,6,l} (with two-level indication N3>19)
# ---------------------------------------------------------------------------

def encode_taps(n3: list[int], N3: int, Mv: int) -> tuple[int, int | None]:
    """Selected taps (already remapped, n3^(0)=0) -> (i_{1,6,l}, i_{1,5} or None).

    For N3 <= 19 the Mv-1 nonzero taps are encoded directly out of N3-1.
    For N3 > 19, the taps must fit in a cyclic window of 2*Mv taps
    {M_initial, ..., M_initial+2Mv-1} (mod N3) with M_initial in {-2Mv+1,..,0};
    the window-relative index of a tap t is t itself for the non-negative part
    and t-(N3-2Mv) for the wrapped part (inverse of Algorithm 3's mapping).
    """
    taps = sorted(n3)
    if taps[0] != 0:
        raise ValueError("taps must be remapped so the strongest tap is 0")
    if len(taps) != Mv:
        raise ValueError("expected Mv taps")
    rest = taps[1:]
    if N3 <= 19:
        i16 = combo_to_index([t - 1 for t in rest], N3 - 1)
        return i16, None
    for m_init in range(0, -2 * Mv, -1):
        if all(t <= m_init + 2 * Mv - 1 or t >= N3 + m_init for t in rest):
            rel = sorted(t if t <= m_init + 2 * Mv - 1 else t - (N3 - 2 * Mv) for t in rest)
            i16 = combo_to_index([r - 1 for r in rel], 2 * Mv - 1)
            i15 = m_init if m_init == 0 else m_init + 2 * Mv
            return i16, i15
    raise ValueError(f"taps {taps} do not fit any 2*Mv={2*Mv} window for N3={N3}")


def decode_taps(i16: int, N3: int, Mv: int, i15: int | None = None) -> list[int]:
    """(i_{1,6,l}, M_initial) -> remapped tap indices n3 (Algorithm 3).

    Returns the full sorted tap list including the implicit strongest tap 0.
    """
    if Mv == 1:
        return [0]
    if N3 <= 19:
        rest = [r + 1 for r in index_to_combo(i16, N3 - 1, Mv - 1)]
        return sorted([0] + rest)
    if i15 is None:
        raise ValueError("i_{1,5} is required when N3 > 19")
    m_initial = i15 if i15 == 0 else i15 - 2 * Mv
    rel = [r + 1 for r in index_to_combo(i16, 2 * Mv - 1, Mv - 1)]
    taps = []
    for n_rel in rel:  # n_l^(f) in 0..2Mv-1, window-relative as in Algorithm 3
        if n_rel <= m_initial + 2 * Mv - 1:
            taps.append(n_rel)
        else:
            taps.append(n_rel + (N3 - 2 * Mv))
    return sorted([0] + [t % N3 for t in taps])


# ---------------------------------------------------------------------------
# Algorithm 4 (corrected): free port selection i_{1,2} in R17
# ---------------------------------------------------------------------------

def encode_ports(m: list[int], P_csirs: int) -> int:
    """Selected ports m^(0..L-1) (per polarization) -> i_{1,2}."""
    return combo_to_index(m, P_csirs // 2)


def decode_ports(i12: int, P_csirs: int, L: int) -> list[int]:
    """i_{1,2} -> sorted ports m^(0..L-1) out of P_CSI-RS/2."""
    return index_to_combo(i12, P_csirs // 2, L)
