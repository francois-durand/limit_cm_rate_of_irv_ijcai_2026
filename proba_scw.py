from typing import Callable, Tuple, Iterable, Mapping, OrderedDict
from collections import OrderedDict
import numpy as np
from scipy.stats import qmc
from scipy.special import erfinv


def _masks_for_nonempty_subsets(m: int) -> np.ndarray:
    """
    Generate bitmasks of all non-empty subsets of {1, ..., m-1}.

    Each subset is encoded as an unsigned 64-bit integer, where the
    (m-1) least significant bits represent membership of elements
    1 through m-1. For example, with m=4:
    - The subset {1} is encoded as binary 001 -> 1
    - The subset {2} is encoded as binary 010 -> 2
    - The subset {1,3} is encoded as binary 101 -> 5

    The function returns all integers from 1 to 2^(m-1)-1 inclusive,
    i.e. all non-empty subsets.

    Parameters
    ----------
    m : int
        Number of elements plus one. The subsets are drawn from
        {1, ..., m-1}. Must be >= 2.

    Returns
    -------
    masks : np.ndarray of dtype uint64
        Array of length 2^(m-1)-1 containing the bitmasks of all
        non-empty subsets of {1, ..., m-1}, sorted in increasing order.

    Raises
    ------
    ValueError
        If m < 2 or if m-1 > 63 (since only 63 bits fit into uint64).

    Examples
    --------
    >>> _masks_for_nonempty_subsets(3)
    array([1, 2, 3], dtype=uint64)
    >>> [bin(x) for x in _masks_for_nonempty_subsets(3)]
    ['0b1', '0b10', '0b11']
    """
    if m < 2:
        raise ValueError("m must be >= 2 (otherwise {1,...,m-1} is empty).")
    nbits = m - 1
    if nbits > 63:
        raise ValueError("This implementation only supports up to 63 bits (uint64).")
    # Return integers from 1 to (2^(m-1)-1), each representing a non-empty subset
    return np.arange(1, 1 << nbits, dtype=np.uint64)


def _popcount_uint64(arr: np.ndarray) -> np.ndarray:
    """
    Count the number of set bits (population count) in an array of uint64.

    This function uses NumPy's `unpackbits` to expand each 64-bit integer
    into its binary representation (as 8 bytes = 64 bits), then sums the
    bits across each row.

    Parameters
    ----------
    arr : np.ndarray
        Input array of dtype uint64. Can be 1D or higher-dimensional;
        the result will always be 1D with length equal to arr.size.

    Returns
    -------
    counts : np.ndarray of dtype int64
        1D array of length equal to arr.size, where each entry is the
        number of bits set to 1 in the corresponding uint64.

    Examples
    --------
    >>> import numpy as np
    >>> _popcount_uint64(np.array([0, 1, 3, 7], dtype=np.uint64))
    array([0, 1, 2, 3])

    >>> _popcount_uint64(np.array([255], dtype=np.uint64))  # 8 ones
    array([8])
    """
    # Interpret each uint64 as 8 bytes
    bytes_view = arr.view(np.uint8).reshape(-1, 8)   # shape (N, 8)
    # Expand to 64 bits per number
    bits = np.unpackbits(bytes_view, axis=1)         # shape (N, 64)
    # Sum bits along the 64 positions
    return bits.sum(axis=1).astype(np.int64)         # shape (N,)


def compute_H(m: int) -> Tuple[np.ndarray, np.ndarray, Callable[[int], set]]:
    """
    Build the matrix H indexed by non-empty subsets of {1, ..., m-1}.

    The matrix is of size ``d x d`` with ``d = 2^(m-1) - 1``. Rows and columns
    are indexed by non-empty subsets ``S, T ⊆ {1, ..., m-1}`` (encoded as
    bitmasks in increasing order). For each pair (S, T), the entry is
    defined by
        H[S,T] = 1 / (|S ∪ T| + 1)  -  (1 / (|S| + 1)) * (1 / (|T| + 1)).

    This implementation uses bitmasks and vectorized operations to compute
    union cardinalities via bitwise OR and population counts.

    Parameters
    ----------
    m : int
        The ground set is {1, ..., m-1}. Must satisfy m >= 2. For memory and
        dtype reasons, we also require m-1 <= 63 so that bitmasks fit in uint64.

    Returns
    -------
    H : np.ndarray, shape (d, d), dtype float64
        The matrix defined above, ordered according to the bitmask order of
        the non-empty subsets of {1, ..., m-1}.
    masks : np.ndarray, shape (d,), dtype uint64
        The bitmask for each row/column index. The i-th row/column corresponds
        to subset ``to_set(int(masks[i]))``.
    to_set : callable
        A helper function ``to_set(mask: int) -> set[int]`` that converts a
        bitmask into the corresponding subset of {1, ..., m-1}.

    Raises
    ------
    ValueError
        If m < 2, or if m-1 > 63 (bitmask does not fit in uint64).

    Notes
    -----
    Complexity is ``O(d^2)`` in both time and memory with
    ``d = 2^(m-1) - 1``. This is fine for small/moderate m but becomes
    expensive as m grows (e.g., m=20 gives d=524,287).

    Examples
    --------
    For m = 3, the non-empty subsets of {1,2} are {1}, {2}, {1,2}, so d=3.
    The resulting H (rounded) is:

    >>> H, masks, to_set = compute_H(3)
    >>> H.shape
    (3, 3)
    >>> masks
    array([1, 2, 3], dtype=uint64)
    >>> [to_set(int(x)) for x in masks]
    [{1}, {2}, {1, 2}]
    >>> import numpy as _np
    >>> _expected = _np.array([
    ...     [0.25      , 0.08333333, 0.16666667],
    ...     [0.08333333, 0.25      , 0.16666667],
    ...     [0.16666667, 0.16666667, 0.22222222],
    ... ])
    >>> _np.allclose(H, _expected, rtol=1e-6, atol=1e-6)
    True
    """
    # Get all non-empty subset bitmasks for {1, ..., m-1}
    masks = _masks_for_nonempty_subsets(m)  # shape (d,)
    d = masks.size

    # |S| for all subsets S (vectorized popcount on masks)
    card = _popcount_uint64(masks).astype(np.float64)  # shape (d,)
    inv_card_plus_1 = 1.0 / (card + 1.0)               # shape (d,)

    # |S ∪ T| via bitwise OR on the bitmasks (broadcast to dxd), then popcount
    union_masks = np.bitwise_or(masks[:, None], masks[None, :])  # (d, d) uint64
    union_card = _popcount_uint64(union_masks.ravel()).reshape(d, d).astype(np.float64)
    inv_union_plus_1 = 1.0 / (union_card + 1.0)        # shape (d, d)

    # Outer product of 1/(|S|+1) and 1/(|T|+1)
    outer_prod = np.multiply.outer(inv_card_plus_1, inv_card_plus_1)  # (d, d)

    # Final matrix
    H = inv_union_plus_1 - outer_prod

    def to_set(mask: int) -> set[int]:
        """Convert a bitmask to the corresponding subset of {1, ..., m-1}."""
        S: set[int] = set()
        bit = 1
        for i in range(1, m):  # positions 1..m-1
            if mask & bit:
                S.add(i)
            bit <<= 1
        return S

    return H, masks, to_set


def _next_pow2(n: int) -> int:
    """
    Return the smallest power of two greater than or equal to n.

    This helper is often used to ensure the number of samples generated
    by a Sobol sequence is a power of two, since Sobol balance properties
    are guaranteed only for such sizes.

    Parameters
    ----------
    n : int
        Positive integer.

    Returns
    -------
    p2 : int
        The smallest integer power of two greater than or equal to `n`.

    Examples
    --------
    >>> _next_pow2(1)
    1
    >>> _next_pow2(5)
    8
    >>> _next_pow2(16)
    16
    >>> _next_pow2(17)
    32
    """
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _orthant_prob_qmc(
    H: np.ndarray,
    n_samples: int = 200_000,
    seed: int | None = None,
    batch: int = 50_000
) -> Tuple[float, float]:
    """
    Estimate p = P[X > 0] for X ~ N(0, H) using scrambled Sobol QMC + Cholesky.

    The estimator draws quasi-random Sobol points `u` in [0, 1]^d, maps them to
    standard normals via the inverse CDF, then applies a Cholesky factor of `H`
    to obtain X ~ N(0, H). It returns the quasi-MC estimate of the orthant
    probability p along with a conservative MC-style standard error.

    To avoid the Sobol balance-property warning, the function automatically
    rounds the *total* number of generated points up to the nearest power of two,
    but only the first `n_samples` are used in the estimate.

    Parameters
    ----------
    H : np.ndarray, shape (d, d)
        Symmetric positive definite covariance matrix.
    n_samples : int, default=200_000
        Number of quasi-random samples effectively used in the estimate.
        Internally, generation is rounded up to a power of two to preserve
        Sobol balance properties.
    seed : int or None, default=None
        Seed for the scrambled Sobol engine (reproducibility).
    batch : int, default=50_000
        Processing block-size. For memory efficiency, the transformation from
        uniforms to X is performed in batches. The function will internally use
        a power-of-two batch size to avoid warnings.

    Returns
    -------
    p_hat : float
        Quasi–Monte Carlo estimate of P[X > 0] (component-wise positivity).
    stderr : float
        Conservative MC-style standard error: sqrt(p_hat * (1 - p_hat) / n_samples).

    Raises
    ------
    np.linalg.LinAlgError
        If `H` is not symmetric positive definite (Cholesky fails).
    ValueError
        If input shapes or parameters are invalid (e.g., non-square `H`,
        non-positive `n_samples` or `batch`).

    Notes
    -----
    - Complexity per batch is O(d^2 * batch) due to the matrix-vector product
      `X = Z @ L^T`, where `L` is the Cholesky factor of `H`.
    - The returned standard error is a conservative bound; in practice, QMC
      often achieves lower variance than plain MC.
    - The Sobol engine's balance property is guaranteed when *generation counts*
      are powers of two. We therefore generate a total of `n_pow2 = 2^k >= n_samples`
      points in power-of-two blocks and use only the first `n_samples`.

    Examples
    --------
    For an identity covariance in 2D, the true orthant probability is 0.25.
    With QMC and a modest sample size, we should get reasonably close.

    >>> import numpy as _np
    >>> H = _np.eye(2)
    >>> p_hat, stderr = _orthant_prob_qmc(H, n_samples=16384, seed=123, batch=4096)
    >>> round(p_hat, 3)  # doctest: +ELLIPSIS
    0.25...
    >>> # A loose check that we're within ~2% absolute error for this small n
    >>> abs(p_hat - 0.25) < 0.02
    True
    """
    H = np.asarray(H, dtype=float)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be a square 2D array.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if batch <= 0:
        raise ValueError("batch must be positive.")

    # Symmetrize numerically; then factorize (also checks SPD).
    H = 0.5 * (H + H.T)
    L = np.linalg.cholesky(H)
    d = H.shape[0]

    # Prepare Sobol engine (scrambled for variance reduction).
    rng = qmc.Sobol(d, scramble=True, seed=seed)

    # Use power-of-two totals and batches to avoid Sobol warnings
    n_pow2 = _next_pow2(n_samples)
    batch_pow2 = min(_next_pow2(batch), n_pow2)

    total_generated = 0
    total_used = 0
    count_positive = 0

    # Process in power-of-two blocks until we generate n_pow2 points.
    while total_generated < n_pow2:
        b = min(batch_pow2, n_pow2 - total_generated)  # this is a power of two
        u = rng.random(b)  # (b, d) in [0, 1]^d, no warning since b is 2^k
        # Map uniforms to standard normals via inverse CDF
        z = np.sqrt(2.0) * erfinv(2.0 * u - 1.0)  # (b, d)
        # X ~ N(0, H) via Cholesky
        x = z @ L.T  # (b, d)

        # Only count up to n_samples points (we may have over-generated)
        can_use = min(b, n_samples - total_used)
        if can_use > 0:
            # Check positivity only on the prefix we keep
            count_positive += np.sum(np.all(x[:can_use] > 0.0, axis=1))
            total_used += can_use

        total_generated += b

    # Compute estimate and a conservative MC-style standard error
    p_hat = count_positive / total_used
    stderr = np.sqrt(max(p_hat * (1.0 - p_hat), 1e-16) / total_used)
    return float(p_hat), float(stderr)


def probability_m_is_scw(m: int, **qmc_kwargs) -> Tuple[float, float]:
    """
    Estimate the probability that candidate m is a Super Condorcet Winner (SCW)
    under Impartial Culture in the large-electorate limit (n -> infinity).

    Under the Gaussian limit for Impartial Culture, the SCW event for the
    distinguished candidate corresponds to an orthant probability for a
    centered multivariate normal distribution with covariance matrix ``H``.
    Here, ``H`` is the (d x d) matrix indexed by non-empty subsets of
    {1, ..., m-1}, with d = 2^(m-1) - 1, constructed by :func:`compute_H`.
    The probability is computed via scrambled Sobol quasi–Monte Carlo using
    :func:`_orthant_prob_qmc`.

    Parameters
    ----------
    m : int
        Number of candidates (>= 2). The SCW probability is for the last
        candidate (labelled ``m``), by symmetry.
    **qmc_kwargs :
        Keyword arguments forwarded to :func:`_orthant_prob_qmc`, e.g.:
        - n_samples : int, number of QMC samples (effective)
        - seed : int or None, Sobol engine seed
        - batch : int, batch size for memory efficiency

    Returns
    -------
    p_hat : float
        QMC estimate of the SCW probability for candidate m (in [0, 1]).
    stderr : float
        Conservative MC-style standard error (non-negative).

    Raises
    ------
    ValueError
        If ``m < 2`` or if the construction of ``H`` fails due to size limits.
    np.linalg.LinAlgError
        If ``H`` is not symmetric positive definite (unexpected; indicates
        numerical issues).

    Notes
    -----
    The dimension grows as d = 2^(m-1) - 1, so the method is practical only
    for small/moderate m. The QMC estimator typically exhibits lower variance
    than plain Monte Carlo for the same sample size.

    Examples
    --------
    A minimal smoke test (probability is within [0,1] and stderr >= 0):

    >>> p, se = probability_m_is_scw(3, n_samples=8192, seed=123, batch=2048)
    >>> 0.0 <= p <= 1.0
    True
    >>> se >= 0.0
    True
    """
    # Build covariance matrix H for the Gaussian limit and compute orthant prob.
    H, masks, to_set = compute_H(m)
    p_hat, stderr_p = _orthant_prob_qmc(H, **qmc_kwargs)
    return p_hat, stderr_p


def probability_exists_scw(m: int, **qmc_kwargs) -> Tuple[float, float]:
    """
    Estimate the probability that *some* candidate is a Super Condorcet Winner (SCW)
    under Impartial Culture in the large-electorate limit (n -> infinity).

    Because an SCW (if it exists) is unique, the events
    {candidate i is SCW} for i=1,...,m are disjoint and symmetric. Therefore,
    the existence probability is **exactly**:
        P(exists SCW) = m * P(candidate m is SCW).
    We estimate P(candidate m is SCW) via :func:`probability_m_is_scw` using
    scrambled Sobol QMC, and scale the standard error accordingly.

    Parameters
    ----------
    m : int
        Number of candidates (>= 2).
    **qmc_kwargs :
        Keyword arguments forwarded to :func:`probability_m_is_scw`, e.g.:
        - n_samples : int, number of QMC samples (effective)
        - seed : int or None, Sobol engine seed
        - batch : int, batch size for memory efficiency

    Returns
    -------
    p_exists : float
        QMC estimate of the probability that there exists at least one SCW among
        the m candidates. Mathematically equals m * p, where p is the single-candidate
        SCW probability; in finite-sample estimation this value may exceed 1 slightly
        due to Monte Carlo noise.
    stderr_exists : float
        Conservative MC-style standard error, scaled by m.

    Examples
    --------
    >>> p_exists, se_exists = probability_exists_scw(3, n_samples=8192, seed=42, batch=2048)
    >>> 0.0 <= p_exists  # doctest: +ELLIPSIS
    True
    >>> se_exists >= 0.0
    True
    """
    p_hat, stderr_p = probability_m_is_scw(m, **qmc_kwargs)
    return m * p_hat, m * stderr_p


def probability_irv_is_cm(m: int, **qmc_kwargs) -> Tuple[float, float]:
    """
    Estimate the probability that Instant-Runoff Voting (IRV) is susceptible
    to coalitional manipulation under Impartial Culture in the large-electorate
    limit (n -> infinity).

    In this limit of large electorates:
    - IRV is immune to coalitional manipulation if and only if a Super Condorcet
      Winner (SCW) exists.
    - Therefore, the probability that IRV is manipulable equals
          P(IRV is CM) = 1 - P(exists SCW).
    The existence probability is computed exactly as
          P(exists SCW) = m * P(candidate m is SCW),
    and is estimated via scrambled Sobol QMC.

    Parameters
    ----------
    m : int
        Number of candidates (>= 2).
    **qmc_kwargs :
        Keyword arguments forwarded to :func:`probability_exists_scw`, e.g.:
        - n_samples : int, number of QMC samples (effective)
        - seed : int or None, Sobol engine seed
        - batch : int, batch size for memory efficiency

    Returns
    -------
    p_cm : float
        QMC estimate of the probability that IRV is susceptible to coalitional
        manipulation (in [0, 1]).
    stderr_cm : float
        Conservative MC-style standard error, equal to the error from
        :func:`probability_exists_scw`.

    Examples
    --------
    >>> p_cm, se_cm = probability_irv_is_cm(3, n_samples=8192, seed=7, batch=2048)
    >>> 0.0 <= p_cm <= 1.0
    True
    >>> se_cm >= 0.0
    True
    """
    p_hat, stderr_p = probability_exists_scw(m, **qmc_kwargs)
    return 1.0 - p_hat, stderr_p


def compute_irv_cm_over_m(
    m_values: Iterable[int],
    **qmc_kwargs
) -> "OrderedDict[int, Tuple[float, float]]":
    """
    Compute the probability that IRV is coalitionally manipulable for multiple m.

    For each value of `m` in the provided iterable, this function calls
    :func:`probability_irv_is_cm` and stores the pair (estimate, stderr) in an
    ordered dictionary that preserves the iteration order of `m_values`.

    Parameters
    ----------
    m_values : Iterable[int]
        Iterable of numbers of candidates (each >= 2). The iteration order is
        preserved in the returned mapping.
    **qmc_kwargs
        Keyword arguments forwarded to :func:`probability_irv_is_cm`, e.g.:
        - n_samples : int, number of QMC samples (effective)
        - seed : int or None, Sobol engine seed
        - batch : int, batch size for memory efficiency

    Returns
    -------
    results : OrderedDict[int, Tuple[float, float]]
        Ordered mapping from m -> (p_cm, stderr_cm), where p_cm is the QMC
        estimate of P(IRV is CM) and stderr_cm the conservative standard error.

    Raises
    ------
    ValueError
        If any m < 2.

    Examples
    --------
    Minimal smoke test with small sample size (values are just illustrative):

    >>> res = compute_irv_cm_over_m([3, 4], n_samples=2048, seed=123, batch=1024)
    >>> isinstance(res, OrderedDict)
    True
    >>> sorted(res.keys())
    [3, 4]
    >>> all(0.0 <= p <= 1.0 and se >= 0.0 for (p, se) in res.values())
    True
    """
    results: "OrderedDict[int, Tuple[float, float]]" = OrderedDict()
    for m in m_values:
        if m < 2:
            raise ValueError(f"m must be >= 2, got {m}.")
        p_cm, se_cm = probability_irv_is_cm(m, **qmc_kwargs)
        results[m] = (float(p_cm), float(se_cm))
    return results


def latex_table_irv_cm(
    results: Mapping[int, Tuple[float, float]],
    digits: int = 3,
    right_align_m: bool = True,
    caption: str | None = None,
    label: str | None = None
) -> str:
    """
    Render a LaTeX table of IRV coalitional manipulability probabilities.

    The table has two columns:
      - Left: m (number of candidates)
      - Right: estimate ± standard error, formatted to the requested precision.

    Parameters
    ----------
    results : Mapping[int, Tuple[float, float]]
        Mapping from m -> (p_cm, stderr_cm). Typically produced by
        :func:`compute_irv_cm_over_m`.
    digits : int, default=3
        Number of decimal places for both estimate and standard error.
    right_align_m : bool, default=True
        If True, right-align the `m` column (using 'r'); otherwise left-align ('l').
    caption : str or None, default=None
        Optional LaTeX caption text (table will be wrapped in a tabular only if None).
        If provided, the table is wrapped in a LaTeX `table` environment with a caption.
    label : str or None, default=None
        Optional LaTeX label (e.g., 'tab:irv-cm'); used only if `caption` is provided.

    Returns
    -------
    table_tex : str
        A LaTeX string containing the formatted table. Uses `\\pm` for the
        plus/minus symbol and formats numeric values with the specified precision.

    Notes
    -----
    - Values are **not** clipped to [0, 1]; if your estimator plus/minus error
      exceeds bounds, consider adding notes in the paper.
    - Rows are emitted in the iteration order of `results`.

    Examples
    --------
    >>> demo = OrderedDict([(3, (0.12345, 0.00432)), (4, (0.23456, 0.00543))])
    >>> tex = latex_table_irv_cm(demo, digits=3)
    >>> "\\pm" in tex and "0.123" in tex and "0.004" in tex
    True
    >>> "\\begin{tabular}" in tex
    True
    """
    col_m = 'r' if right_align_m else 'l'
    header = (
        "\\begin{tabular}{" + col_m + " r}\n"
        "\\toprule\n"
        "$m$ & $\\Pr[\\text{IRV is CM}]$ \\\\\n"
        "\\midrule\n"
    )

    rows = []
    fmt = f"{{:.{digits}f}}"
    for m, (p, se) in results.items():
        p_str = fmt.format(p)
        se_str = fmt.format(se)
        rows.append(f"{m} & ${p_str} \\pm {se_str}$ \\\\")
    body = "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}"

    tabular = header + body

    if caption is None:
        return tabular

    # Wrap in a floating table environment with caption/label if provided
    cap = f"\\caption{{{caption}}}\n"
    lab = f"\\label{{{label}}}\n" if label else ""
    return "\\begin{table}[t]\n\\centering\n" + tabular + "\n" + cap + lab + "\\end{table}\n"
