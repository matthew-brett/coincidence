""" Series ids and Jaccard coefficients
"""

import numpy as np


def link_table(series_ids):
    """ Make shape (n, n) array `arr` of link dummy indicators

    `series_ids` contains values that are unique for a particular linked
    series.  Therefore, two rows with the same value in `series_ids` are
    linked.

    A 1 in ``arr[i, j]`` means that case ``i`` has a link to case ``j``.
    """
    series_ids = np.array(series_ids, dtype=float)
    n = len(series_ids)
    arr = np.zeros((n, n))
    col, row = series_ids[:, None], series_ids[None, :]
    arr[:, :] = col == row
    arr[np.isnan(col + row)] = np.nan
    return arr


def jaccard_table(col):
    """ Make shape (n, n) array `arr` of 1, 1 coincidences in `col`.

    A 1 in ``arr[i, j]`` means that both case ``i`` and ``j`` have a 1 in the
    corresponding position in `col`.

    See :func:`jaccard_table_slow` for a version that might be easier to read.
    """
    col = np.array(col, dtype=float)[:, None]  # Row
    res = col * col.T
    # If either value is 0, 0 overrides nan
    is_zero = col == 0
    res[is_zero | is_zero.T] = 0
    return res


def equal_to(col):
    # Slow calculation for illustration.
    col = np.array(col, dtype=float)
    n = len(col)
    arr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a, b = col[i], col[j]
            if np.isnan(a) or np.isnan(b):
                arr[i, j] = np.nan
            else:
                arr[i, j] = a == b
    return arr


def both_1(col):
    # Slow calculation for illustration.
    col = np.array(col, dtype=float)
    n = len(col)
    arr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a, b = col[i], col[j]
            if 0 in (a, b):
                arr[i, j] = 0
            elif np.isnan(a + b):
                arr[i, j] = np.nan
            else:
                arr[i, j] = 1
    return arr


def calc_ratios(link_pairs, jacc_pairs):
    not_nan = ~np.isnan(link_pairs + jacc_pairs)
    link_pairs, jacc_pairs = link_pairs[not_nan], jacc_pairs[not_nan]
    n_links = np.sum(link_pairs)
    n_jaccs = np.sum(jacc_pairs)
    # Multiply the links and Jaccards, and sum them to get the number
    # of links that are also Jaccards.
    n_jl = np.sum(link_pairs * jacc_pairs)
    # Linked Jaccards divided by all links.
    jl_rat = n_jl / n_links
    # Non-linked Jaccards divided by all non-links.
    jnl_rat = (n_jaccs - n_jl) / (len(link_pairs) - n_links)
    return jl_rat, jnl_rat


def test_jaccard():
    # Case links
    in_col = [0, 1, 1, 2, 3, 3, 3, 4]
    # Presence of feature
    feature = [0, 0, 1, 0, 1, 1, 1, 0]
    link_arr = link_table(in_col)
    assert np.all(link_arr == equal_to(in_col))
    # Extract lower triangle, and unravel to 1D.
    inds = np.tril_indices(8, -1)
    link_pairs = link_arr[inds]
    assert np.all(link_pairs ==
                  [0,
                   0, 1,
                   0, 0, 0,
                   0, 0, 0, 0,
                   0, 0, 0, 0, 1,
                   0, 0, 0, 0, 1, 1,
                   0, 0, 0, 0, 0, 0, 0])
    jacc_pairs = jaccard_table(feature)[inds]
    assert np.all(jacc_pairs ==
                  [0,
                   0, 0,
                   0, 0, 0,
                   0, 0, 1, 0,
                   0, 0, 1, 0, 1,
                   0, 0, 1, 0, 1, 1,
                   0, 0, 0, 0, 0, 0, 0])
    # Check slow version gives same answer.
    assert np.all(jacc_pairs == both_1(feature)[inds])
    jl_r, jnl_r = calc_ratios(link_pairs, jacc_pairs)
    assert np.isclose(jl_r, 3 / 4)
    assert np.isclose(jnl_r, 3 / 24)


def nan_eq(a, b):
    return np.array_equal(a, b, equal_nan=True)


def test_jaccard_nan():
    # Case links
    in_col = [0, 1, 1, 2, 3, np.nan, 3, 4]
    # Presence of feature
    feature = [0, np.nan, 1, 0, 1, np.nan, 1, 0]
    link_arr = link_table(in_col)
    assert nan_eq(link_arr, equal_to(in_col))
    # Extract lower triangle, and unravel to 1D.
    inds = np.tril_indices(8, -1)
    link_pairs = link_arr[inds]
    assert nan_eq(link_pairs,
                  [0,
                   0, 1,
                   0, 0, 0,
                   0, 0, 0, 0,
                   np.nan, np.nan, np.nan, np.nan, np.nan,
                   0, 0, 0, 0, 1, np.nan,
                   0, 0, 0, 0, 0, np.nan, 0])
    jacc_pairs = jaccard_table(feature)[inds]
    # Check slow version gives same answer.
    assert nan_eq(jacc_pairs, both_1(feature)[inds])
    assert nan_eq(jacc_pairs,
                  [0,
                   0, np.nan,
                   0, 0, 0,
                   0, np.nan, 1, 0,
                   0, np.nan, np.nan, 0, np.nan,
                   0, np.nan, 1, 0, 1, np.nan,
                   0, 0, 0, 0, 0, 0, 0])
    jl_r, jnl_r = calc_ratios(link_pairs, jacc_pairs)
    assert np.isclose(jl_r, 1)
    assert np.isclose(jnl_r, 2 / 17)


def test_jaccard2():
    # Case links (case IDs)
    in_col = [0, 0, 0, 1, 2]
    # Presence of feature
    feature = [1, 1, 0, 0, 1]
    link_arr = link_table(in_col)
    assert np.all(link_arr == equal_to(in_col))
    # Extract lower triangle, and unravel to 1D.
    inds = np.tril_indices(5, -1)
    link_pairs = link_arr[inds]
    assert np.all(link_pairs ==
                  [1,
                   1, 1,
                   0, 0, 0,
                   0, 0, 0, 0])
    jacc_pairs = jaccard_table(feature)[inds]
    assert np.all(jacc_pairs ==
                  [1,
                   0, 0,
                   0, 0, 0,
                   1, 1, 0, 0])
    # Check slow version gives same answer.
    assert np.all(jacc_pairs == both_1(feature)[inds])
    jl_r, jnl_r = calc_ratios(link_pairs, jacc_pairs)
    assert np.isclose(jl_r, 1 / 3)
    assert np.isclose(jnl_r, 2 / 7)
