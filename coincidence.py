""" Jaccard coefficients
"""

import operator

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
    for i, series in enumerate(series_ids):
        if np.isnan(series):
            continue
        arr[:, i] = series_ids == series
    return arr


def jaccard_table(col):
    """ Make shape (n, n) array `arr` of 1, 1 coincidences in `col`.

    A 1 in ``arr[i, j]`` means that both case ``i`` and ``j`` have a 1 in the
    corresponding position in `col`.

    See :func:`jaccard_table_slow` for a version that might be easier to read.
    """
    col = np.array(col, dtype=float)
    return np.outer(col, col)


def jaccard_table_slow(col):
    # Slower calculation, to make more obvious what it's doing.
    col = np.array(col, dtype=float)
    n = len(col)
    # Replicate the column n times.
    arr = np.repeat(col[:, None], n, axis=1)
    # Set all columns to 0 with corresponding 0 in input column.
    arr[:, col == 0] = 0
    return arr


def funcand(a, b):
    return a and b


def sameas(col, op=funcand):
    # Another slow calculation for illustration.
    col = np.array(col, dtype=float)
    n = len(col)
    arr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if op(col[i], col[j]):
                arr[i, j] = 1
    return arr


def calc_ratios(link_pairs, jacc_pairs):
    n_links = np.sum(link_pairs)
    n_jaccs = np.sum(jacc_pairs)
    n_pairs = len(link_pairs)
    # Multiply the links and Jaccards, and sum them to get the number
    # of links that are also Jaccards.
    # Equivalent to np.sum(link_pairs * jacc_pairs)
    n_jl = np.dot(link_pairs, jacc_pairs)
    # Linked Jaccards divided by all links.
    jl_rat = n_jl / n_links
    # Non-linked Jaccards divided by all non-links.
    jnl_rat = (n_jaccs - n_jl) / (n_pairs - n_links)
    return jl_rat, jnl_rat


def test_jaccard():
    # Case links
    in_col = [0, 1, 1, 2, 3, 3, 3, 4]
    # Presence of feature
    feature = [0, 0, 1, 0, 1, 1, 1, 0]
    link_arr = link_table(in_col)
    assert np.all(link_arr == sameas(in_col, operator.eq))
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
    assert np.all(jacc_pairs == jaccard_table_slow(feature)[inds])
    assert np.all(jacc_pairs == sameas(feature)[inds])
    jl_r, jnl_r = calc_ratios(link_pairs, jacc_pairs)
    assert np.isclose(jl_r, 3 / 4)
    assert np.isclose(jnl_r, 3 / 24)


def test_jaccard_nan():
    # Case links
    in_col = [0, 1, 1, 2, 3, np.nan, 3, 4]
    # Presence of feature
    feature = [0, np.nan, 1, 0, 1, np.nan, 1, 0]
    link_arr = link_table(in_col)
    assert np.all(link_arr == sameas(in_col, operator.eq))
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
    assert np.all(jacc_pairs == jaccard_table_slow(feature)[inds])
    assert np.all(jacc_pairs == sameas(feature)[inds])
    jl_r, jnl_r = calc_ratios(link_pairs, jacc_pairs)
    assert np.isclose(jl_r, 3 / 4)
    assert np.isclose(jnl_r, 3 / 24)

def test_jaccard2():
    # Case links (case IDs)
    in_col = [0, 0, 0, 1, 2]
    # Presence of feature
    feature = [1, 1, 0, 0, 1]
    link_arr = link_table(in_col)
    assert np.all(link_arr == sameas(in_col, operator.eq))
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
    assert np.all(jacc_pairs == jaccard_table_slow(feature)[inds])
    assert np.all(jacc_pairs == sameas(feature)[inds])
    jl_r, jnl_r = calc_ratios(link_pairs, jacc_pairs)
    assert np.isclose(jl_r, 1 / 3)
    assert np.isclose(jnl_r, 2 / 7)


# Make sure test passes first.
test_jaccard()
test_jaccard2()
