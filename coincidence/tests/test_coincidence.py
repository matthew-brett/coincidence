""" Tests for cooincidnce functions
"""

import numpy as np

from coincidence import link_table, jaccard_table, calc_ratios


def equal_to(col):
    # Slow version for link_table calculation
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
    # Slow version for jaccard_table calculation
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
