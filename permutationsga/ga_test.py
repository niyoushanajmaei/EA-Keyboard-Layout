import numpy as np

from .ga import crossover_cx, crossover_pmx, crossover_ox_neg
from .problem import Solution

def test_crossover_cx_basic():
    l = 10
    x = Solution(np.arange(l))
    y = Solution(np.arange(l)[::-1])
    indices = [0, 1, 2]
    # not_indices = list(set(np.arange(l)) - set(indices))
    
    r = crossover_cx(indices, x, y)

    assert len(r) == 2, "CX should return two solutions"
    assert len(r[0].e) == l, "CX should return solutions of length l"
    assert len(r[1].e) == l, "CX should return solutions of length l"
    assert set(r[0].e) == set(np.arange(l)), "CX should return solutions in which all elements 0-(l-1) occur once"
    assert set(r[1].e) == set(np.arange(l)), "CX should return solutions in which all elements 0-(l-1) occur once"

def test_crossover_cx_2():
    l = 10
    xs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    x = Solution(xs)
    ys = np.array([2, 1, 0, 5, 4, 3, 8, 7, 6, 9])
    print(ys)
    y = Solution(ys)
    indices = [0]
    # not_indices = list(set(np.arange(l)) - set(indices))
    
    r = crossover_cx(indices, x, y)

    np.testing.assert_array_equal(r[0].e, np.array([2, 1, 0, 3, 4, 5, 6, 7, 8, 9]))
    np.testing.assert_array_equal(r[1].e, np.array([0, 1, 2, 5, 4, 3, 8, 7, 6, 9]))

def test_crossover_cx_3():
    l = 10
    xs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    x = Solution(xs)
    ys = np.array([2, 1, 0, 5, 4, 3, 8, 7, 6, 9])
    print(ys)
    y = Solution(ys)
    indices = [0, 6]
    # not_indices = list(set(np.arange(l)) - set(indices))
    
    r = crossover_cx(indices, x, y)

    np.testing.assert_array_equal(r[0].e, np.array([2, 1, 0, 3, 4, 5, 8, 7, 6, 9]))
    np.testing.assert_array_equal(r[1].e, np.array([0, 1, 2, 5, 4, 3, 6, 7, 8, 9]))


def test_crossover_pmx_basic():
    l = 10
    x = Solution(np.arange(l))
    y = Solution(np.arange(l)[::-1])
    indices = [0, 1, 2]
    # not_indices = list(set(np.arange(l)) - set(indices))
    
    r = crossover_pmx(indices, x, y)

    assert len(r) == 2, "PMX should return two solutions"
    assert len(r[0].e) == l, "PMX should return solutions of length l"
    assert len(r[1].e) == l, "PMX should return solutions of length l"
    assert set(r[0].e) == set(np.arange(l)), "PMX should return solutions in which all elements 0-(l-1) occur once"
    assert set(r[1].e) == set(np.arange(l)), "PMX should return solutions in which all elements 0-(l-1) occur once"

def test_crossover_pmx_2():
    l = 10
    xs = np.array([0, 1, 3, 2, 4, 5, 6, 7, 8, 9])
    xs_re = np.array([2, 1, 3, 0, 4, 5, 6, 7, 8, 9])
    
    ys = np.array([2, 1, 0, 5, 4, 3, 8, 7, 6, 9])
    ys_re = np.array([0, 1, 2, 5, 4, 3, 8, 7, 6, 9])
    
    x = Solution(xs)
    y = Solution(ys)
    indices = [0]
    # not_indices = list(set(np.arange(l)) - set(indices))
    
    r = crossover_pmx(indices, x, y)

    np.testing.assert_array_equal(r[0].e, xs_re)
    np.testing.assert_array_equal(r[1].e, ys_re)

def test_crossover_ox_basic():
    l = 10
    x = Solution(np.arange(l))
    y = Solution(np.arange(l)[::-1])
    indices = [0, 1, 2]
    not_indices = list(set(np.arange(l)) - set(indices))
    
    r = crossover_ox_neg(not_indices, x, y)

    assert len(r) == 2, "OX should return two solutions"
    assert len(r[0].e) == l, "OX should return solutions of length l"
    assert len(r[1].e) == l, "OX should return solutions of length l"
    assert set(r[0].e) == set(np.arange(l)), "OX should return solutions in which all elements 0-(l-1) occur once"
    assert set(r[1].e) == set(np.arange(l)), "OX should return solutions in which all elements 0-(l-1) occur once"

def test_crossover_ox_2():
    l = 10
    xs = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(xs)
    x = Solution(xs)
    ys = np.array([2, 1, 0, 5, 4, 3, 8, 7, 6, 9])
    print(ys)
    y = Solution(ys)
    indices = [0]
    not_indices = list(set(np.arange(l)) - set(indices))
    
    r = crossover_ox_neg(not_indices, x, y)

    print(r[0].e)
    print(r[1].e)

    np.testing.assert_array_equal(r[0].e, np.array([2, 0, 1, 3, 4, 5, 6, 7, 8, 9]))
    np.testing.assert_array_equal(r[1].e, np.array([0, 2, 1, 5, 4, 3, 8, 7, 6, 9]))
