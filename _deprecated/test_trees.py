"""Tests for tree classes.

"""

import pytest
import numpy as np

from humprot.trees import KmerTree

###################
##  Begin Tests  ##
###################


@pytest.mark.parametrize('kmers, k, keys_exp, values_exp', [
    [
        [[0, 1, 2], [0, 1, 3], [1, 2, 3], [1, 1, 3]], 3,
        [np.nan, 0, 1, 1, 1, 2, 2, 3, 3, 3],
        [4, 2, 2, 2, 1, 1, 1, 1, 1, 1],
    ],
])
class TestKmerTreeConstruction:
    
    def test_keys(self, kmers, k, keys_exp, values_exp):
        tree = KmerTree()
        for kmer in kmers:
            tree.add_kmer(kmer)
        
        f = lambda n: n.get_key()
        keys = tree.map_bfs(f)
        keys = np.array([k if k is not None else np.nan for k in keys])
        errors = []
        if not np.allclose(keys, keys_exp, equal_nan=True):
            msg = f"Wrong keys. Expected {keys_exp}. Got {keys}."
            errors.append(msg)
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

    def test_values(self, kmers, k, keys_exp, values_exp):
        tree = KmerTree()
        for kmer in kmers:
            tree.add_kmer(kmer)
        
        count_func = lambda n: n.get_value()
        values = tree.map_bfs(count_func)
        errors = []
        if not np.allclose(values, values_exp):
            msg = f"Wrong values. Expected {values_exp}. Got {values}."
            errors.append(msg)
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

    

@pytest.fixture
def simple_tree1():
    tree = KmerTree()
    tree.add_kmer([0, 1, 2])
    tree.add_kmer([0, 1, 3])
    tree.add_kmer([1, 2, 3])
    tree.add_kmer([1, 1, 3])
    return tree


@pytest.mark.parametrize("query, value", [
    [[0, 1, 2],     1],
    [[0, 1],        2],
    [[1, 2, 3, 4],  0],
    [[0, 2, 2],     0],
    [[0],           2],
    [[2],           0],
])
def test_query_simple_tree1(simple_tree1, query, value):
    res = simple_tree1.query(query)
    assert  res == value, f"Expected {value} for query {query}. Got {res}."


@pytest.mark.parametrize("query, value", [
    [[0, -1, 1], 0],
    [[0, -1, 2], 1],
    [[0, 1, -1], 2],
    [[0, -1, -1], 2],
    [[-1, -1, -1], 4],
    [[0, -1, -1, -1], 0],
    [[-1, -1, -1, 0], 0],
    [[-1, 1, 3], 2],
])
def test_masked_query_simple_tree1(simple_tree1, query, value):
    mask = 10
    query = [i if i != -1 else mask for i in query]
    res = simple_tree1.query_masked(query, mask)
    assert  res == value, f"Expected {value} for query {query}. Got {res}."
