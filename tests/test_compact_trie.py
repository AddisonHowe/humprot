"""Tests for tree classes.

"""

import os
import pytest
import numpy as np

from humprot.compact_trie import CompactTrie
from humprot.helpers import get_kmers_from_sequences, int2sym, sym2int

DATDIR = "./tests/data/_tmp"
os.makedirs("DATDIR", exist_ok=True)

def get_tree(name="tmp_test_tree.dat"):
    return CompactTrie(f"{DATDIR}/{name}", mode="w+", initial_nodes=2)

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
        tree = get_tree("test_keys.dat")
        for kmer in kmers:
            tree.add_kmer(kmer)
        
        # f = lambda n: n.get_key()
        keys = tree.map_bfs(tree.get_key)
        keys = np.array([k if k is not None else np.nan for k in keys])
        errors = []
        if not np.allclose(keys, keys_exp, equal_nan=True):
            msg = f"Wrong keys. Expected {keys_exp}. Got {keys}."
            errors.append(msg)
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

    def test_values(self, kmers, k, keys_exp, values_exp):
        tree = get_tree("test_keys.dat")
        for kmer in kmers:
            tree.add_kmer(kmer)
        
        values = tree.map_bfs(tree.get_value)
        errors = []
        if not np.allclose(values, values_exp):
            msg = f"Wrong values. Expected {values_exp}. Got {values}."
            errors.append(msg)
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

    

@pytest.fixture
def simple_tree1():
    tree = get_tree("test_simpletree1.dat")
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


# @pytest.mark.skip()
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


def test_add_many():
    tree = get_tree("test_add_many.dat")
    for i in range(5):
        for j in range(5):
            for k in range(5):
                for m in range(5):
                    tree.add_kmer((i, j, k, m))


@pytest.fixture()
def large_tree1(k):
    aa_list = list("ABCD")
    map_sym2int = {sym: i for i, sym in enumerate(aa_list)}
    sequence_strs = [
        "AABBCCDD",
        "AAAABBBB",
    ]
    seqs = [sym2int(seq, map_sym2int) for seq in sequence_strs]
    kmers = get_kmers_from_sequences(seqs, k)
    tree = get_tree("test_largetree1.dat")
    for kmer in kmers:
        tree.add_kmer(kmer)
    return tree, seqs, kmers


@pytest.mark.parametrize("k, size_exp, depth_exp, n_uniq_kmers_exp", [
    [4, 10, 4, 9,],
    [5, 8,  5, 8,],
    [6, 6,  6, 6,],
    [7, 4,  7, 4,],
    [8, 2,  8, 2,],
])
def test_large_tree(large_tree1, k, size_exp, depth_exp, n_uniq_kmers_exp):
    tree, seqs, kmers = large_tree1
    assert tree.size == size_exp
    assert tree.depth == depth_exp
    # assert tree.num_uniq == n_uniq_kmers_exp


@pytest.mark.parametrize("k, query, depth, count_exp", [
    [4, [0,0,1,1], None, 2], 
    [4, [0,1,1,2], None, 1], 
    [4, [1,1,1,0], None, 0], 
    [4, [0,0,0], None, 3],
    [8, [0,0,2], None, 0],
    [4, [3,3], None, 1],
])
def test_deep_query(large_tree1, k, query, depth, count_exp):
    tree, seqs, kmers = large_tree1
    count = tree.deep_query(query, depth)
    assert count == count_exp


@pytest.mark.parametrize("query, value", [
    [[0, 1, 2],     1],
    [[0, 1],        2],
    [[1, 2, 3, 4],  0],
    [[0, 2, 2],     0],
    [[0],           2],
    [[2],           0],
])
def test_save_load_simple_tree1(simple_tree1, query, value):
    savefpath = "./tests/data/_tmp/saved_simple_tree1.npz"
    simple_tree1.save(savefpath)
    loaded_tree = CompactTrie.load(savefpath)
    res = loaded_tree.query(query)
    assert  res == value, f"Expected {value} for query {query}. Got {res}."
