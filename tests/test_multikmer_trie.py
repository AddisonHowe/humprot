"""Tests for tree classes.

"""

import pytest

import os

from humprot.trees.multikmer_tree import MultiKmerTree
from humprot.helpers import get_kmers_from_sequences

DATDIR = "./tests/data/_tmp"
os.makedirs(DATDIR, exist_ok=True)

# def get_tree(name="tmp_test_tree.dat"):
#     return CompactTrie(f"{DATDIR}/{name}", mode="w+", initial_nodes=2)

###################
##  Begin Tests  ##
###################


@pytest.fixture
def multikmer_tree1(initial_nodes):
    seqs = [
        [0, 1, 1, 2],
        [0, 1, 1],
        [0, 1, 2],
        [1, 0, 0],
    ]
    tree = MultiKmerTree(alphabet_size=3, initial_nodes=initial_nodes)
    for k in range(1, 4):
        kmers = get_kmers_from_sequences(seqs, k)
        for kmer in kmers:
            tree.add_kmer(kmer, value=1)
    return tree


@pytest.mark.parametrize("initial_nodes", [1, 2, 10, 20])
@pytest.mark.parametrize("query, value", [
    [[0],           5],
    [[1],           6],
    [[2],           2],
    [[0, 0],        1],
    [[0, 1],        3],
    [[0, 2],        0],
    [[1, 0],        1],
    [[1, 1],        2],
    [[1, 2],        2],
    [[2, 0],        0],
    [[2, 1],        0],
    [[2, 2],        0],
    [[0, 1, 1],     2],
    [[1, 1, 2],     1],
    [[0, 1, 2],     1],
    [[1, 0, 0],     1],
    [[0, 2, 1],     0],
    [[2, 0, 0],     0],
    [[2, 0, 1],     0],
    [[2, 0, 2],     0],
])
def test_multikmer_tree1(initial_nodes, multikmer_tree1, query, value):
        tree = multikmer_tree1
        # print(tree.nodes)
        res = tree.query(query)
        assert res == value, f"Expected {value} for query {query}. Got {res}."


@pytest.mark.parametrize("initial_nodes", [1, 2, 10, 20])
@pytest.mark.parametrize("query, mask, value", [
    [[0],           3,   5],
    [[1],           3,   6],
    [[2],           3,   2],
    [[0, 0],        3,   1],
    [[0, 1],        3,   3],
    [[0, 2],        3,   0],
    [[1, 0],        3,   1],
    [[1, 1],        3,   2],
    [[1, 2],        3,   2],
    [[2, 2],        3,   0],
    [[0, 1, 1],     3,   2],
    [[1, 1, 2],     3,   1],
    [[0, 1, 2],     3,   1],
    [[0, 2, 1],     3,   0],
    [[2, 0, 0],     3,   0],
    [[3, 3, 2],     3,   2],
    [[3, 3, 3],     3,   5],
    [[3, 3],        3,   9],
    [[3],           3,   13],
    [[3, 0],        3,   2],
    [[3, 1, 3],     3,   4],
])
def test_query_masked_multikmer_tree1(
        initial_nodes, multikmer_tree1, query, mask, value
):
        tree = multikmer_tree1
        res = tree.query_masked(query, mask)
        assert res == value, f"Expected {value} for query {query}. Got {res}."
