"""Tests for tree classes.

"""

import pytest
from contextlib import nullcontext as does_not_raise

import os
import numpy as np

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
def multikmer_tree1(
        memmap, filename, mode, initial_nodes
) -> tuple[MultiKmerTree, dict]:
    seqs = [
        [0, 1, 1, 2],
        [0, 1, 1],
        [0, 1, 2],
        [1, 0, 0],
    ]
    tree = MultiKmerTree(
        alphabet_size=3, initial_nodes=initial_nodes,
        filename=filename, mode=mode,
    )
    for k in range(1, 5):
        kmers = get_kmers_from_sequences(seqs, k)
        for kmer in kmers:
            tree.add_kmer(kmer, value=1)
    info = {
        "memmapped": memmap,
        "depth_exp": 4,
        "length_exp": 13,
        "layer_size_exp": [1, 3, 5, 4, 1],
        "layer_value_sum_exp": [0, 13, 9, 5, 1],
    }
    return tree, info


@pytest.mark.parametrize("memmap, filename, mode", [
     [False, None, None],
     [True, f"{DATDIR}/tmp_mktree1w.npz", "w+"],
     [True, f"{DATDIR}/tmp_mktree1r.npz", "r+"],
     [True, f"{DATDIR}/tmp_mktree1c.npz", "c"],
])
@pytest.mark.parametrize("initial_nodes", [1, 2, 10, 20])
@pytest.mark.parametrize("do_trim", [True, False])
class TestAttributesMultikmerTree1:
    
    def test_depth(
            self, memmap, filename, mode, initial_nodes, do_trim, 
            multikmer_tree1
    ):
        tree, info = multikmer_tree1
        if do_trim:
            tree.trim_free_nodes()
        depth = tree.get_depth()
        depth_exp = info["depth_exp"]
        msg = f"Wrong depth. Expected {depth_exp}. Got {depth}."
        assert depth == depth_exp, msg

    def test_length(
            self, memmap, filename, mode, initial_nodes, do_trim, 
            multikmer_tree1
    ):
        tree, info = multikmer_tree1
        if do_trim:
            tree.trim_free_nodes()
        length = len(tree)
        length_exp = info["length_exp"]
        msg = f"Wrong length. Expected {length_exp}. Got {length}."
        assert length == length_exp, msg

    @pytest.mark.parametrize("depth", [0, 1, 2, 3, 4])
    def test_get_layer_size(
            self, memmap, filename, mode, initial_nodes, do_trim, 
            multikmer_tree1, depth,
    ):
        tree, info = multikmer_tree1
        if do_trim:
            tree.trim_free_nodes()
        layer_size = tree.get_layer_size(depth)
        layer_size_exp = info["layer_size_exp"][depth]
        msg = f"Wrong layer_size. Expected {layer_size_exp}. Got {layer_size}."
        assert layer_size == layer_size_exp, msg

    @pytest.mark.parametrize("depth", [0, 1, 2, 3, 4])
    def test_get_layer_value_sum(
            self, memmap, filename, mode, initial_nodes, do_trim, 
            multikmer_tree1, depth,
    ):
        tree, info = multikmer_tree1
        if do_trim:
            tree.trim_free_nodes()
        layer_value_sum = tree.get_layer_value_sum(depth)
        layer_value_sum_exp = info["layer_value_sum_exp"][depth]
        msg = f"Wrong layer_value_sum. Expected {layer_value_sum_exp}. "
        msg += f"Got {layer_value_sum}."
        assert layer_value_sum == layer_value_sum_exp, msg


@pytest.mark.parametrize("memmap, filename, mode", [
     [False, None, None],
     [True, f"{DATDIR}/tmp_mktree1w.npz", "w+"],
     [True, f"{DATDIR}/tmp_mktree1r.npz", "r+"],
     [True, f"{DATDIR}/tmp_mktree1c.npz", "c"],
])
@pytest.mark.parametrize("initial_nodes", [1, 2, 10, 20])
@pytest.mark.parametrize("do_trim", [True, False])
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
def test_query_multikmer_tree1(
        initial_nodes, memmap, filename, mode, do_trim, 
        multikmer_tree1, query, value
):
    tree, info = multikmer_tree1
    if do_trim:
        tree.trim_free_nodes()
    res = tree.query(query)
    assert res == value, f"Expected {value} for query {query}. Got {res}."


@pytest.mark.parametrize("memmap, filename, mode", [
     [False, None, None],
     [True, f"{DATDIR}/tmp_mktree1w.npz", "w+"],
     [True, f"{DATDIR}/tmp_mktree1r.npz", "r+"],
     [True, f"{DATDIR}/tmp_mktree1c.npz", "c"],
])
@pytest.mark.parametrize("initial_nodes", [1, 2, 10, 20])
@pytest.mark.parametrize("do_trim", [True, False])
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
        initial_nodes, memmap, filename, mode, do_trim, 
        multikmer_tree1, query, mask, value
):
    tree, info = multikmer_tree1
    if do_trim:
        tree.trim_free_nodes()
    res = tree.query_masked(query, mask)
    assert res == value, f"Expected {value} for query {query}. Got {res}."


@pytest.mark.parametrize("memmap, filename, mode", [
     [False, None, None],
     [True, f"{DATDIR}/tmp_mktree1w.npz", "w+"],
     [True, f"{DATDIR}/tmp_mktree1r.npz", "r+"],
     [True, f"{DATDIR}/tmp_mktree1c.npz", "c"],
])
@pytest.mark.parametrize("initial_nodes", [1, 2, 10, 20])
@pytest.mark.parametrize("do_trim", [True, False])
class TestSaveLoadMultiKmerTree1:

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
    def test_save_load_mktree1(
            initial_nodes, memmap, filename, mode, do_trim,
            multikmer_tree1, query, value
    ):
        tree, info = multikmer_tree1
        if do_trim:
            tree.trim_free_nodes()
        savefpath = f"{DATDIR}/saved_mktree1.npz"
        tree.save(savefpath)
        loaded_tree = MultiKmerTree.load(savefpath)
        res = loaded_tree.query(query)
        assert  res == value, f"Expected {value} for query {query}. Got {res}."

    def test_depth(
            self, memmap, filename, mode, initial_nodes, do_trim, 
            multikmer_tree1
    ):
        tree, info = multikmer_tree1
        if do_trim:
            tree.trim_free_nodes()
        savefpath = f"{DATDIR}/saved_mktree1.npz"
        tree.save(savefpath)
        tree = MultiKmerTree.load(savefpath)
        depth = tree.get_depth()
        depth_exp = info["depth_exp"]
        msg = f"Wrong depth. Expected {depth_exp}. Got {depth}."
        assert depth == depth_exp, msg

    def test_length(
            self, memmap, filename, mode, initial_nodes, do_trim, 
            multikmer_tree1
    ):
        tree, info = multikmer_tree1
        if do_trim:
            tree.trim_free_nodes()
        savefpath = f"{DATDIR}/saved_mktree1.npz"
        tree.save(savefpath)
        tree = MultiKmerTree.load(savefpath)
        length = len(tree)
        length_exp = info["length_exp"]
        msg = f"Wrong length. Expected {length_exp}. Got {length}."
        assert length == length_exp, msg


@pytest.mark.parametrize("filename, mode", [
     [None, None],
     [f"{DATDIR}/tmp_mktmptree1w.npz", "w+"],
     [f"{DATDIR}/tmp_mktmptree1r.npz", "r+"],
     [f"{DATDIR}/tmp_mktmptree1c.npz", "c"],
])
@pytest.mark.parametrize("initial_nodes", [1, 2, 10, 20])
@pytest.mark.parametrize("kmers2add, expect_context", [
    [
        [[0], [1], [0,1], [0,1,2]], 
        does_not_raise()
    ],
    [
        [[0,1]], 
        pytest.raises(
            RuntimeError, 
            match="Cannot add 2-mer to tree with current depth 0"
        )
    ]
    ,[
        [[0,1], [0,1], [0,1,2], [0,1,2]], 
        pytest.raises(
            RuntimeError, 
            match="Cannot add 2-mer to tree with current depth 0"
        )
    ],[
        [[0], [1], [0,1,2], [0,1]], 
        pytest.raises(
            RuntimeError, 
            match="Cannot add 3-mer to tree with current depth 1"
        )
    ],[
        [[0], [1], [0,1,2], [0]], 
        pytest.raises(
            RuntimeError, 
            match="Cannot add 3-mer to tree with current depth 1"
        )
    ],
])
def test_bad_kmer_additions(
        filename, mode, initial_nodes, kmers2add, expect_context,
):
    tree = MultiKmerTree(
        alphabet_size=10, filename=filename, mode=mode,
        initial_nodes=initial_nodes,
    )
    with expect_context:
        for kmer in kmers2add:
            tree.add_kmer(kmer)
    return


@pytest.mark.parametrize("memmap, filename, mode", [
     [False, None, None],
     [True, f"{DATDIR}/tmp_mktree1w.npz", "w+"],
     [True, f"{DATDIR}/tmp_mktree1r.npz", "r+"],
     [True, f"{DATDIR}/tmp_mktree1c.npz", "c"],
])
@pytest.mark.parametrize("initial_nodes", [1, 2, 10, 20])
@pytest.mark.parametrize("do_trim", [True, False])
@pytest.mark.parametrize("max_depth, vals_exp", [
    [1, [0,1,2]],
    [2, [0,1,2,0,1,0,1,2]],
    [3, [0,1,2,0,1,0,1,2,1,2,0,2]],
    [4, [0,1,2,0,1,0,1,2,1,2,0,2,2]],
    [5, [0,1,2,0,1,0,1,2,1,2,0,2,2]],
])
def test_iternodekeys_multikmer_tree1(
        initial_nodes, memmap, filename, mode, do_trim, 
        multikmer_tree1, max_depth, vals_exp
):
    tree, info = multikmer_tree1
    if do_trim:
        tree.trim_free_nodes()
    res = [x for x in tree.iternodekeys(max_depth=max_depth)]
    assert np.all(np.equal(res, vals_exp)), \
        f"Expected {vals_exp}. Got {res}."


@pytest.mark.parametrize("memmap, filename, mode", [
     [False, None, None],
     [True, f"{DATDIR}/tmp_mktree1w.npz", "w+"],
     [True, f"{DATDIR}/tmp_mktree1r.npz", "r+"],
     [True, f"{DATDIR}/tmp_mktree1c.npz", "c"],
])
@pytest.mark.parametrize("initial_nodes", [1, 2, 10, 20])
@pytest.mark.parametrize("do_trim", [True, False])
@pytest.mark.parametrize("max_depth, vals_exp", [
    [1, [5,6,2]],
    [2, [5,6,2,1,3,1,2,2]],
    [3, [5,6,2,1,3,1,2,2,2,1,1,1]],
    [4, [5,6,2,1,3,1,2,2,2,1,1,1,1]],
    [5, [5,6,2,1,3,1,2,2,2,1,1,1,1]],
])
def test_iternodevalues_multikmer_tree1(
        initial_nodes, memmap, filename, mode, do_trim, 
        multikmer_tree1, max_depth, vals_exp
):
    tree, info = multikmer_tree1
    if do_trim:
        tree.trim_free_nodes()
    res = [x for x in tree.iternodevalues(max_depth=max_depth)]
    assert np.all(np.equal(res, vals_exp)), \
        f"Expected {vals_exp}. Got {res}."


@pytest.mark.parametrize("memmap, filename, mode", [
     [False, None, None],
     [True, f"{DATDIR}/tmp_mktree1w.npz", "w+"],
     [True, f"{DATDIR}/tmp_mktree1r.npz", "r+"],
     [True, f"{DATDIR}/tmp_mktree1c.npz", "c"],
])
@pytest.mark.parametrize("initial_nodes", [1, 2, 10, 20])
@pytest.mark.parametrize("do_trim", [True, False])
@pytest.mark.parametrize("max_depth, vals_exp", [
    [1, [2,3,0]],
    [2, [2,3,0,0,2,1,1,0]],
    [3, [2,3,0,0,2,1,1,0,1,0,0,0]],
    [4, [2,3,0,0,2,1,1,0,1,0,0,0,0]],
    [5, [2,3,0,0,2,1,1,0,1,0,0,0,0]],
])
def test_num_children_of_nodes(
        initial_nodes, memmap, filename, mode, do_trim, 
        multikmer_tree1, max_depth, vals_exp
):
    tree, info = multikmer_tree1
    if do_trim:
        tree.trim_free_nodes()
    node_func = lambda x: tree._get_numchildren_at_node(x)
    res = [
        x for x in tree._iternodes(max_depth=max_depth, node_func=node_func)
    ]
    assert np.all(np.equal(res, vals_exp)), \
        f"Expected {vals_exp}. Got {res}."
