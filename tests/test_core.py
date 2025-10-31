"""Tests for core functions.

"""

import pytest

import os
import numpy as np

from humprot.trees.multikmer_tree import MultiKmerTree
from humprot.helpers import get_kmers_from_sequences, int2sym, sym2int
from humprot.core import get_substitution_counts_at_position
from humprot.core import get_substitution_counts_across_sequence

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
@pytest.mark.parametrize(
     "sequence, position, kmin, kmax, mask, candidates, counts_exp", [
     [
        [0,3,0,3,2,3,2], 3, 
        2, 3, 
        3, [0, 1, 2], 
        {
            2: [[1,3,0], [0,2,0]], 
            3: [[1,0,0], [0,1,0], [0,0,0]],
        },
    ],[
        [0,3,0,3,2,3,2], 3, 
        2, 2, 
        3, [0, 1, 2], 
        {
            2: [[1,3,0], [0,2,0]], 
        },
    ],[
        [0,3,0,3,2,3,2], 1, 
        2, 3, 
        3, [0, 1, 2], 
        {
            2: [[1,3,0], [1,1,0]], 
            3: [[-1,-1,-1], [0,0,0], [0,1,0]],
        },
    ],
])
def test_get_substitution_counts_at_position(
        initial_nodes, multikmer_tree1, 
        sequence, position, kmin, kmax, mask, candidates, 
        counts_exp,
):
    tree = multikmer_tree1

    counts = get_substitution_counts_at_position(
         sequence, position, kmin, kmax, candidates, tree, mask,
    )

    errors = []
    if set(counts_exp.keys()) != set(counts.keys()):
        msg = f"Mismatch in count keys!"
        msg += f"\nExpected {set(counts_exp.keys())}"
        msg += f"\nGot {set(counts.keys())}"
        errors.append(msg)

    for k in counts.keys():
        c = counts[k]
        c_exp = counts_exp[k]
        if not np.all(np.equal(c, c_exp)):
            msg = f"Mismatch in counts for k={k}"
            msg += f"\nExpected:\n{c_exp}"
            msg += f"\nGot:\n{c}"
            errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))


@pytest.mark.parametrize("initial_nodes", [1, 2, 10, 20])
@pytest.mark.parametrize(
     "sequence, kmin, kmax, mask, candidates, counts_by_position_exp", [
     [
        [0,3,0,3,2,3,2],
        2, 3, 
        3, [0, 1, 2], 
        {
            1: {
                2: [[1,3,0], [1,1,0]], 
                3: [[-1,-1,-1], [0,0,0], [0,1,0]],
            },
            3: {
                2: [[1,3,0], [0,2,0]], 
                3: [[1,0,0], [0,1,0], [0,0,0]],
            },
            5: {
                2: [[0,0,0], [0,2,0]], 
                3: [[0,0,0], [0,0,0], [-1,-1,-1]],
            },
        },
    ],[
        [0,3,0,3,2,3,2],
        2, 2, 
        3, [0, 1, 2], 
        {
            1: {
                2: [[1,3,0], [1,1,0]], 
            },
            3: {
                2: [[1,3,0], [0,2,0]], 
            },
            5: {
                2: [[0,0,0], [0,2,0]], 
            },
        },
    ],
])
def test_get_substitution_counts_across_sequence(
        initial_nodes, multikmer_tree1, 
        sequence, kmin, kmax, mask, candidates, 
        counts_by_position_exp,
):
    tree = multikmer_tree1

    for _ in range(2):
        counts_by_position = get_substitution_counts_across_sequence(
            sequence, kmin, kmax, candidates, tree, mask,
        )

        errors = []
        if set(counts_by_position_exp.keys()) != set(counts_by_position.keys()):
            msg = f"Mismatch in counts_by_position keys!"
            msg += f"\nExpected {set(counts_by_position_exp.keys())}"
            msg += f"\nGot {set(counts_by_position.keys())}"
            errors.append(msg)
        for j in counts_by_position.keys():
            counts = counts_by_position[j]
            counts_exp = counts_by_position_exp[j]
            if set(counts_exp.keys()) != set(counts.keys()):
                msg = f"Mismatch in counts_by_position keys for j={j}!"
                msg += f"\nExpected {set(counts_by_position_exp.keys())}"
                msg += f"\nGot {set(counts_by_position.keys())}"
                errors.append(msg)
            for k in counts.keys():
                c = counts[k]
                c_exp = counts_exp[k]
                if not np.all(np.equal(c, c_exp)):
                    msg = f"Mismatch in counts for k={k}"
                    msg += f"\nExpected:\n{c_exp}"
                    msg += f"\nGot:\n{c}"
                    errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
