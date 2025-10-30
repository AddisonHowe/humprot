"""Tests for helper functions.

"""

import pytest
import numpy as np

from humprot.helpers import get_sequence_kmers, get_kmers_from_sequences
from humprot.helpers import count_kmers_in_seqs

###################
##  Begin Tests  ##
###################

@pytest.mark.parametrize('seq, k, kmers_exp', [
    [
        [0,1,2,3], 3, 
        [[0, 1, 2], [1, 2, 3]]
    ],[
        [0,1,2,3], 4, 
        [[0, 1, 2, 3]]
    ],[
        [0,1,2,3], 5, 
        []
    ],
])
def test_get_sequence_kmers(seq, k, kmers_exp):
    seq = np.array(seq, dtype=np.uint8)
    kmers_exp = np.array(kmers_exp, dtype=np.uint8).reshape([-1,k])
    # Compute kmers
    kmers = get_sequence_kmers(seq, k)
    errors = []
    if kmers.shape != kmers_exp.shape:
        msg = f"Shape mismatch. {kmers.shape} != {kmers_exp.shape} <- expected"
        errors.append(msg)
    if not np.all(kmers == kmers_exp):
        msg = f"Value mismatch. Expected {kmers_exp}. Got {kmers}."
        errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
    
    
@pytest.mark.parametrize('seqs, k, kmers_exp', [
    [
        [[0,1,2,3],[1,2,3,4],[2,3,4,5]], 3,
        [[0,1,2],[1,2,3],[1,2,3],[2,3,4],[2,3,4],[3,4,5]]
    ],[
        [[0,1,2,3],[1,2,3,4],[2,3,4,5]], 4,
        [[0,1,2,3],[1,2,3,4],[2,3,4,5]]
    ],[
        [[0,1,2,3],[1,2,3,4],[2,3,4,5]], 5,
        []
    ],
])
def test_get_kmers_from_sequences(seqs, k, kmers_exp):
    seqs = np.array(seqs, dtype=np.uint8)
    kmers_exp = np.array(kmers_exp, dtype=np.uint8).reshape([-1, k])
    kmers = get_kmers_from_sequences(seqs, k)
    errors = []
    if kmers.shape != kmers_exp.shape:
        msg = f"Shape mismatch. {kmers.shape} != {kmers_exp.shape} <- expected"
        errors.append(msg)
    if not np.all(kmers == kmers_exp):
        msg = f"Value mismatch. Expected {kmers_exp}. Got {kmers}."
        errors.append(msg)
    assert not errors, "Errors occurred:\n{}".format("\n".join(errors))

    assert np.all(kmers_exp == kmers)


@pytest.mark.parametrize("seqs, k, kmer_counts_exp", [
    [
        [[0,1,2,3],[1,2,3,4],[2,3,4,5]], 3,
        {(0,1,2):1, (1,2,3):2, (2,3,4):2, (3,4,5):1}
    ],[
        [[0,1,2,3],[1,2,3,4],[2,3,4,5]], 4,
        {(0,1,2,3): 1, (1,2,3,4): 1, (2,3,4,5): 1} 
    ],[
        [[0,1,2,3],[1,2,3,4],[2,3,4,5]], 5,
        {}
    ],
])
def test_count_kmers_in_seqs(seqs, k, kmer_counts_exp):
    seqs = np.array(seqs, dtype=np.uint8)
    kmer_counts = count_kmers_in_seqs(seqs, k)
    assert kmer_counts == kmer_counts_exp
    