"""Core functions

"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from humprot.trees import MultiKmerTree


def get_substitution_counts_at_position(
        sequence: NDArray[np.int_],
        position: int,
        kmin: int,
        kmax: int,
        candidates: ArrayLike,
        tree: MultiKmerTree,
        mask: int,
) -> dict[int, NDArray[np.int_]]:
    """Count kmers conditional on specifying different characters at position j.

    Args:
        sequence (NDArray[np.int_]): Length L sequence, containing masks.
        position (int): Index j in sequence being specified.
        kmin (int): Starting value of k.
        kmax (int): Final value of k (inclusive).
        candidates (ArrayLike): Values to specify at position j.
        tree (MultiKmerTree): Tree specifying kmers.
        mask (int): Value of the mask.

    Returns:
        dict[int, NDArray[np.int_]]: Dictionary of counts. Value corresponding 
            to key `k` is a k by c array, with c the number of candidate 
            characters. The value counts[k][a,b] is the number of times the
            kmer KMER_{j,k,a,b} occurs in the given tree. KMER_{j,k,a,b} is the
            k-mer overlapping position j, covering the indices 
                                [j-k+a+1, j+a+1)
            of the given sequence. If these start and stop indices exceed the 
            bounds of the sequence, the value is -1.
    """
    counts_by_k = {}
    num_aas = len(candidates)
    for k in range(kmin, kmax + 1):
        kmer = np.zeros(k, dtype=int)
        counts_by_k[k] = np.zeros([k, num_aas], dtype=int)
        for i in range(k):
            start = position - k + 1 + i
            stop = position + 1 + i
            if start < 0 or stop > len(sequence):
                counts_by_k[k][i,:] = -1
            else:
                for aaidx, aa in enumerate(candidates):
                    kmer[:] = sequence[start:stop]
                    kmer[-(i + 1)] = aa
                    score = tree.query_masked(kmer, mask)
                    counts_by_k[k][i,aaidx] = score
    return counts_by_k


def get_substitution_counts_across_sequence(
        sequence: NDArray[np.int_],
        kmin: int,
        kmax: int,
        candidates: ArrayLike,
        tree: MultiKmerTree,
        mask: int,
) -> dict[int, dict[int, NDArray[np.int_]]]:
    """Apply kmer substitutions across all masked sequence positions.

    Args:
        sequence (NDArray[np.int_]): Length L sequence, containing masks.
        kmin (int): Starting value of k.
        kmax (int): Final value of k (inclusive).
        candidates (ArrayLike): Values to specify at position j.
        tree (MultiKmerTree): Tree specifying kmers.
        mask (int): Value of the mask.

    Returns:
        dict[dict[int, NDArray[np.int_]]]: Nested count dictionaries, per 
            sequence position. Given positional index j and key k, the element 
            counts_by_position[j][k] is a k by c array, with c the number of 
            candidate characters. The value counts[k][a,b] is the number of 
            times the kmer KMER_{j,k,a,b} occurs in the given tree. 
            KMER_{j,k,a,b} is the k-mer overlapping position j, covering the 
            indices 
                                [j-k+a+1, j+a+1)
            of the given sequence. If these start and stop indices exceed the 
            bounds of the sequence, the value is -1.
    """
    counts_by_position = {}
    for j in range(len(sequence)):
        if sequence[j] == mask:
            counts_by_position[j] = get_substitution_counts_at_position(
                sequence, 
                position=j, 
                kmin=kmin, 
                kmax=kmax, 
                candidates=candidates,
                tree=tree, 
                mask=mask,
            )
    return counts_by_position


def pretty_print_substitution_counts(
        sequence, counts_by_position, candidates, *, 
        sequence_str,
        int2sym_map, 
        min_count=1
):
    for j in range(len(sequence)):
        if j not in counts_by_position:
            continue
        kmer_counts = counts_by_position[j]
        kmin = sorted(list(kmer_counts.keys()))[0]
        kmax = sorted(list(kmer_counts.keys()))[-1]
        for k in range(kmin, kmax + 1):
            counts = kmer_counts[k]
            for jj in range(k):
                kmerhatstr = f"KMER[j={j},k={k},jj={jj}]"
                for aaidx, aa in enumerate(candidates):
                    start = j - k + 1 + jj
                    stop = j + 1 + jj
                    count = counts[jj, aaidx]
                    kmerstring_mask = "".join(
                        [" " if ii < start or ii >= stop else ("?" if ii == j else "*") 
                        for ii in range(len(sequence))]
                    )
                    if start < 0 or stop > len(sequence):
                        newseq = "<undefined>"
                        kmer = "NULL"
                    else:
                        kmer = sequence_str[start:stop]
                        newkmer = kmer[:-jj-1] + int2sym_map[aa] + kmer[k-jj:]
                        newseq = sequence_str[:start] + newkmer + sequence_str[stop:]
                        assert len(kmer) == k
                        assert len(kmer) == len(newkmer), kmer + "   " + newkmer
                    s = "{} = {} : aa={}, count={} | newseq: {}".format(
                        kmerhatstr, kmer, int2sym_map[aa], count, newseq
                    )
                    if count >= min_count:
                        print(s)
                        print(f"\t         {kmerstring_mask}")
                        print(f"\told seq: {sequence_str}")
                        print(f"\tnew seq: {newseq}")
                        print("-" * 60)
