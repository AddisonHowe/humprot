"""Helper functions

"""

import numpy as np
from numpy.typing import NDArray
import tqdm as tqdm


def sym2int(
        seq: str, 
        mapping: dict[str,int],
) -> NDArray[np.uint8]:
    return np.array([mapping[c] for c in seq], dtype=np.uint8)


def int2sym(
        seq: NDArray[np.uint8], 
        mapping: dict[int,str]
) -> str :
    return "".join([mapping[i] for i in seq])


def get_sequence_kmers(
        seq: NDArray[np.uint8], 
        k: int,
) -> NDArray[np.uint8]:
    if k > seq.shape[0]:
        return np.empty((0, k), dtype=seq.dtype)
    return np.lib.stride_tricks.sliding_window_view(seq, window_shape=k)


def get_kmers_from_sequences(
        seqs,  
        k: int,
) -> NDArray[np.uint8]:
    kmers = []
    for seq in seqs:
        kmers.append(get_sequence_kmers(seq, k))
    return np.concatenate(kmers)


def count_kmers_in_seqs(
        seqs,
        k: int,
        max_num: int = -1,
        pbar: bool = False,
        leave_pbar: bool = True,
):
    if max_num < 0:
        max_num = np.inf
    kmer_counts = {}
    nseqs_processed = 0
    for seq in tqdm.tqdm(seqs, total=len(seqs), disable=not pbar, leave=leave_pbar):
        if nseqs_processed >= max_num:
            break
        kmers = get_sequence_kmers(seq, k)
        for kmer in kmers:
            kmer_key = tuple(kmer)
            if kmer_key in kmer_counts:
                kmer_counts[kmer_key] += 1
            else:
                kmer_counts[kmer_key] = 1
        nseqs_processed += 1
    return kmer_counts
