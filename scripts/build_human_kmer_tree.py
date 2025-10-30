"""Construct a kmer tree from a human proteome.

"""


import argparse
import os, sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from Bio import SeqIO
import tqdm as tqdm
from importlib import reload

import humprot
from humprot.helpers import sym2int, int2sym
from humprot.helpers import get_kmers_from_sequences
from humprot.helpers import count_kmers_in_seqs
from humprot.compact_trie import CompactTrie as KmerTree


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infname", type=str, required=True,
                        help="Input filename. E.g. 'human_proteins.faa'")
    parser.add_argument("-k", type=int, nargs="+", required=True)
    parser.add_argument("-d", "--datdir", type=str, required=True)
    parser.add_argument("-o", "--outdir", type=str, required=True)
    parser.add_argument("-aa", "--aa_list", type=str, default=None)
    parser.add_argument("--outfname", type=str, default="saved_tree")
    parser.add_argument("--pbar", action="store_true")
    parser.add_argument("-v", "--verbosity", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args(args)


def main(args):
    INFNAME = args.infname
    k_list = args.k
    outfname = args.outfname
    DATDIR = args.datdir
    OUTDIR = args.outdir
    aa_list = args.aa_list
    pbar = args.pbar
    SEED = args.seed
    verbosity = args.verbosity

    # Housekeeping
    if SEED == 0:
        SEED = np.random.randint(2**32)
    print("seed:", SEED)
    rng = np.random.default_rng(seed=SEED)

    IMGDIR = os.path.join(OUTDIR, "images")
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(IMGDIR, exist_ok=True)
    
    HUMAN_DATA_FPATH = os.path.join(DATDIR, INFNAME)

    if aa_list is None:
        AA_LIST = list("ACDEFGHIKLMNPQRSTVWYXU")
    else:
        AA_LIST = list(aa_list)

    alphabet_size = len(AA_LIST)
    MASK = len(AA_LIST)
    
    SYM2INT = {sym: i for i, sym in enumerate(AA_LIST)}
    INT2SYM = {i: sym for i, sym in enumerate(AA_LIST)}
    INT2SYM[MASK] = "-"

    # Load protein sequences
    print(f"Loading protein sequences from file: {HUMAN_DATA_FPATH}")
    id2seq = {}
    for record in SeqIO.parse(HUMAN_DATA_FPATH, "fasta"):
        seqid = record.id
        seq = sym2int(record.seq, SYM2INT)
        id2seq[seqid] = seq
    print(f"Loaded {len(id2seq)} protein sequences")

    time0 = time.time()
    for k in k_list:
        # Construct the tree
        print(f"Building {k}-mer tree...")
        t0 = time.time()
        tree = build_kmer_tree(
            id2seq, k, alphabet_size, 
            rng=rng, pbar=pbar, verbosity=verbosity,
        )
        tt1 = time.time()
        print(f"Build complete ({tt1-t0:.3g} seconds)")
        # Save the tree
        saveas = os.path.join(OUTDIR, outfname + f"_{k}.npz")
        print(f"Saving to file: {saveas}")
        tt0 = time.time()
        tree.save(saveas)
        tt1 = time.time()
        print(f"Save complete ({tt1-tt0:.3g} seconds)")
        print(f"{k}-mer tree total time: {tt1 - t0:.3g}")
        del tree
    time1 = time.time()
    print("Done!")
    print(f"Total Elapsed time: {time1-time0:.3g}")
    

def build_kmer_tree(
        id2seq, k, alphabet_size, *, 
        rng=None, verbosity=1, pbar=False
):
    rng = np.random.default_rng(rng)

    # Compute kmers
    if verbosity:
        print("Computing kmer counts...")
    kmer_counts = count_kmers_in_seqs(
        id2seq.values(), k, pbar=pbar, leave_pbar=True,
    )
    if verbosity:
        print("Finished computing kmer counts")

    # Construct kmer tree
    tree = KmerTree(None, mode="w+", alphabet_size=alphabet_size)

    if verbosity:
        print("Adding kmers to tree...")
    max_count = np.inf
    count = 0
    for kmer in tqdm.tqdm(
            kmer_counts, total=min(max_count, len(kmer_counts)), 
            disable=not pbar, mininterval=2
    ):
        if count >= max_count:
            break
        tree.add_kmer(kmer, kmer_counts[tuple(kmer)])
        count += 1
    if verbosity:
        print("Finished adding kmers")

    return tree

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
