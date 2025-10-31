#!/usr/bin/env bash

for k in {21..25}; do
    echo k=$k
    python scripts/build_human_kmer_tree.py \
        -i human_proteins.faa -k $k -d ./data -o out/human_kmer_trees \
        --outfname human_kmer_tree --pbar
done
