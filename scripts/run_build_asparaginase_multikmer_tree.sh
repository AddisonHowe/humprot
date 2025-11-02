#!/usr/bin/env bash

datdir=data/asp_seqs
outdir=data/asp_trees
infile=P0A962.fasta
outfname=asp_P0A962

kmax=25


fnamecomplete="${outfname}_mktree_${kmax}.npz"
fpath="${outdir}/${fnamecomplete}"
if [ -f "$fpath" ]; then
    echo "Warning: '$fpath' already exists and will be overwritten."
    read -p "Continue? (y/n): " answer
    if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
        echo "Aborted."
        exit 1
    fi
fi

python scripts/build_multikmer_tree.py \
    -i $infile -k $kmax -d $datdir -o $outdir \
    --outfname $fnamecomplete --pbar
