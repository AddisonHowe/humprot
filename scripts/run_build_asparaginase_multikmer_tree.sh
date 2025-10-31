#!/usr/bin/env bash

datdir=data/asp_seqs
outdir=out/asparaginase_multikmer_trees
infile=AAA23446.1.fasta
outfname=asp_AAA23446.1

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
