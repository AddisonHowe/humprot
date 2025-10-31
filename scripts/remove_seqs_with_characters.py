#!/usr/bin/env python3

import sys
import numpy as np
from Bio import SeqIO

def remove_seqs_with_characters(input_fasta, output_fasta, to_remove):
    """Read a FASTA file and write sequences that do NOT contain elements."""
    with open(input_fasta) as infile, open(output_fasta, "w") as outfile:
        kept_records = (
            record for record in SeqIO.parse(infile, "fasta")
            if np.all([x not in str(record.seq) for x in to_remove])
        )
        count = SeqIO.write(kept_records, outfile, "fasta")

    print(f"Wrote {count} sequences (removed those containing {to_remove}).")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python remove_seqs_with_characters.py " \
              "input.fasta output.fasta SYM1 [SYM2 [SYM3 ...]]")
        sys.exit(1)

    input_fasta = sys.argv[1]
    output_fasta = sys.argv[2]
    to_remove = sys.argv[3:]

    remove_seqs_with_characters(input_fasta, output_fasta, to_remove)
