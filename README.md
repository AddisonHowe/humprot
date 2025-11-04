# Humanizing proteins

## Setup

For an editable installation, clone the repository and create a conda environment as follows:

```bash
git clone https://github.com/AddisonHowe/humprot
cd humprot
conda env create -n humprot-env -f environment.yml
```

Check that tests pass.

```bash
conda activate humprot-env
pytest tests
```

## Demo

The notebook `notebooks/mktree_demo.ipynb` provides examples of how the `MultiKmerTree` class can be used to query $k$-mers in a given set.

## Data

### Human proteome

We can access the human proteome through [UniProt Accession UP000005640](https://www.uniprot.org/proteomes/UP000005640). We download the [83,156 entries](https://www.uniprot.org/uniprotkb?query=proteome:UP000005640) as a fasta file, stored as  `data/human_proteins.faa`. [[Click to download](https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta&query=%28proteome%3AUP000005640%29)]

### L-asparaginase sequence

From UniProt, we can find various L-asparaginase amino acid sequences:

| Entry | Sequence file |
| -------- | -------- |
| [ASPG1_ECOLI](https://www.uniprot.org/uniprotkb/P0A962/entry#sequences) | [P0A962](https://rest.uniprot.org/uniprotkb/P0A962.fasta) |

These are stored in the directory `data/asp_seqs`.
