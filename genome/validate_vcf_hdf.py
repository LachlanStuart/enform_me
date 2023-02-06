from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from kipoiseq import Interval, Variant, extractors, transforms
from pyfaidx import Fasta


chrs = pd.read_hdf("data/my_genome.vcf.hdf", key="vcf").chr.cat.categories
# chr = chrs[0]
chr = "chrY"
fasta = Fasta("/Volumes/Backup/Genome/hg38/hg38.fa")
variants = [
    Variant(chr, pos, ref, alt, _id)
    for pos, ref, alt, _id in pd.read_hdf("data/my_genome.vcf.hdf", key="vcf")
    .loc[lambda df: df.chr == chr, ["pos", "ref_seq", "alt_seq", "id"]]
    .itertuples(False, None)
]

ref_ex = extractors.FastaStringExtractor("/Volumes/Backup/Genome/hg38/hg38.fa", use_strand=True, force_upper=True)
var_ex = extractors.VariantSeqExtractor(reference_sequence=ref_ex)
variant = variants[72]
ref = var_ex.extract(Interval(chr, variant.pos - 10, variant.pos + 10), [], variant.pos, fixed_len=True)
var = var_ex.extract(Interval(chr, variant.pos - 10, variant.pos + 10), variants, variant.pos, fixed_len=True)
# transforms.F.one_hot_dna(var.upper())
print(variant) or print(ref) or print(var)
print(len(variants))
print("done")


def old():
    def split_fasta_to_chrs():
        filename = "/Volumes/Backup/Genome/hg38/hg38.fa"
        chr_file = None
        for line in open(filename, "rt").readlines():
            line = line.strip()
            if line.startswith(">"):
                print(line)
                if chr_file:
                    chr_file.close()
                chr_file = open(f"/Volumes/Backup/Genome/hg38/chrs/{line[1:]}", "wt")
            else:
                chr_file.write(line)

        chr_file.close()

    df = pd.read_hdf("data/my_genome.vcf.hdf", key="vcf").reset_index().drop(columns=["id", "alt_type"])

    df["is_match"] = pd.Series(255, index=df.index, dtype="uint8")
    chrs = df.chr.unique()

    for chr in chrs:
        try:
            chr_data = open(f"/Volumes/Backup/Genome/hg38/chrs/{chr}").read().upper()
            print(f"{chr}")
            idx = df.index[df.chr == chr]
            df.loc[idx, "is_match"] = [
                chr_data[pos - 1 : pos + len(ref_seq) - 1] == ref_seq
                for pos, ref_seq in zip(df.pos[idx], df.ref_seq[idx])
            ]
            # print(pd.Series(
            #     [chr_data[pos-1:pos + len(ref_seq)-1] == ref_seq for pos, ref_seq in zip(df.pos[mask], df.ref_seq[mask])]
            # ).value_counts())
            # foo = df[mask].assign(
            #     hood=lambda subs: [chr_data[pos-5:pos+5] for pos in subs.pos],
            #     is_match=lambda subs: [chr_data[pos-1:pos + len(ref_seq)-1] == ref_seq for pos, ref_seq in zip(subs.pos, subs.ref_seq)]
            # )
        except IOError as err:
            print(f"{chr=} {err=}")

    df["is_match"] = df["is_match"].astype("uint8")
    chr_matches = df.groupby("chr").is_match.value_counts()
