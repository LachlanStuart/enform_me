import gzip
from urllib.request import urlopen
from gtfparse import read_gtf

import pandas as pd

GTF_URL = "https://ftp.ensembl.org/pub/release-108/gtf/homo_sapiens/Homo_sapiens.GRCh38.108.gtf.gz"
with urlopen(GTF_URL) as url_stream:
    with gzip.open(url_stream) as gzip_stream:
        df = read_gtf(gzip_stream)[
            lambda df: (df.feature == "gene") & (df.gene_biotype == "protein_coding") & (df.seqname != "GL000194.1")
        ]
        df["chr"] = pd.Categorical("chr" + df.seqname)
        df["strand"] = pd.Categorical(df.strand)
        df["start"] = df["start"].astype("int32")
        df["end"] = df["end"].astype("int32")
        df = df[["chr", "start", "end", "strand", "gene_id", "gene_name"]]
    df.to_hdf("data/genes.hdf", key="genes", format="table")

df.drop(columns=[])
print("done")
print(df.columns)
