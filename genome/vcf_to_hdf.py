import numpy as np
import pandas as pd
import vcfpy
from tqdm import tqdm

# filename = "/Volumes/Backup/Genome/.vcf"
# out_file = 'data/.vcf.hdf'
#
# df = pd.read_hdf(out_file, key="vcf").reset_index().drop(columns=['id', 'alt_type'])
# print(len(df))
# print(len(df[["chr","pos"]].drop_duplicates(ignore_index=True)))


def vcf_to_hdf(filename: str, out_file: str):
    rows = []
    with vcfpy.Reader.from_path(filename) as reader:
        for record in tqdm(reader, total=4730000):
            assert len(record.ID) <= 1
            assert len(record.ALT) >= 1
            var_name = record.ID[0] if record.ID else None
            for alt_id, alt in enumerate(record.ALT):
                rows.append(
                    (
                        record.CHROM,
                        record.POS,
                        var_name,
                        alt_id,
                        record.REF,
                        alt.type,
                        alt.value,
                        record.QUAL,
                        record.INFO["DP"],
                    )
                )
    print("to_df")
    df = pd.DataFrame(
        rows, columns=["chr", "pos", "id", "alt_idx", "ref_seq", "alt_type", "alt_seq", "quality", "depth"]
    )
    print(df.memory_usage().sum())
    for col in df.columns:
        print(col)
        print(df[col].describe())
    df["chr"] = df["chr"].astype("category")
    df["pos"] = df["pos"].astype(np.int32)
    df["alt_idx"] = df["alt_idx"].astype(np.uint8)
    df["ref_seq"] = df["ref_seq"].astype("category")
    df["alt_seq"] = df["alt_seq"].astype("category")
    df["alt_type"] = df["alt_type"].astype("category")
    df["quality"] = df["quality"].astype(np.float32)
    df["depth"] = df["depth"].astype(np.uint8)
    df.to_hdf(out_file, key="vcf", mode="w", format="table", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("out_file")
    args = parser.parse_args()
    vcf_to_hdf(args.filename, args.out_file)
