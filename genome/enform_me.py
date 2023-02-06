import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.putenv("HDF5_USE_FILE_LOCKING", "FALSE")
from collections import namedtuple
from functools import cached_property
from pathlib import Path
from traceback import print_exc
import pandas as pd
import numpy as np
import gzip
import torch
import tqdm
from pyfaidx import Fasta
from pyfaidx import complement
from kipoiseq import Interval, Variant, extractors, transforms

# from enformer_pytorch import str_to_one_hot
from genome.enformer import Enformer, from_pretrained, str_to_one_hot
from torch.utils.data import Dataset
import lovely_tensors as lt

lt.monkey_patch()


GeneSequence = namedtuple("GeneSequence", ["gene_id", "type", "sequence"])
GeneTuple = namedtuple("GeneTuple", ("Index", "chr", "start", "end", "strand", "gene_id", "gene_name"))
# VARIANT_TYPES = ["base_mid", "my_mid", "mysnp_mid"]
VARIANT_TYPES = ["base", "my", "mysnp", "base_mid", "my_mid", "mysnp_mid"]
EMBEDDING_WIDTH = 3072


class GenomeDataset(Dataset):
    def __init__(self, data_path: Path, ensembl_genes: pd.DataFrame, context_length=196_608):
        # chr, start, end, strand, gene_id, gene_name
        self.data_path = data_path
        self.context_length = context_length
        self.ensembl_genes = ensembl_genes
        # chr, pos, id, alt_idx, ref_seq, alt_type, alt_seq, quality, depth
        self.fasta_path = self.data_path / "hg38/hg38.latest_2021.fa"
        if not self.fasta_path.exists():
            with gzip.open(self.data_path / "hg38/hg38.latest_2021.fa.gz") as src:
                with open(self.fasta_path, "wb") as dest:
                    buf = src.read(2**20)
                    while buf:
                        dest.write(buf)
                        buf = src.read(2**20)

    @cached_property
    def ensembl_genes_tuples(self):
        return [GeneTuple(*tup) for tup in self.ensembl_genes.itertuples()]

    @cached_property
    def chr_variants(self):
        my_vcf = pd.read_hdf(self.data_path / "my_genome.vcf.hdf", key="vcf")
        return {
            chr: [Variant(chr, pos, ref, alt, _id) for pos, ref, alt, _id in chr_vcf.itertuples(False, None)]
            for chr, chr_vcf in my_vcf[["pos", "ref_seq", "alt_seq", "id"]].groupby(my_vcf.chr)
            if chr != "chrEBV"
        }

    @cached_property
    def chr_variant_index(self):
        return {chr: np.array([var.start for var in variants]) for chr, variants in self.chr_variants.items()}

    @cached_property
    def chr_snp_variants(self):
        return {
            chr: [variant for variant in chr_variants if len(variant.ref) == len(variant.alt)]
            for chr, chr_variants in self.chr_variants.items()
        }

    @cached_property
    def chr_snp_variant_index(self):
        return {chr: np.array([var.start for var in variants]) for chr, variants in self.chr_snp_variants.items()}

    @cached_property
    def ref_extractor(self):
        return extractors.FastaStringExtractor(str(self.fasta_path), use_strand=True, force_upper=True)

    @cached_property
    def var_extractor(self):
        def safe_complement(s):
            return orig_complement(s.replace("*", "N"))

        orig_complement = extractors.vcf_seq.complement
        if extractors.vcf_seq.complement.__name__ != "safe_complement":
            extractors.vcf_seq.complement = safe_complement

        return extractors.VariantSeqExtractor(reference_sequence=self.ref_extractor)

    @cached_property
    def extract(self):
        import numba

        var_extractor = self.var_extractor

        @numba.jit(forceobj=True, boundscheck=True)
        def extract(interval, variants, start, fixed_len):
            return var_extractor.extract(interval, variants, start, fixed_len=fixed_len)

        return extract

    def validate(self):
        print("validating")
        errors = 0
        for chr, variants in self.chr_variants.items():
            if chr in ("chrEBV", "chrMT"):
                continue
            for i, variant in enumerate(variants):
                interval = Interval(chr, variant.pos - 6, variant.pos + 4 + len(variant.ref))
                found = self.ref_extractor.extract(interval)
                lr = len(variant.ref)
                la = len(variant.alt)
                read_var = self.var_extractor.extract(interval, [variant], 0, fixed_length=False)
                if found[5 : 5 + lr] != variant.ref or read_var[5 : 5 + la] != variant.alt[: lr + 5]:
                    print(
                        f"{chr}[{i}] {found[:5]} {found[5:5+lr]} {found[5+lr:]} -> {read_var[:5]} {read_var[5:5+la]} {read_var[5+la:]} expected {variant.ref} -> {variant.alt}"
                    )
                    errors += 1
                    if errors > 5:
                        raise AssertionError()
        print("validated")

    def __getitem__(self, item):
        variant_type = VARIANT_TYPES[item % len(VARIANT_TYPES)]
        gene_row = self.ensembl_genes_tuples[item // len(VARIANT_TYPES)]
        offset = 64 if variant_type.endswith("_mid") else 0
        # interval = self.gene_intervals[item // len(VARIANT_TYPES)]
        interval = Interval(
            gene_row.chr,
            gene_row.start - self.context_length // 2 + offset,
            gene_row.start + self.context_length // 2 + offset,
            strand=gene_row.strand,
        )
        variant_mode = variant_type.split("_")[0]
        if variant_mode == "base":
            variants = []
            var_idx = []
        elif variant_mode == "my":
            variants = self.chr_variants[gene_row.chr]
            var_idx = self.chr_variant_index[gene_row.chr]
        elif variant_mode == "mysnp":
            variants = self.chr_snp_variants[gene_row.chr]
            var_idx = self.chr_snp_variant_index[gene_row.chr]

        # VariantSeqExtractor doesn't efficiently filter variants to the interval. Do it manually for a big speedup.
        # This variants be presorted (they are), and shorter than the 1000 padding (biggest is 450ish)
        start = np.searchsorted(var_idx, interval.start - 1000, side="left")
        end = np.searchsorted(var_idx, interval.end + 1000, side="right")
        variants = variants[start:end]

        try:
            # seq = self.var_extractor.extract(interval, variants, gene_row.start, fixed_len=True).replace('*', 'N')
            seq = self.extract(interval, variants, gene_row.start, fixed_len=True).replace("*", "N")

            return GeneSequence(gene_row.gene_id, variant_type, str_to_one_hot(seq))
        except Exception:
            print_exc()
            return GeneSequence(gene_row.gene_id, variant_type, torch.full((self.context_length, 4), torch.nan))
        # output, embeddings = model(str_to_one_hot([ref_seq, my_seq]), return_embeddings=True)

    def __len__(self):
        return len(self.ensembl_genes_tuples) * len(VARIANT_TYPES)


def init_model():
    with torch.no_grad(), torch.inference_mode(), torch.autocast("cuda"):
        model = from_pretrained("EleutherAI/enformer-official-rough", target_length=2).eval().cuda()

        model = torch.jit.script(model, example_inputs=[(torch.zeros((1, 196_608, 4)).cuda(),)]).cuda()
        model = torch.jit.optimize_for_inference(model).cuda()
        model(torch.zeros((1, 196_608, 4)).cuda())
    return model


def test_with_batch_size(model, batch_size):
    try:
        with torch.no_grad(), torch.inference_mode(), torch.autocast("cuda"):
            _ = model(torch.zeros((batch_size, 196_608, 4), device="cuda"))
        return True
    except:
        print_exc()
        return False


def get_max_batch_size(model):
    for i in range(12, 2048, 3):
        print(i)
        if not test_with_batch_size(model, i):
            return i
    return i


def batch_iter(dataset, batch_size=2):
    for batch_start in tqdm.tqdm(range(0, len(dataset), batch_size)):
        batch_items = [dataset[batch_start + i] for i in range(batch_size)]
        yield GeneSequence(
            [it.gene_id for it in batch_items],
            [it.type for it in batch_items],
            torch.stack([it.sequence for it in batch_items]),
        )


def init_or_load_embeddings(data_path: Path):
    ensembl_genes = pd.read_hdf(data_path / "genes.hdf", key="genes")
    ensembl_genes = ensembl_genes[ensembl_genes.chr != "chrMT"]  # mitochondria variants are missing?
    embeddings_paths = {
        (ty, side): data_path / f"embeddings_{ty}_{side}.h5" for ty in VARIANT_TYPES for side in ["before", "after"]
    }
    try:
        print(f"Loading {embeddings_paths}")
        embeddings = {key: pd.read_hdf(path, key="embeddings") for key, path in embeddings_paths.items()}
        all_embeddings = pd.concat([df.isna().sum(1) == 0 for df in embeddings.values()], axis=1)
        all_embeddings.columns = [",".join(key) for key in embeddings.keys()]
        print(f"Loaded embedding stats: {all_embeddings.sum()}")
        existing_gene_ids = all_embeddings.index[all_embeddings.all(axis=1)]
        missing_gene_ids = all_embeddings.index[~all_embeddings.all(axis=1)]
        ensembl_genes = ensembl_genes[ensembl_genes.gene_id.isin(missing_gene_ids)]
        print(f"Loaded embeddings for {len(existing_gene_ids)} ({len(missing_gene_ids)} remaining)")
    except (IOError, AttributeError):
        embeddings = {
            key: pd.DataFrame(
                np.full((len(ensembl_genes), EMBEDDING_WIDTH), np.nan),
                index=ensembl_genes.gene_id,
                columns=range(EMBEDDING_WIDTH),
            )
            for key, path in embeddings_paths.items()
        }

    return ensembl_genes, embeddings_paths, embeddings


def save_embeddings(embeddings_paths, embeddings):
    for key, path in embeddings_paths.items():
        print(f"Saving {path}")
        embeddings[key].to_hdf(path.with_suffix(".tmp"), key="embeddings")
        path.unlink(missing_ok=True)
        path.with_suffix(".tmp").rename(path)


def enform_me(
    data_path: Path,
    model=None,
    dataset=None,
):
    ensembl_genes, embeddings_paths, embeddings = init_or_load_embeddings(data_path)
    clear_mem()
    if model is None:
        model = init_model()
    if dataset is None:
        dataset = GenomeDataset(data_path, ensembl_genes)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        pin_memory_device="cuda",
        prefetch_factor=3,
        drop_last=False,
        persistent_workers=True,
    )
    try:
        # raise AssertionError()
        for i, batch in enumerate(tqdm.tqdm(loader)):
            # for batch in batch_iter(dataset, 1):

            preds = model(batch.sequence.half().cuda()).detach().cpu().numpy()
            for gene_id, variant_type, pred in zip(batch.gene_id, batch.type, preds):
                embeddings[(variant_type, "before")].loc[gene_id] = pred[0]
                embeddings[(variant_type, "after")].loc[gene_id] = pred[1]
            if i % 1000 == 0:
                save_embeddings(embeddings_paths, embeddings)
    finally:
        save_embeddings(embeddings_paths, embeddings)

    return embeddings


def clear_mem():
    import gc, sys

    if "last_traceback" in dir(sys):
        del sys.last_type
        del sys.last_value
        del sys.last_traceback
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    enform_me(Path("C:\\Genome").resolve())

if False:
    from genome.enform_me import *

    data_path = Path("C:\\Genome").resolve()
    model = init_model()
    get_max_batch_size(model)

    from genome.enform_me import *

    data_path = Path("C:\\Genome").resolve()
    ensembl_genes = pd.read_hdf(data_path / "genes.hdf", key="genes")
    dataset = GenomeDataset(data_path, ensembl_genes)
    enform_me(data_path)

    from genome.enform_me import *

    data_path = Path("C:\\Genome").resolve()
    ensembl_genes = pd.read_hdf(data_path / "genes.hdf", key="genes")

    model = init_model()
    # (data_path / "embeddings.h5").unlink()
    enform_me(data_path, model, dataset=None)
