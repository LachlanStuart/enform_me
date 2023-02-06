# Enform Me

Enform Me is a WIP project for using Enformer predictions to explore how one's
personal genome differs in predicted expression to a reference genome.

This is mainly just scripts for data exploration and is unlikely to be bundled
into an easily executable format.

### Scripts

* vcf_to_hdf.py - Parses a VCF file, extracting minimal fields and saving them as a much faster HDF file
* validate_vcf_hdf.py - Validates that the correct reference genome is being used by comparing reference sequences
* ensembl.py - Downloads the Ensembl gene database, extracts useful fields and converts to HDF
* enform_me.py - Runs Enformer across the reference genome and personal genome
* enformer.py - A hackily-modified version of [enformer-pytorch](https://github.com/lucidrains/enformer-pytorch)
to improve inference speed
