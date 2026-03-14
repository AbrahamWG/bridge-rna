"""
RNA-seq preprocessing module for ARCHS4 expression data.

Handles: ortholog loading, gene selection, TPM normalization,
log transformation, QC filtering, cross-split deduplication, and export.

Usage:
    from preprocessing import RNADatasetBuilder

    builder = RNADatasetBuilder(
        species="both",
        gene_set="shared_orthologs",
    )
    builder.process()
    builder.save_parquet("output/")

    # Split later with scikit-learn
    X, meta = builder.get_data()
"""

import os
import gc
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================
@dataclass
class PreprocessingConfig:
    """All preprocessing parameters in one place."""

    # --- Species & gene set ---
    species: str = "both"                    # "human", "mouse", "both"
    gene_set: str = "shared_orthologs"       # "shared_orthologs", "union_orthologs"

    # --- Sample count per species ---
    n_samples: int = 120_000

    # --- QC ---
    qc_min_nonzero: int = 14_000             # min non-zero genes per sample
    remove_single_cell: bool = True

    # --- Normalization ---
    normalization: str = "log1p_tpm"         # "log1p_tpm", "raw_counts"
    extraction_batch_size: int = 10_000

    # --- Paths ---
    archs4_dir: str = "data/archs4"
    orthologs_file: str = "data/ensembl/orthologs_one2one.txt"
    exon_lengths_human: str = "data/gencode/gencode_v49_gene_exon_lengths.csv"
    exon_lengths_mouse: str = "data/gencode/gencode_v49_mouse_gene_exon_lengths.csv"
    output_dir: str = "data/archs4/train_orthologs"

    # --- Reproducibility ---
    seed: int = 42


# ============================================================
# GENE REGISTRY
# ============================================================
class GeneRegistry:
    """
    Loads ortholog mappings and determines the canonical gene list
    based on the chosen gene_set mode.
    """

    def __init__(self, orthologs_file: str):
        self.ortho_df = pd.read_csv(orthologs_file, sep="\t")

        # Mouse→Human and Human→Mouse mappings
        self.mouse_to_human = dict(
            zip(self.ortho_df["Gene name"], self.ortho_df["Human gene name"])
        )
        self.human_to_mouse = dict(
            zip(self.ortho_df["Human gene name"], self.ortho_df["Gene name"])
        )

        # All valid human ortholog gene names (no NaN)
        self.all_human_ortho = sorted(
            g for g in self.ortho_df["Human gene name"].unique()
            if isinstance(g, str)
        )
        self.all_mouse_ortho = sorted(
            g for g in self.ortho_df["Gene name"].unique()
            if isinstance(g, str)
        )

    def get_canonical_genes(
        self,
        gene_set: str,
        human_zero_genes: Optional[set] = None,
        mouse_zero_genes: Optional[set] = None,
    ) -> list[str]:
        """
        Return the canonical gene list (human gene symbols, sorted).

        Args:
            gene_set: "shared_orthologs" or "union_orthologs"
            human_zero_genes: genes with zero expression across all human samples
            mouse_zero_genes: genes with zero expression across all mouse samples
        """
        human_zero = human_zero_genes or set()
        mouse_zero = mouse_zero_genes or set()

        if gene_set == "shared_orthologs":
            # Only genes expressed in BOTH species
            genes = [
                g for g in self.all_human_ortho
                if g not in human_zero and g not in mouse_zero
            ]
        elif gene_set == "union_orthologs":
            # All ortholog genes, even if expressed in only one species
            genes = list(self.all_human_ortho)
        else:
            raise ValueError(f"Unknown gene_set: {gene_set!r}")

        return sorted(genes)


# ============================================================
# EXPRESSION LOADER
# ============================================================
class ExpressionLoader:
    """Loads and normalizes expression data from ARCHS4 H5 files."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self._exon_human = None
        self._exon_mouse = None

    @property
    def exon_lengths_human(self) -> pd.Series:
        if self._exon_human is None:
            df = pd.read_csv(self.config.exon_lengths_human)
            self._exon_human = df.set_index("gene_symbol")["exon_length"]
        return self._exon_human

    @property
    def exon_lengths_mouse(self) -> pd.Series:
        if self._exon_mouse is None:
            df = pd.read_csv(self.config.exon_lengths_mouse)
            self._exon_mouse = df.set_index("gene_symbol")["exon_length"]
        return self._exon_mouse

    def extract_and_normalize(
        self,
        h5_path: str,
        species: str,
        n_samples: int,
        canonical_genes: list[str],
        global_seen: set,
        seed_offset: int = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame, set]:
        """
        Extract random samples from ARCHS4, normalize, align to canonical genes.

        Returns:
            (expression_df, metadata_df, updated_global_seen)
            expression_df: genes × samples (canonical gene index, float32)
            metadata_df: DataFrame with [geo_accession, species] columns
        """
        import archs4py as a4

        gene_lengths = (
            self.exon_lengths_human if species == "human"
            else self.exon_lengths_mouse
        )

        cfg = self.config
        batches = []
        seen_local = set()
        total = 0
        batch_num = 0

        while total < n_samples:
            batch_num += 1
            to_extract = min(cfg.extraction_batch_size, n_samples - total)

            expr = a4.data.rand(
                h5_path, to_extract,
                remove_sc=cfg.remove_single_cell,
                seed=cfg.seed + seed_offset + batch_num,
            )
            if expr.empty or expr.shape[1] == 0:
                print(f"    No more samples available. Stopping.")
                break

            # Aggregate duplicate gene rows
            expr = expr.groupby(level=0).sum()

            # Keep only genes with exon lengths
            expr = expr.loc[expr.index.intersection(gene_lengths.index)]

            # QC: min non-zero genes
            nonzero = (expr > 0).sum(axis=0)
            expr = expr[nonzero[nonzero >= cfg.qc_min_nonzero].index]

            # Local dedup
            new = [s for s in expr.columns if s not in seen_local]
            expr = expr[new]

            # Global dedup
            new = [s for s in expr.columns if s not in global_seen]
            expr = expr[new]

            if expr.shape[1] == 0:
                continue

            # Normalize
            if cfg.normalization == "log1p_tpm":
                lengths_kb = gene_lengths.loc[expr.index].fillna(1000) / 1000.0
                rate = expr.div(lengths_kb, axis=0)
                tpm = rate.div(rate.sum(axis=0), axis=1) * 1e6
                expr = np.log1p(tpm)
            # elif cfg.normalization == "raw_counts": pass through

            # Align to canonical gene list
            expr = expr.reindex(canonical_genes, fill_value=0).astype("float32")

            batches.append(expr)
            seen_local.update(expr.columns)
            total += expr.shape[1]

            print(f"    Batch {batch_num}: +{expr.shape[1]:,} samples "
                  f"({total:,}/{n_samples:,})")

            del expr
            gc.collect()

        if not batches:
            return pd.DataFrame(), pd.DataFrame(), global_seen

        combined = pd.concat(batches, axis=1)
        # Safety: drop any remaining column duplicates
        combined = combined.loc[:, ~combined.columns.duplicated(keep="first")]

        metadata = pd.DataFrame({
            "geo_accession": combined.columns,
            "species": species,
        })

        global_seen.update(combined.columns)
        return combined, metadata, global_seen

    def find_zero_genes(
        self,
        h5_path: str,
        species: str,
        canonical_genes: list[str],
        n_probe: int = 20_000,
    ) -> set[str]:
        """
        Probe a sample of data to find genes with zero expression.
        Returns set of canonical gene names that are all-zero.
        """
        import archs4py as a4

        gene_lengths = (
            self.exon_lengths_human if species == "human"
            else self.exon_lengths_mouse
        )

        expr = a4.data.rand(
            h5_path, n_probe,
            remove_sc=True,
            seed=self.config.seed + 999,
        )
        expr = expr.groupby(level=0).sum()
        expr = expr.loc[expr.index.intersection(gene_lengths.index)]

        # Normalize same as main pipeline
        lengths_kb = gene_lengths.loc[expr.index].fillna(1000) / 1000.0
        rate = expr.div(lengths_kb, axis=0)
        tpm = rate.div(rate.sum(axis=0), axis=1) * 1e6
        expr = np.log1p(tpm)

        expr = expr.reindex(canonical_genes, fill_value=0)
        zero_genes = set(expr.index[expr.sum(axis=1) == 0])
        return zero_genes


# ============================================================
# DATASET BUILDER (main API)
# ============================================================
class RNADatasetBuilder:
    """
    End-to-end RNA-seq preprocessing pipeline.
    Outputs a single concatenated dataset; train/val/test splitting
    is deferred to scikit-learn.

    Usage:
        builder = RNADatasetBuilder(species="both", gene_set="shared_orthologs")
        builder.process()
        builder.save_parquet("output/")
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None, **kwargs):
        """
        Args:
            config: Full PreprocessingConfig, OR pass individual kwargs:
                species, gene_set, train_samples, val_samples, etc.
        """
        if config is not None:
            self.config = config
        else:
            self.config = PreprocessingConfig(**kwargs)

        self.registry = GeneRegistry(self.config.orthologs_file)
        self.loader = ExpressionLoader(self.config)

        # Populated by process()
        self.canonical_genes: list[str] = []
        self.expr: Optional[pd.DataFrame] = None   # genes × samples
        self.meta: Optional[pd.DataFrame] = None

    def _species_list(self) -> list[str]:
        if self.config.species == "both":
            return ["human", "mouse"]
        return [self.config.species]

    def _h5_path(self, species: str) -> str:
        fname = "human_gene_v2.5.h5" if species == "human" else "mouse_gene_v2.5.h5"
        return os.path.join(self.config.archs4_dir, fname)

    def process(self):
        """Run the full pipeline: determine gene set → extract → normalize."""
        t0 = time.time()
        cfg = self.config
        species_list = self._species_list()

        # --- Determine canonical gene list ---
        print("=" * 70)
        print(f"Gene set: {cfg.gene_set} | Species: {cfg.species}")
        print("=" * 70)

        if cfg.gene_set == "shared_orthologs":
            # Need to find zero-expression genes in each species
            all_ortho = self.registry.all_human_ortho
            human_zero = set()
            mouse_zero = set()

            if "human" in species_list:
                print("\nProbing human for zero-expression genes...")
                human_zero = self.loader.find_zero_genes(
                    self._h5_path("human"), "human", all_ortho
                )
                print(f"  {len(human_zero):,} genes with zero expression")

            if "mouse" in species_list:
                print("Probing mouse for zero-expression genes...")
                mouse_zero = self.loader.find_zero_genes(
                    self._h5_path("mouse"), "mouse", all_ortho
                )
                print(f"  {len(mouse_zero):,} genes with zero expression")

            self.canonical_genes = self.registry.get_canonical_genes(
                cfg.gene_set, human_zero, mouse_zero
            )
        else:
            self.canonical_genes = self.registry.get_canonical_genes(cfg.gene_set)

        print(f"\nCanonical genes: {len(self.canonical_genes):,}")

        # --- Extract all samples ---
        print(f"\n{'='*70}")
        print(f"Extracting {cfg.n_samples:,} samples per species")
        print(f"{'='*70}")

        all_exprs = []
        all_metas = []
        global_seen = set()

        for species in species_list:
            print(f"\n  [{species.upper()}]")
            expr, meta, global_seen = self.loader.extract_and_normalize(
                self._h5_path(species),
                species,
                cfg.n_samples,
                self.canonical_genes,
                global_seen,
            )
            if not expr.empty:
                all_exprs.append(expr)
                all_metas.append(meta)

        if all_exprs:
            self.expr = pd.concat(all_exprs, axis=1)
            self.meta = pd.concat(all_metas, ignore_index=True)
            print(f"\nTotal: {self.expr.shape[1]:,} samples × "
                  f"{self.expr.shape[0]:,} genes")

        elapsed = time.time() - t0
        print(f"Processing complete ({elapsed / 60:.1f} min)")

    def save_parquet(self, output_dir: Optional[str] = None):
        """
        Save processed data as a single parquet file + metadata.

        Output:
            {output_dir}/expression.parquet   (genes × samples, float32, zstd)
            {output_dir}/metadata.csv          (geo_accession, species)
            {output_dir}/canonical_genes.csv   (token_id, gene_symbol)

        Splitting into train/val/test is done later with scikit-learn.
        """
        out = Path(output_dir or self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Canonical gene list
        gene_df = pd.DataFrame({
            "token_id": range(1, len(self.canonical_genes) + 1),
            "gene_symbol": self.canonical_genes,
        })
        gene_df.to_csv(out / "canonical_genes.csv", index=False)

        with open(out / "genes.json", "w") as f:
            json.dump(self.canonical_genes, f)

        samples = self.meta["geo_accession"].tolist()
        with open(out / "samples.json", "w") as f:
            json.dump(samples, f)

        print(f"Saved canonical_genes.csv, genes.json ({len(self.canonical_genes):,} genes), "
              f"samples.json ({len(samples):,} samples)")

        # Expression matrix
        expr_path = out / "expression.parquet"
        self.expr.to_parquet(expr_path, compression="zstd")
        print(f"Saved {expr_path}: {self.expr.shape}")

        # Metadata
        meta_path = out / "metadata.csv"
        self.meta.to_csv(meta_path, index=False)
        print(f"Saved {meta_path}: {len(self.meta):,} rows")

        print("Done.")

    def get_data(self) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Get processed data as a numpy array.

        Returns:
            (X, metadata) where X is [samples, genes] float32 array
        """
        X = self.expr.values.T.astype(np.float32)  # [samples, genes]
        return X, self.meta


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RNA-seq preprocessing pipeline")
    parser.add_argument("--species", default="both", choices=["human", "mouse", "both"])
    parser.add_argument("--gene-set", default="shared_orthologs",
                        choices=["shared_orthologs", "union_orthologs"])
    parser.add_argument("--n-samples", type=int, default=120_000,
                        help="Samples per species")
    parser.add_argument("--output-dir", default="data/archs4/train_orthologs")
    parser.add_argument("--qc-min-nonzero", type=int, default=14_000)
    parser.add_argument("--normalization", default="log1p_tpm",
                        choices=["log1p_tpm", "raw_counts"])
    args = parser.parse_args()

    config = PreprocessingConfig(
        species=args.species,
        gene_set=args.gene_set,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        qc_min_nonzero=args.qc_min_nonzero,
        normalization=args.normalization,
    )

    builder = RNADatasetBuilder(config=config)
    builder.process()
    builder.save_parquet()
