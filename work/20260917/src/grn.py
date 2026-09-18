"""Same TSV columns/filter as build_edge_mask; explicit target/source mapping."""
import pandas as pd
import torch


def load_grn(genes, path):
    genes = list(map(str, genes))
    if len(set(genes)) != len(genes):
        raise ValueError("duplicate dataset gene names make GRN mapping ambiguous")
    # Source loader: ODE.ode_20260609_mathmlp.build_edge_mask reads from/to.
    table = pd.read_csv(path, sep="\t", usecols=["from", "to"], dtype=str)
    if table.isna().any().any():
        raise ValueError("GRN has missing gene names")
    mapping = {g: i for i, g in enumerate(genes)}
    valid = table["from"].isin(mapping) & table["to"].isin(mapping)
    kept = table.loc[valid].drop_duplicates(["from", "to"])
    if kept.empty:
        raise ValueError("no GRN edges match dataset genes; check symbol/Ensembl convention")
    src = torch.tensor([mapping[g] for g in kept["from"]], dtype=torch.long)
    dst = torch.tensor([mapping[g] for g in kept["to"]], dtype=torch.long)
    grn_genes = set(table["from"]) | set(table["to"])
    report = dict(orientation="source=from -> target=to; W[target,source]",
                  gene_to_column=mapping, input_edges=len(table), retained_edges=len(kept),
                  dropped_unmapped_edges=int((~valid).sum()),
                  duplicate_mapped_edges=int(valid.sum())-len(kept),
                  grn_genes_absent_from_data=sorted(grn_genes-set(genes)),
                  data_genes_absent_from_grn=sorted(set(genes)-grn_genes))
    return src, dst, report
