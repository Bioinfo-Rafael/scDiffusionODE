"""Read-only barcode/expression matching and saved-embedding diagnostics.

No AnnData mutation, matrix normalization, or embedding computation. Matching
uses only barcode and expression; metadata is read after matching has finished.
"""
from collections import defaultdict
import importlib.metadata
import inspect
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import sparse
from IPython.display import display

LINEAGE = ["Blood progenitors 1", "Blood progenitors 2", "Erythroid1",
           "Erythroid2", "Erythroid3"]
MATRICES = ("X", "spliced", "unspliced")
SEED = 20261007


def matrix(adata, name):
    return adata.X if name == "X" else adata.layers.get(name)


def _save(table, directory, name, index=False):
    directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(directory / (name + ".csv"), index=index)


def barcode_candidates(full, ery, reports):
    """One record per Ery row; no suffix stripping or annotation matching."""
    if "barcode" not in full.obs:
        raise ValueError("Full.obs['barcode'] is required for this analysis")
    barcodes = full.obs["barcode"].astype("string").reset_index(drop=True)
    groups = defaultdict(list)
    for pos, barcode in enumerate(barcodes):
        if pd.notna(barcode):
            groups[str(barcode)].append(pos)
    rows, edges = [], []
    for pos, barcode in enumerate(ery.obs_names):
        candidates = groups.get(str(barcode), []) if pd.notna(barcode) else []
        rows.append({"ery_pos": pos, "ery_id": barcode,
                     "barcode_candidates": len(candidates)})
        edges.extend((pos, candidate) for candidate in candidates)
    cells = pd.DataFrame(rows, columns=["ery_pos", "ery_id", "barcode_candidates"])
    pairs = pd.DataFrame(edges, columns=["ery_pos", "full_pos"], dtype=int)
    counts = barcodes.value_counts()
    duplicated = counts[counts > 1]
    summary = pd.Series({
        "ery_cells": ery.n_obs,
        "barcode_matched_ery": int((cells.barcode_candidates > 0).sum()),
        "barcode_match_rate": float((cells.barcode_candidates > 0).mean()) if ery.n_obs else np.nan,
        "unique_barcode_candidate_ery": int((cells.barcode_candidates == 1).sum()),
        "duplicate_barcode_candidate_ery": int((cells.barcode_candidates > 1).sum()),
        "no_barcode_candidate_ery": int((cells.barcode_candidates == 0).sum()),
        "full_duplicate_barcode_values": len(duplicated),
        "full_cells_with_duplicate_barcode": int(duplicated.sum()),
        "full_missing_barcode": int(barcodes.isna().sum()),
        "ery_duplicate_index_extra_rows": int(ery.obs_names.duplicated().sum()),
    })
    display(summary)
    duplicate_rows = pd.DataFrame({"full_pos": np.arange(full.n_obs), "barcode": barcodes})
    duplicate_rows = duplicate_rows[duplicate_rows.barcode.isin(duplicated.index)].copy()
    duplicate_rows["full_id"] = full.obs_names.take(duplicate_rows.full_pos).to_numpy()
    # sample is descriptive only; it is never passed to the expression matcher.
    if "sample" in full.obs:
        duplicate_rows["sample"] = full.obs.iloc[duplicate_rows.full_pos]["sample"].astype("string").to_numpy()
        duplicate_rows["sample"] = duplicate_rows["sample"].fillna("<missing>")
        distribution = duplicate_rows.groupby(["barcode", "sample"], observed=True).size().rename("cells").reset_index()
        display(duplicate_rows.groupby("sample", observed=True).agg(
            cells=("barcode", "size"), distinct_barcodes=("barcode", "nunique")))
        sample_counts = duplicate_rows.groupby("barcode", observed=True)["sample"].nunique()
        print("Duplicated barcodes by number of distinct samples:")
        display(sample_counts.value_counts().sort_index().rename("barcodes"))
        print("Same-sample duplicate barcode groups:", int((distribution.cells > 1).sum()))
        display(distribution.head(40))
        _save(distribution, reports, "duplicate_barcode_sample_distribution")
    else:
        print("sample column absent; cross-sample distribution unavailable")
    _save(duplicate_rows, reports, "duplicate_barcode_full_cells")
    _save(cells, reports, "barcode_candidate_counts")
    return {"cells": cells, "pairs": pairs, "summary": summary}


def _aligned_genes(full, ery):
    # Preserve exact IDs and exclude all occurrences of ambiguous gene names.
    f = full.var_names[~full.var_names.duplicated(keep=False)]
    e = ery.var_names[~ery.var_names.duplicated(keep=False)]
    genes = f.intersection(e, sort=False)
    fm = {g: i for i, g in enumerate(full.var_names)}
    em = {g: i for i, g in enumerate(ery.var_names)}
    fi = np.array([fm[g] for g in genes], dtype=int)
    ei = np.array([em[g] for g in genes], dtype=int)
    assert full.var_names.take(fi).equals(ery.var_names.take(ei))
    return genes, fi, ei


def _block(mat, rows, cols):
    """Bounded row batch; convert only a dense input's small block to sparse."""
    if sparse.issparse(mat):
        block = mat[rows, :][:, cols].astype(np.float64, copy=True).tocsr()
    else:
        block = sparse.csr_matrix(np.asarray(mat[np.ix_(rows, cols)], dtype=np.float64))
    block.sum_duplicates()
    block.eliminate_zeros()
    return block


def _row_comparison(a, b):
    """Sparse exact comparison; any NaN/Inf invalidates its pair, even if equal."""
    bad_a, bad_b = a.copy(), b.copy()
    bad_a.data = (~np.isfinite(a.data)).astype(np.int8)
    bad_b.data = (~np.isfinite(b.data)).astype(np.int8)
    invalid = (np.asarray(bad_a.sum(axis=1)).ravel() +
               np.asarray(bad_b.sum(axis=1)).ravel()) > 0
    with np.errstate(invalid="ignore", over="ignore"):
        delta = a - b
    delta.eliminate_zeros()
    equal = (np.diff(delta.indptr) == 0) & ~invalid
    nonzero_evidence = (np.diff(a.indptr) + np.diff(b.indptr)) > 0
    return equal, invalid, nonzero_evidence


def match_expression(full, ery, candidate_info, reports, batch_size=32):
    """Refine each matrix independently on 256, 2048, then ALL shared genes.

    The panels are disjoint, so surviving edges have every gene checked exactly
    once per matrix. One surviving candidate is NOT enough to stop early.
    Rejected edges report only genes actually checked before rejection.
    """
    genes, fi, ei = _aligned_genes(full, ery)
    pairs = candidate_info["pairs"].copy()
    cells = candidate_info["cells"].copy()
    available = [k for k in MATRICES if matrix(full, k) is not None and matrix(ery, k) is not None]
    print("Shared matrices:", available, "; missing:", sorted(set(MATRICES) - set(available)))
    print("Unique shared genes:", len(genes), "; ambiguous or nonshared genes excluded:",
          full.n_vars - len(genes), ery.n_vars - len(genes))
    _save(pd.DataFrame({"gene": genes, "full_var_pos": fi, "ery_var_pos": ei}), reports, "matched_gene_order")
    tested = bool(available) and len(genes) > 0
    panel_order = np.random.default_rng(SEED).permutation(len(genes))
    ends = sorted(set(min(n, len(genes)) for n in [256, 2048, len(genes)]))
    steps, per_matrix = [], []
    for key in available if tested else []:
        alive = np.ones(len(pairs), dtype=bool)
        checked = np.zeros(len(pairs), dtype=int)
        invalid = np.zeros(len(pairs), dtype=bool)
        evidence = np.zeros(len(pairs), dtype=bool)
        start = 0
        for end in ends:
            panel = panel_order[start:end]
            active = np.flatnonzero(alive)
            for offset in range(0, len(active), batch_size):
                ix = active[offset:offset + batch_size]
                fr = pairs.iloc[ix].full_pos.to_numpy(dtype=int)
                er = pairs.iloc[ix].ery_pos.to_numpy(dtype=int)
                a = _block(matrix(full, key), fr, fi[panel])
                b = _block(matrix(ery, key), er, ei[panel])
                equal, bad, nz = _row_comparison(a, b)
                alive[ix] &= equal
                invalid[ix] |= bad
                evidence[ix] |= nz
                checked[ix] += len(panel)
            steps.append({"matrix": key, "cumulative_genes": end,
                          "candidate_pairs_remaining": int(alive.sum()),
                          "ery_cells_remaining": pairs.loc[alive, "ery_pos"].nunique()})
            print(key, "genes:", end, "surviving candidate pairs:", int(alive.sum()), flush=True)
            start = end
        pairs[key + "_exact"] = alive
        pairs[key + "_genes_checked"] = checked
        pairs[key + "_invalid"] = invalid
        pairs[key + "_nonzero_evidence"] = evidence
        nmatch = pairs.loc[alive].groupby("ery_pos").size().reindex(range(ery.n_obs), fill_value=0)
        per_matrix.append({"matrix": key, "genes_for_exact_matches": len(genes),
                           "ery_with_exact_candidate": int((nmatch > 0).sum()),
                           "ery_one_exact_candidate": int((nmatch == 1).sum()),
                           "ery_multiple_exact_candidates": int((nmatch > 1).sum()),
                           "ery_candidates_but_no_exact": int(((nmatch.to_numpy() == 0) &
                               (cells.barcode_candidates.to_numpy() > 0)).sum())})
    pairs["joint_exact"] = (pairs[[k + "_exact" for k in available]].all(axis=1)
                            if tested else False)
    pairs["nonzero_evidence"] = (pairs[[k + "_nonzero_evidence" for k in available]].any(axis=1)
                                 if tested else False)
    joint = pairs[pairs.joint_exact]
    exact_counts = joint.groupby("ery_pos").size().reindex(range(ery.n_obs), fill_value=0)
    cells["exact_candidates"] = exact_counts.to_numpy()
    cells["status"] = "no_barcode_candidate"
    has = cells.barcode_candidates > 0
    cells.loc[has, "status"] = "expression_mismatch" if tested else "not_tested"
    cells.loc[cells.exact_candidates > 1, "status"] = "multiple_exact_candidates"
    unique = joint[joint.ery_pos.isin(cells.loc[cells.exact_candidates == 1, "ery_pos"])].copy()
    # Do not assign the same Full row to multiple Ery rows, even for duplicate Ery IDs.
    collision = unique.full_pos.duplicated(keep=False)
    zero_only = ~unique.nonzero_evidence
    cells.loc[unique.ery_pos, "status"] = "unique_exact"
    cells.loc[unique.loc[zero_only, "ery_pos"], "status"] = "zero_only_uninformative"
    cells.loc[unique.loc[collision, "ery_pos"], "status"] = "many_to_one_conflict"
    resolved = unique.loc[~collision & ~zero_only, ["ery_pos", "full_pos"]].copy()
    resolved["ery_id"] = ery.obs_names.take(resolved.ery_pos).to_numpy()
    resolved["full_id"] = full.obs_names.take(resolved.full_pos).to_numpy()
    resolved["genes_verified"] = len(genes)
    resolved["matrices_verified"] = ",".join(available)
    for key in available:
        if tested:
            assert (pairs.loc[pairs[key + "_exact"], key + "_genes_checked"] == len(genes)).all()
    summary = {"verification_performed": tested, "matrices_verified": available,
               "genes_verified_for_exact_matches": len(genes) if tested else 0,
               "exact_expression_cells": int((cells.exact_candidates > 0).sum()) if tested else None,
               "unique_resolved_cells": len(resolved) if tested else None,
               "multiple_exact_cells": int((cells.exact_candidates > 1).sum()) if tested else None,
               "expression_mismatch_cells": int((cells.status == "expression_mismatch").sum()) if tested else None}
    display(pd.DataFrame(per_matrix))
    display(pd.Series(summary))
    display(cells.status.value_counts())
    display(cells[cells.status != "unique_exact"].head(30))
    _save(pairs, reports, "expression_candidate_pairs")
    _save(cells, reports, "cell_resolution_status")
    _save(resolved, reports, "expression_verified_cell_map")
    _save(pd.DataFrame(steps), reports, "expression_matching_stages")
    return {"cells": cells, "pairs": pairs, "resolved": resolved, "summary": summary}


def lineage_counts(full, ery, reports):
    if "celltype" not in full.obs or "celltype" not in ery.obs:
        print("celltype absent; lineage count comparison unavailable")
        return {"full_lineage_count": None}
    selected = full.obs.celltype.astype("string").isin(LINEAGE)
    counts = pd.DataFrame({
        "full_lineage": full.obs.loc[selected, "celltype"].value_counts().reindex(LINEAGE, fill_value=0),
        "ery": ery.obs.celltype.value_counts().reindex(LINEAGE, fill_value=0)})
    counts["difference"] = counts.full_lineage - counts.ery
    counts["equal"] = counts["difference"] == 0
    display(counts)
    n = int(selected.sum())
    print("Full five-celltype count:", n, "; Ery:", ery.n_obs,
          "; same count:", n == ery.n_obs, "; equals 9815:", n == 9815)
    print("Ery outside the five celltypes (including missing):",
          int((~ery.obs.celltype.astype("string").isin(LINEAGE)).sum()))
    _save(counts, reports, "five_celltype_counts", index=True)
    if "stage" in full.obs and "stage" in ery.obs:
        def tab(frame):
            return pd.crosstab(frame.stage.astype("string").fillna("<missing>"),
                               frame.celltype.astype("string").fillna("<missing>"))
        a, b = tab(full.obs.loc[selected]), tab(ery.obs)
        idx, cols = a.index.union(b.index), a.columns.union(b.columns)
        a, b = (t.reindex(index=idx, columns=cols, fill_value=0) for t in (a, b))
        print("Full lineage stage × celltype"); display(a)
        print("Ery stage × celltype"); display(b)
        print("Difference (Full lineage minus Ery)"); display(a-b)
        print("Entire stage × celltype table equal:", a.equals(b))
        _save(a-b, reports, "stage_celltype_count_difference", index=True)
    return {"full_lineage_count": n, "counts": counts}


def _heatmap(table, title, path):
    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, .55*len(table.columns)), max(4, .4*len(table))))
    im = ax.imshow(table.to_numpy(), aspect="auto", cmap="Blues")
    ax.set_xticks(np.arange(len(table.columns)), table.columns, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(table)), table.index)
    ax.set(xlabel="Ery", ylabel="Full", title=title)
    if table.size <= 500:
        for (r, c), v in np.ndenumerate(table.to_numpy()):
            ax.text(c, r, str(v), ha="center", va="center", fontsize=8,
                    color="white" if v > table.to_numpy().max()/2 else "black")
    fig.colorbar(im, ax=ax, label="Cells")
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.show(); plt.close(fig)


def annotations_after_matching(full, ery, matched, figures, reports):
    mapping = matched["resolved"]
    if mapping.empty:
        print("No expression-verified one-to-one cells; annotation comparison skipped")
        return pd.DataFrame()
    a = full.obs.iloc[mapping.full_pos].reset_index(drop=True)
    b = ery.obs.iloc[mapping.ery_pos].reset_index(drop=True)
    records = []
    for col in ["celltype", "stage"]:
        if col not in a or col not in b:
            print(col, "absent; skipped")
            continue
        x, y = a[col].astype("string"), b[col].astype("string")
        valid = x.notna() & y.notna()
        equal = x.eq(y).fillna(False) & valid
        mismatch = (valid & ~equal) | (x.isna() ^ y.isna())
        records.append({"column": col, "matched_cells": len(mapping), "comparable": int(valid.sum()),
                        "matches": int(equal.sum()), "mismatches": int((valid & ~equal).sum()),
                        "one_missing": int((x.isna() ^ y.isna()).sum()),
                        "both_missing": int((x.isna() & y.isna()).sum()),
                        "agreement": float(equal.sum()/valid.sum()) if valid.any() else np.nan})
        details = mapping.reset_index(drop=True).copy()
        details["full_value"], details["ery_value"] = x, y
        details = details.loc[mismatch]
        print(col, "disagreement list (full list also saved to CSV):")
        with pd.option_context("display.max_rows", None):
            display(details)
        _save(details, reports, col + "_disagreements")
    for fcol, ecol in [("celltype", "celltype"), ("stage", "stage"),
                       ("cluster.sub", "celltype"), ("haem_subclust", "celltype")]:
        if fcol not in a or ecol not in b:
            print(fcol, ecol, "absent; crosstab skipped")
            continue
        table = pd.crosstab(a[fcol].astype("string").fillna("<missing>"),
                            b[ecol].astype("string").fillna("<missing>"))
        name = "matched_" + fcol.replace(".", "_") + "_vs_" + ecol
        display(table)
        _save(table, reports, name, index=True)
        _heatmap(table, "Full " + fcol + " vs Ery " + ecol, figures / (name + ".png"))
    result = pd.DataFrame(records).set_index("column") if records else pd.DataFrame()
    display(result)
    _save(result, reports, "matched_annotation_agreement", index=True)
    return result


def saved_embedding(adata, name, basis, column, figures, colors, lineage_only=False):
    key = "X_" + basis
    if key not in adata.obsm or column not in adata.obs:
        print(name, key, column, "missing; plot skipped")
        return
    coords = adata.obsm[key]
    if coords.ndim != 2 or coords.shape[1] < 2:
        print(name, key, "has fewer than two dimensions; skipped")
        return
    # Embeddings are small; only their first two columns are materialized.
    xy = coords[:, :2].toarray() if sparse.issparse(coords) else np.asarray(coords[:, :2])
    values = adata.obs[column].astype("string").reset_index(drop=True)
    valid = np.isfinite(xy).all(axis=1)
    categories = sorted(values.dropna().unique())
    if lineage_only:
        categories = [c for c in LINEAGE if c in categories]
    height = max(5, .23*(len(categories)+2))
    fig, ax = plt.subplots(figsize=(10, height))
    handles = []
    other = valid & ~values.isin(categories).to_numpy()
    if other.any():
        ax.scatter(xy[other, 0], xy[other, 1], c="#cccccc", s=2, alpha=.5, rasterized=True)
        handles.append(Line2D([], [], marker="o", linestyle="", color="#cccccc",
                              label="Other / missing" if lineage_only else "Missing"))
    # Fixed shuffle prevents an input-order pattern within each category.
    order = np.random.default_rng(SEED).permutation(adata.n_obs)
    for category in categories:
        mask = valid & values.eq(category).fillna(False).to_numpy(dtype=bool)
        ix = order[mask[order]]
        color = colors[category]
        ax.scatter(xy[ix, 0], xy[ix, 1], c=[color], s=3, alpha=.65, rasterized=True)
        handles.append(Line2D([], [], marker="o", linestyle="", color=color, label=category))
    labels = ("PC1", "PC2") if basis == "pca" else ("UMAP1", "UMAP2")
    ax.set(xlabel=labels[0], ylabel=labels[1], title=f"{name}: saved {basis.upper()} / {column}")
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, .5), fontsize=8,
              title=column, frameon=False)
    suffix = "erythroid_lineage" if lineage_only else column
    path = figures / f"{name}_saved_{basis}_{suffix}.png"
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.show(); plt.close(fig)
    print(path.name, "; cells with invalid coordinates:", int((~valid).sum()))


def plot_saved_embeddings(full, ery, figures):
    figures.mkdir(parents=True, exist_ok=True)
    for column in ["celltype", "stage", "haem_subclust"]:
        categories = sorted(set().union(*[
            set(a.obs[column].astype("string").dropna()) for a in [full, ery] if column in a.obs]))
        palette = [plt.get_cmap(c)(i) for c in ["tab20", "tab20b", "tab20c"] for i in range(20)]
        colors = {c: palette[i % len(palette)] for i, c in enumerate(categories)}
        if column == "haem_subclust":
            saved_embedding(full, "full", "umap", column, figures, colors)
        else:
            for name, adata in [("full", full), ("ery", ery)]:
                for basis in ["pca", "umap"]:
                    saved_embedding(adata, name, basis, column, figures, colors)
            if column == "celltype":
                saved_embedding(full, "full", "umap", column, figures, colors, lineage_only=True)


def exact_matrix_difference(a, b, block_rows=32):
    """Exhaustive element comparison via bounded sparse differences (no sampling).

    MAE denominator includes implicit zeros. Any nonfinite input marks equality
    unverified and MAE/max error NaN; nonfinite locations are counted separately.
    """
    if a.shape != b.shape:
        return {"status": "shape_mismatch", "shape_a": a.shape, "shape_b": b.shape}
    nrows, ncols = a.shape
    total = nrows*ncols
    different = bad_count = nonfinite_difference = 0
    max_error = sum_error = 0.
    cols = np.arange(ncols)
    for start in range(0, nrows, block_rows):
        rows = np.arange(start, min(start+block_rows, nrows))
        x, y = _block(a, rows, cols), _block(b, rows, cols)
        bad_x, bad_y = x.copy(), y.copy()
        bad_x.data = (~np.isfinite(x.data)).astype(np.int8)
        bad_y.data = (~np.isfinite(y.data)).astype(np.int8)
        bad_union = bad_x + bad_y
        bad_union.eliminate_zeros()
        bad_count += bad_union.nnz
        with np.errstate(invalid="ignore", over="ignore"):
            delta = x-y
        delta.eliminate_zeros()
        finite = np.isfinite(delta.data)
        nonfinite_difference += int((~finite).sum())
        values = np.abs(delta.data[finite])
        different += int((values != 0).sum())
        if values.size:
            sum_error += values.sum(dtype=np.float64)
            max_error = max(max_error, values.max())
    # Finite inputs can still overflow on subtraction; never report those equal.
    overflow = nonfinite_difference if not bad_count else 0
    return {"status": "nonfinite_input" if bad_count else "checked_all_elements",
            "elements_checked": total, "finite_different_elements": different,
            "nonfinite_locations": bad_count,
            "different_elements": different + overflow if not bad_count else None,
            "max_absolute_error": (np.inf if overflow else max_error) if not bad_count and total else np.nan,
            "mean_absolute_error": (np.inf if overflow else sum_error/total) if total and not bad_count else np.nan,
            "exact_equal": (different + overflow == 0) if total and not bad_count else None}


def compare_internal_matrices(full, ery, reports):
    records = []
    for name, adata, left, right in [("full", full, "X", "spliced"),
                                    ("ery", ery, "X", "spliced"),
                                    ("ery", ery, "X", "raw_counts"),
                                    ("ery", ery, "spliced", "raw_counts")]:
        a, b = matrix(adata, left), matrix(adata, right)
        print("Checking all elements:", name, left, right, flush=True)
        result = {"status": "matrix_absent"} if a is None or b is None else exact_matrix_difference(a, b)
        records.append({"dataset": name, "left": left, "right": right, **result})
    table = pd.DataFrame(records)
    display(table)
    _save(table, reports, "internal_matrix_equality")
    return table


def preprocessing_evidence(full, ery, diagnostics, equality, reports):
    """Observed cached diagnostics vs interpretations, without categorical claims."""
    rows = []
    for name, adata in [("full", full), ("ery", ery)]:
        entries = diagnostics.get(name, {})
        for key in ["X", "spliced", "unspliced", "raw_counts"]:
            if key not in entries:
                continue
            d = entries[key]
            s = d["summary"]
            values = d["library"]
            finite = values[np.isfinite(values)]
            cv = finite.std()/abs(finite.mean()) if len(finite) and finite.mean() else np.nan
            count_like = s.get("integer_like_fraction") == 1 and s.get("negative_count") == 0 and s.get("nonfinite_count") == 0
            rows.append({"dataset": name, "topic": key + " integer/count evidence",
                         "observed": str({k: s.get(k) for k in ["min", "max", "integer_like_fraction", "nonzero_sample_n", "nonzero_fraction_finite", "negative_count", "nonfinite_count"]}),
                         "interpretation": "Sample and range support nonnegative integer counts; not proof of original raw counts." if count_like else "Count status inconclusive; inspect values.",
                         "unknown": "Nonzero integer fraction is sampled, not an exhaustive integer test."})
            rows.append({"dataset": name, "topic": key + " library-size normalization",
                         "observed": str({"mean": finite.mean() if len(finite) else None,
                                          "std": finite.std() if len(finite) else None,
                                          "min": finite.min() if len(finite) else None,
                                          "max": finite.max() if len(finite) else None, "CV": cv}),
                         "interpretation": "Totals vary; no common fixed library total observed." if np.isfinite(cv) and cv > 1e-6 else "Totals nearly fixed or unavailable; inspect normalization history.",
                         "unknown": "Variable totals do not exclude every normalization procedure."})
        x_summary = entries.get("X", {}).get("summary", {})
        unlogged_hint = (x_summary.get("integer_like_fraction") == 1 and
                         x_summary.get("negative_count") == 0)
        rows.extend([
            {"dataset": name, "topic": "log1p", "observed": repr(adata.uns.get("log1p", "absent")),
             "interpretation": "Integer-like X samples support an unlogged count representation." if unlogged_hint else "X sample does not establish an unlogged count representation.",
             "unknown": "Missing log1p metadata alone does not prove no historical log transform."},
            {"dataset": name, "topic": "gene filtering / HVG subset",
             "observed": f"n_vars={adata.n_vars}; var columns={adata.var.columns.tolist()}",
             "interpretation": "Not a 1000–3000-gene HVG subset." if adata.n_vars > 3000 else "Gene count alone cannot identify HVG subsetting.",
             "unknown": "Original gene universe and upstream gene filtering; PCA may use selected genes without subsetting AnnData."},
            {"dataset": name, "topic": "saved embeddings",
             "observed": str({k: adata.obsm[k].shape for k in ["X_pca", "X_umap"] if k in adata.obsm}),
             "interpretation": "Embeddings already exist where listed.",
             "unknown": "Cannot conclude only PCA/UMAP was done; their input matrix and preprocessing history are unknown."},
        ])
        if "highly_variable" in adata.var:
            rows.append({"dataset": name, "topic": "HVG flags", "observed": str(adata.var.highly_variable.value_counts(dropna=False).to_dict()),
                         "interpretation": "HVG annotation is present.", "unknown": "Whether/where those genes were used."})
    rows.append({"dataset": "full vs ery", "topic": "gene identity/order",
                 "observed": str(full.var_names.equals(ery.var_names)),
                 "interpretation": "Equal IDs/order supports the same retained gene universe when True.",
                 "unknown": "Does not establish equality to the original assay's unfiltered genes."})
    result = pd.DataFrame(rows)
    with pd.option_context("display.max_colwidth", None, "display.max_rows", None):
        display(result)
    print("Exhaustive within-dataset equality evidence:")
    display(equality)
    _save(result, reports, "preprocessing_evidence")
    return result


def provenance(data_dir, full):
    """Read local metadata and loader source only; never call the dataset loader."""
    import scvelo as scv
    print("Current kernel packages:", {p: importlib.metadata.version(p) for p in
          ["anndata", "scanpy", "scvelo", "numpy", "scipy", "pandas", "matplotlib"]})
    path = data_dir / "full_source.json"
    print("Download-time provenance:", json.loads(path.read_text()) if path.exists() else "full_source.json absent")
    print("Installed scVelo loader source (not executed):")
    try:
        print(inspect.getsource(scv.datasets.gastrulation))
    except (OSError, TypeError) as exc:
        print("Source unavailable:", exc)
    # Read only dataset dimensions directly; do not load a second AnnData into RAM.
    cache = data_dir / "scvelo_cache" / "gastrulation.h5ad"
    if cache.exists():
        import h5py
        with h5py.File(cache, "r") as f:
            obs_key = f["obs"].attrs.get("_index", "_index")
            var_key = f["var"].attrs.get("_index", "_index")
            shape = (len(f["obs"][obs_key]), len(f["var"][var_key]))
        print("Original scVelo cache shape:", shape, "; saved Full shape:", full.shape)
    else:
        print("Original cache absent; pre-save shape cannot be checked locally.")
    print("Paper atlas: 116312; observed Full:", full.n_obs, "; difference:", 116312-full.n_obs)
    print("v0.3.3/v0.3.4 loader reads Figshare file 28095525 and makes gene names unique; no explicit cell filter.")
    print("The precise upstream reason for the cell-count difference is unverified. Do not label it QC/doublet removal without evidence.")


def final_summary(full, ery, barcode, matched, lineage, annotation, equality, diagnostics):
    def rate(col):
        return annotation.loc[col, "agreement"] if col in annotation.index else "not available"
    comparison = equality[(equality.left == "X") & (equality.right == "spliced")]
    evidence = []
    for name, entries in diagnostics.items():
        if "X" in entries:
            s = entries["X"]["summary"]
            library = entries["X"]["library"]
            finite_library = library[np.isfinite(library)]
            span = (finite_library.min(), finite_library.max()) if len(finite_library) else None
            evidence.append(f"{name}: sampled integer fraction={s.get('integer_like_fraction')}, "
                            f"min={s.get('min')}, negative count={s.get('negative_count')}, "
                            f"library size range={span}")
    raw_comparison = equality[(equality.dataset == "ery") & (equality.right == "raw_counts")]
    evidence.append("Ery raw_counts exhaustive comparisons: " + str(raw_comparison.to_dict("records")))
    rows = [
        ("Full cell count", full.n_obs), ("Erythroid cell count", ery.n_obs),
        ("Fullの5 celltype抽出数", lineage["full_lineage_count"]),
        ("Barcode一致率", barcode["summary"]["barcode_match_rate"]),
        ("一意に対応できた細胞数", matched["summary"]["unique_resolved_cells"]),
        ("発現量完全一致の細胞数（候補あり・曖昧例を含む）", matched["summary"]["exact_expression_cells"]),
        ("検証した共通gene数", matched["summary"]["genes_verified_for_exact_matches"]),
        ("検証matrix", matched["summary"]["matrices_verified"]),
        ("celltype一致率（双方nonmissing）", rate("celltype")),
        ("stage一致率（双方nonmissing）", rate("stage")),
        ("Xとsplicedの一致", comparison.to_dict("records")),
        ("raw countsと判断できる根拠（推定材料）", "; ".join(evidence) or "not available"),
        ("未確認", "raw由来と前処理履歴は未確定。全要素matrix一致は上表参照。annotationは未採用。"),
    ]
    return pd.DataFrame(rows, columns=["項目", "結果"])
