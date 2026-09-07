#!/usr/bin/env python3
"""Create only real/generated UMAP without changing raw-count training data."""

from __future__ import annotations
import argparse, json, sys
from datetime import datetime
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from scipy import sparse

HERE=Path(__file__).resolve().parent; SUITE_ROOT=HERE.parent; REPO_ROOT=SUITE_ROOT.parent.parent
for path in (SUITE_ROOT,HERE,REPO_ROOT): sys.path.insert(0,str(path))
from common import read_json, update_stage, validate_run_dir, write_json
# This file is named umap.py.  Keep its directory off the later import search
# path so Scanpy can import the external ``umap`` package instead of this file.
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != HERE]

def main():
    p=argparse.ArgumentParser(allow_abbrev=False); p.add_argument("--run-dir",required=True); p.add_argument("--dry-run",action="store_true")
    args=p.parse_args(); run=validate_run_dir(args.run_dir); config=read_json(run/"exp_config.json"); manifest=read_json(run/"manifest.json")
    sample=Path(manifest.get("sample_path","")); output=run/"umap"/f"{Path(manifest['checkpoint_path']).stem}_{datetime.now():%Y%m%d_%H%M%S}"
    if args.dry_run: print(json.dumps({"run":str(run),"sample":str(sample),"output":str(output)},indent=2)); return
    if not sample.is_file(): raise FileNotFoundError(sample)
    adata=sc.read_h5ad(config["data_dir"],backed="r"); genes=adata.var["gene_name"].astype(str).tolist()
    with np.load(sample,allow_pickle=False) as z: generated=np.asarray(z["cell_gen"],dtype=np.float32)
    if generated.shape[1]!=len(genes) or not np.isfinite(generated).all(): raise ValueError("invalid generated data")
    seed=int(config.get("umap_seed",1234)); n=min(int(config.get("umap_real_cells",3000)),adata.n_obs)
    indices=np.sort(np.random.default_rng(seed).choice(adata.n_obs,n,replace=False)); real=adata.X[indices]
    if sparse.issparse(real): real=real.toarray()
    real=np.asarray(real,dtype=np.float32)
    if not np.isfinite(real).all(): raise FloatingPointError("real data non-finite")
    combined=np.concatenate([real,generated]); labels=np.array(["real"]*len(real)+["generated"]*len(generated))
    # Deliberately no normalize/log/clip: both sources stay in the raw-count model space.
    view=sc.AnnData(combined); view.obs["source"]=labels
    n_comps=min(50,view.n_vars-1,view.n_obs-1); sc.pp.pca(view,n_comps=n_comps,random_state=seed)
    n_pcs=min(int(config.get("umap_n_pcs",40)),n_comps); sc.pp.neighbors(view,n_neighbors=int(config.get("umap_n_neighbors",15)),n_pcs=n_pcs,random_state=seed); sc.tl.umap(view,random_state=seed)
    output.mkdir(parents=True,exist_ok=False); emb=view.obsm["X_umap"]
    fig,ax=plt.subplots(figsize=(8,7))
    for label,color in (("real","#2563eb"),("generated","#dc2626")):
        mask=labels==label; ax.scatter(emb[mask,0],emb[mask,1],s=4,alpha=.45,c=color,label=label,rasterized=True)
    ax.legend(); ax.set_title(f"{config['experiment']} | step={manifest['checkpoint_step']} | raw-count training")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2"); fig.tight_layout(); figure=output/"real_vs_generated_umap.png"; fig.savefig(figure,dpi=320); plt.close(fig)
    np.savetxt(output/"embedding.csv",np.column_stack([emb,labels]),delimiter=",",fmt="%s",header="UMAP1,UMAP2,source",comments="")
    metadata={"experiment":config["experiment"],"checkpoint":manifest["checkpoint_path"],"sample":str(sample),"real_data":config["data_dir"],"real_indices":indices.tolist(),"visualization_transform":"none", "negative_generated_count":int(np.count_nonzero(generated<0)),"pca_components":n_comps,"neighbors":int(config.get("umap_n_neighbors",15)),"n_pcs":n_pcs,"seed":seed,"figure":str(figure)}
    write_json(output/"metadata.json",metadata); update_stage(run,"umap","completed",output_dir=str(output),figure=str(figure)); print(f"UMAP_DIR='{output}'")
    adata.file.close()
if __name__=="__main__": main()
