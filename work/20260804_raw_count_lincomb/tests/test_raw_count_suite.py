#!/usr/bin/env python3
"""Synthetic data and six-model checks without downloading or long training."""

from __future__ import annotations
import json, sys, tempfile, unittest
from pathlib import Path
from unittest import mock
import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SUITE=Path(__file__).resolve().parents[1]; REPO=SUITE.parent.parent
for path in (SUITE,SUITE/"scripts",REPO): sys.path.insert(0,str(path))
from scripts import create_raw_count_data
from scripts.create_raw_count_data import OUTPUT_NAME, build, matrix_audit
from models import factory
from guided_diffusion.cell_datasets_loader import load_data
from guided_diffusion.script_util import create_gaussian_diffusion

class TinyCell(nn.Module):
    def __init__(self,input_dim=2,**kwargs): super().__init__(); self.hidden_num=[8]*4; self.layer=nn.Linear(input_dim,input_dim)
    def forward(self,x,t,y=None): return self.layer(x.float())

class RawCountSuiteTests(unittest.TestCase):
    def test_missing_gdown_is_bootstrapped_in_active_python(self):
        missing=ModuleNotFoundError("No module named 'gdown'",name="gdown"); fake=object()
        with mock.patch.object(create_raw_count_data.importlib,"import_module",side_effect=[missing,fake]) as importer, mock.patch.object(create_raw_count_data.subprocess,"run") as runner:
            self.assertIs(create_raw_count_data.load_gdown(),fake)
        runner.assert_called_once_with([sys.executable,"-m","pip","install","gdown==5.2.0"],check=True)
        self.assertEqual(importer.call_count,2)

    def test_raw_counts_survive_hvg_only_normalize_log(self):
        rng=np.random.default_rng(4); x=rng.poisson(2,size=(40,1100)).astype(np.float32)
        obs=pd.DataFrame({"a":range(40),"original_celltype":["A","B"]*20,"c":0},index=[f"c{i}" for i in range(40)])
        var=pd.DataFrame({"a":0,"b":0,"original_gene":[f"g{i}" for i in range(1100)]},index=[f"v{i}" for i in range(1100)])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/"Embryonic_raw_data.h5ad"; output=root/OUTPUT_NAME
            ad.AnnData(x,obs=obs,var=var).write_h5ad(source)
            metadata=build(source,output,root/"metadata")
            result=ad.read_h5ad(output); audit=matrix_audit(result.X)
            saved_genes=pd.read_csv(root/"metadata/gene_order.csv")["gene_name"].astype(str).tolist()
            self.assertEqual(output.name,"Embryonic_raw_count.h5ad"); self.assertEqual(result.n_vars,1024)
            self.assertTrue(audit["nonnegative"] and audit["integer_like"]); self.assertIn("celltype",result.obs); self.assertIn("gene_name",result.var)
            self.assertEqual(saved_genes,result.var["gene_name"].astype(str).tolist())
            self.assertFalse(result.uns["raw_count_processing"]["final_X_log1p"]); self.assertEqual(metadata["shape_after_hvg"][1],1024)

    def test_non_raw_input_is_rejected(self):
        audit=matrix_audit(np.asarray([[0.0,1.2],[-1,2]],np.float32))
        self.assertFalse(audit["nonnegative"]); self.assertFalse(audit["integer_like"])

    def test_training_loader_adds_no_preprocessing(self):
        matrix=np.asarray([[0,1,2],[3,0,4],[5,6,0],[2,2,2]],dtype=np.float32)
        obs=pd.DataFrame({"celltype":["A","A","B","B"]}); var=pd.DataFrame({"gene_name":["a","b","c"]})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"raw.h5ad"; ad.AnnData(matrix,obs=obs,var=var).write_h5ad(path)
            batch,_=next(load_data(data_dir=str(path),batch_size=4,train_vae=True,preprocess=False,layer=None,deterministic=True))
            actual=sorted(map(tuple,batch.numpy().tolist())); expected=sorted(map(tuple,matrix.tolist()))
            self.assertEqual(actual,expected)

    def test_all_six_models_finite_softmax_penalty_and_restore(self):
        base=json.loads((SUITE/"configs/base.json").read_text()); genes=["source","target","g2","g3","g4"]
        with tempfile.TemporaryDirectory() as directory:
            edge=Path(directory)/"edges.tsv"; edge.write_text("from\tto\nsource\ttarget\n")
            x=torch.poisson(torch.full((3,5),2.0)); t=torch.tensor([0.,10.,19.])
            with mock.patch.object(factory,"Cell_Unet",TinyCell):
                for config_path in sorted((SUITE/"configs").glob("*__*.json")):
                    config={**base,**json.loads(config_path.read_text()),"edge_tsv_path":str(edge),"field_hidden":16,"time_dim":8,"cell_unet_hidden_num":[8]*4}
                    model=factory.build_model_from_config(config,genes,20,"cpu"); output=model(x,t)
                    self.assertEqual(output.shape,x.shape); self.assertTrue(torch.isfinite(output).all())
                    expected_class={"softplus":"ConfigurableLinCombField","hill_after_softplus":"HillAfterLinearField","exp":"ExpField"}[config["ode_type"]]
                    self.assertEqual(type(model.ode_model).__name__,expected_class)
                    gate=model.ode_model.get_gate_values(x,t)["coefficients"]
                    self.assertEqual(gate.shape,(3,8)); self.assertTrue((gate>=0).all()); torch.testing.assert_close(gate.sum(1),torch.ones(3))
                    penalty=model.ode_model.off_mask_penalty()
                    expected_penalty=5.0*(model.ode_model.expert_W*(1-model.ode_model.mask)).abs().mean()
                    torch.testing.assert_close(penalty,expected_penalty)
                    (output.square().mean()+penalty).backward()
                    gradients=[parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                    self.assertTrue(gradients); self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
                    self.assertEqual(model.ode_model.penalty_parameter_name if hasattr(model.ode_model,"penalty_parameter_name") else "W","W")
                    state=model.state_dict(); restored=factory.build_model_from_config(config,genes,20,"cpu"); restored.load_state_dict(state,strict=True)
                    self.assertTrue(torch.isfinite(restored(x,t)).all())
                    diffusion=create_gaussian_diffusion(steps=20,noise_schedule="cosine")
                    sample,_=diffusion.p_sample_loop(restored.eval(),(1,5),clip_denoised=False,start_time=20)
                    self.assertTrue(torch.isfinite(sample).all())
                    if config["model_family"].startswith("ode_only"): self.assertFalse(hasattr(model,"ml_model"))
                    else:
                        self.assertTrue(hasattr(model,"ml_model")); expected=1-t/19; torch.testing.assert_close(model._scheduler(t,x.device,x.dtype),expected)

if __name__=="__main__":unittest.main(verbosity=2)
