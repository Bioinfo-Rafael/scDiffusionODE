"""Run: python -m unittest work.20260917.tests.test_smoke -v"""
import copy
import csv
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

C = importlib.import_module("work.20260917.common")
M = importlib.import_module("work.20260917.models")
K = importlib.import_module("work.20260917.src.knn")
G = importlib.import_module("work.20260917.src.grn")
O = importlib.import_module("work.20260917.training.objectives")
R = importlib.import_module("work.20260917.training.runner")
CP = importlib.import_module("work.20260917.training.checkpoints")
F = importlib.import_module("work.20260917.src.fields")
OLD = importlib.import_module("ODE.ode_20260609_mathmlp")


class NoDenseGeneMatrix(TorchDispatchMode):
    def __init__(self, genes):
        super().__init__(); self.genes = genes

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for value in tree_flatten(result)[0]:
            if isinstance(value, torch.Tensor) and value.layout == torch.strided:
                if value.ndim >= 2 and tuple(value.shape[-2:]) == (self.genes,self.genes):
                    raise AssertionError(f"dense GxG created by {func}")
        return result


class Smoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory(dir=C.SUITE / "cache", prefix="smoke_")
        cls.root = Path(cls.tmp.name)
        cls.genes = ["c", "a", "b", "d", "e", "f", "isolated"]
        cls.edge = cls.root / "edges.tsv"
        cls.edge.write_text("from\tto\na\tb\nb\tc\nc\ta\na\tb\nabsent\tc\ne\tf\n")
        cls.matrix = np.random.default_rng(33).normal(0,.15,(12,7)).astype("float32")
        cls.overrides = dict(edge_tsv_path=str(cls.edge), cell_unet_hidden_num=[12,8],
                             time_dim=8, field_hidden=8, graph_hidden=8, K=2, rank=2,
                             batch_size=3, epochs=2, total_steps=8, save_interval=8, log_interval=8,
                             num_samples=2,sample_batch_size=2,edge_chunk_size=2,
                             knn=dict(k=2,num_negatives=3,edge_chunk_size=3,query_batch_size=4))
        cls.cfg = C.effective_config(overrides=cls.overrides)
        cls.graph = K.prepare_graph(cls.matrix, cls.cfg["knn"], {"fixture":"smoke"}, cls.root/"graphs")
        C.seed_all(1234)
        cls.cell_state = M.old.FrozenCellUNet(input_dim=7, hidden_num=[12,8]).state_dict()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def model(self, kind, loss="reconstruction"):
        c = C.effective_config(kind,loss,self.overrides)
        C.seed_all(c["seed"])
        model = M.build_model(c, self.genes)
        M.freeze_from_stage1(model, self.cell_state)
        model.train()
        return c, model

    def test_mapping_direction_and_exact_source_classes(self):
        src,dst,report = G.load_grn(self.genes,self.edge)
        self.assertEqual(list(zip(src.tolist(),dst.tolist())),[(1,2),(2,0),(0,1),(4,5)])
        self.assertEqual(report["dropped_unmapped_edges"],1)
        self.assertEqual(report["duplicate_mapped_edges"],1)
        with self.assertRaises(ValueError):
            G.load_grn(["a","a"],self.edge)
        for kind,cls in (("lowrank",OLD.LowRankField),("matsum",OLD.MatSumField),("lora",OLD.LoRAField)):
            _,model = self.model(kind)
            self.assertIs(type(model.ode_model),cls)
            self.assertEqual(model.ode_model.mask[2,1],1)
            self.assertEqual(model.ode_model.mask[1,2],0)

    def test_ten_conditions_forward_backward_and_two_epochs(self):
        for kind in C.MODELS:
            for loss_type in C.LOSSES:
                with self.subTest(model=kind,loss=loss_type):
                    c, model = self.model(kind,loss_type)
                    x = torch.from_numpy(self.matrix[:3]); t = torch.tensor([0,250,499])
                    output = model(x,t[:,None]); self.assertEqual(output.shape,x.shape)
                    loss,values = O.training_loss(model,C.build_diffusion(c),x,t,torch.ones(3),c,
                        ids=np.arange(3),matrix=self.matrix,graph=self.graph,rng=np.random.default_rng(1))
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    grads = [p.grad for p in model.ode_model.parameters() if p.grad is not None]
                    self.assertTrue(any(g.abs().sum()>0 for g in grads))
                    self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                    self.assertTrue(all(p.grad is None for p in model.ml_model.parameters()))
                    if kind == "direct_message":
                        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.ode_model.phi.parameters()))
                    if loss_type == "knn":
                        self.assertEqual(values["loss_reconstruction"],0.)
                    else:
                        self.assertEqual(values["loss_knn"],0.)
                    run = C.new_dir(self.root / c["condition"])
                    meta = dict(effective_config=c,gene_names=self.genes,grn_edges=4,
                                frozen_cellunet_hash_before=C.state_hash(model.ml_model))
                    with patch("sklearn.neighbors.NearestNeighbors.fit", side_effect=AssertionError("graph recomputed inside training")):
                        checkpoint = R.train_loop(model,self.matrix,c,run,meta,"cpu",self.graph if loss_type=="knn" else None)
                    report = C.read_json(run/"performance.json")
                    self.assertEqual(report["complete_epochs"],2)
                    restored,_ = CP.restore(checkpoint)
                    self.assertTrue(torch.isfinite(restored(x,t[:,None])).all())
                    M.assert_frozen(restored, expected=C.state_hash(self.cell_state))

    def test_cache_and_negative_exclusion_and_formula(self):
        with patch("sklearn.neighbors.NearestNeighbors.fit", side_effect=AssertionError("cache refit")):
            reused = K.prepare_graph(self.matrix,self.cfg["knn"],{"fixture":"smoke"},self.root/"graphs")
        self.assertTrue(reused.cache_hit)
        ids = np.arange(12)
        negative = K.sample_negatives(ids,self.graph.indices,12,300,np.random.default_rng(4))
        for i in ids:
            self.assertFalse(np.isin(negative[i],np.r_[i,self.graph.indices[i]]).any())
        # Dense-complement edge case: exactly one available negative, no rejection loop.
        self.assertTrue((K.sample_negatives([0],np.array([[1,2]]),4,8,np.random.default_rng(1))==3).all())
        # Scalar definition independently recomputed on the same sampled IDs.
        c,m = self.model("direct_message","knn")
        x = torch.from_numpy(self.matrix[:3]); t=torch.zeros(3)
        loss,pos,neg=K.graph_loss(m.ode_model,x,t,np.arange(3),self.matrix,self.graph,c["knn"],np.random.default_rng(9))
        y=K.integrate(m.ode_model,x,t,c["knn"])
        nn=K.sample_negatives(np.arange(3),self.graph.indices[:3],12,3,np.random.default_rng(9))
        def q(index):
            return 1/(1+(y[:,None,:]-torch.tensor(self.matrix[index])).square().sum(-1))
        pos_ref=-(torch.tensor(np.array(self.graph.weights[:3]))*torch.log(q(self.graph.indices[:3])+1e-8)).mean()
        neg_ref=-torch.log(1-q(nn)+1e-8).mean()
        torch.testing.assert_close(pos,pos_ref); torch.testing.assert_close(neg,neg_ref)
        torch.testing.assert_close(loss,pos_ref+neg_ref)

    def test_schedule_and_original_reconstruction(self):
        c,m=self.model("lora")
        x=torch.from_numpy(self.matrix[:6]); t=torch.tensor([999,501,500,499,250,0])
        branches=m.branch_outputs(x,t[:,None])
        p=((500-t)/500).clamp(0,1)[:,None]
        torch.testing.assert_close(branches["hybrid_cell_weight"],p)
        torch.testing.assert_close(branches["ode_weight"],p)
        torch.testing.assert_close(branches["output"],(1-p)*branches["ml_raw"]+p*branches["ml_raw"]+p*branches["ode_raw"])
        d=C.build_diffusion(c); noise=torch.ones_like(x,dtype=torch.float64)
        total,values=O.training_loss(m,d,x,t,torch.ones(6),c,noise=noise)
        expected=d.training_losses(m,x,t,noise=noise)["loss"].mean()
        self.assertAlmostEqual(values["loss_reconstruction"],float(expected),places=6)
        # Cached LowRank penalty must not leak into all-inactive batches.
        c,m=self.model("lowrank")
        m(x,torch.zeros(6,1)); self.assertIsNotNone(m.ode_model._cached_W_sub)
        m(x,torch.full((6,1),999)); self.assertIsNone(m.ode_model._cached_W_sub)

    def test_zero_lambda_skips_graph_and_reconstruction(self):
        for kind in C.MODELS:
            c,m=self.model(kind,"knn"); c["knn"]["lambda_knn"]=0.
            d=C.build_diffusion(c)
            with patch.object(d,"training_losses",side_effect=AssertionError("reconstruction in K")), patch.object(O,"graph_loss",side_effect=AssertionError("kNN with zero weight")):
                loss,values=O.training_loss(m,d,torch.from_numpy(self.matrix[:3]),torch.zeros(3,dtype=torch.long),torch.ones(3),c)
                loss.backward()
            self.assertEqual(values["knn_contribution"],0.)
            self.assertTrue(torch.isfinite(loss))

    def test_sparse_rhs_forward_and_backward_never_dense(self):
        src,dst,_=G.load_grn(self.genes,self.edge)
        for field in (F.DirectMessageODE(7,src,dst,8,2),F.MultiHopGraphFilterODE(7,src,dst,8)):
            x=torch.from_numpy(self.matrix[:3]).clone().requires_grad_()
            with NoDenseGeneMatrix(7):
                y=field(x,torch.ones(3)); y.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
        # Directed row-normalized P: gene b receives a, gene c receives b.
        field=F.MultiHopGraphFilterODE(7,src,dst,8)
        seen=[]
        handle=field.f_theta.register_forward_pre_hook(lambda _,args:seen.append(args[0].detach()))
        x=torch.arange(7,dtype=torch.float32)[None,:]
        field(x,0); handle.remove()
        self.assertEqual(seen[0][0,2,1].item(),x[0,1].item())
        self.assertEqual(seen[0][0,0,2].item(),x[0,1].item())
        self.assertEqual(seen[0][0,1,3].item(),x[0,1].item())
        self.assertTrue((seen[0][0,6,1:4]==0).all())

    def test_native_sampling_and_evaluation_adapters(self):
        c,m=self.model("direct_message")
        c["post_ode_dt"]=.001
        out=self.root/"trajectory"
        sampler=importlib.import_module("work.20260917.sampling.trajectory")
        C.seed_all(3)
        sampler.sample_to_disk(m.eval(),C.build_diffusion(c),out,c,count=2,batch_size=2,device="cpu",provenance=dict(gene_names=self.genes))
        states=np.load(out/"sample_state.npy")
        predictions=np.load(out/"pred_xstart.npy")
        np.testing.assert_allclose(states[:,19],predictions[:,19],atol=1e-6)
        self.assertEqual(states.shape,(2,22,7))
        for name in ("runner","diagnostics","distributions","umaps","plotting","evaluation_ot","sliced_wasserstein"):
            module=importlib.import_module("work.20260917.analysis."+name)
            self.assertIsNotNone(module)
        # Private adapters preserve original module confinement.
        previous=importlib.import_module("work.20260916_x0predict_hybrid_additive.common")
        self.assertNotEqual(C.SUITE,previous.SUITE)

    def test_evaluation_metrics_and_summary_end_to_end(self):
        campaign=C.new_dir(self.root/"comparison_fixture")
        C.new_dir(campaign/"configs")
        for kind in C.MODELS:
            for loss in C.LOSSES:
                c=C.effective_config(kind,loss,self.overrides)
                C.write_json(campaign/"configs"/(c["condition"]+".json"),c)
        c,m=self.model("direct_message")
        output=C.new_dir(campaign/c["condition"]/"sampling"/"fixture"/"analyze"/"fixture")
        c.update(analysis_ot_cells=6,analysis_pairs=100)
        c["evaluation"].update(sliced_wasserstein_projections=8,sliced_wasserstein_points=8)
        table=[dict(snapshot_index=0,reverse_step=1000,phase="diffusion",diffusion_t=0,
                    ode_step=None,integration_time=None,ode_conditioning_t=None)]
        metrics=importlib.import_module("work.20260917.analysis.distributions")
        metrics.trajectory_metrics(self.matrix[:6,None,:],self.matrix,table,output,c,seed=3)
        diagnostics=importlib.import_module("work.20260917.analysis.diagnostics")
        with patch.dict(diagnostics.diagnostics.__globals__,timestep_grids=lambda:([0,500,999],[0])):
            diagnostics.diagnostics(m.eval(),C.build_diffusion(c),self.matrix[:3],output,batch_size=3,seed=3,division_epsilon=1e-12)
        C.write_json(output/"completed.json",dict(status="completed"))
        summary=importlib.import_module("work.20260917.scripts.summarize").summarize(campaign)
        with (summary/"comparison.csv").open() as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),10)
        row=next(r for r in rows if r["condition"]==c["condition"])
        self.assertEqual(row["status"],"evaluated")
        self.assertIn("true_x0_t0_pearson",row)
        self.assertIn("sinkhorn_to_real__sinkhorn_divergence",row)
        self.assertIn("collapse_coverage__added_real_knn_radius_coverage",row)
        self.assertTrue((summary/"all_metrics.csv").exists())

    def test_rng_and_initialization_match_across_objectives(self):
        snapshots=[]
        for loss_type in C.LOSSES:
            c,m=self.model("direct_message",loss_type)
            before=C.state_hash(m)
            C.seed_all(c["seed"])
            diffusion=C.build_diffusion(c)
            sampler=O.timestep_sampler(c,diffusion)
            t,w=sampler.sample(3,"cpu")
            loss,_=O.training_loss(m,diffusion,torch.from_numpy(self.matrix[:3]),t,w,c,
                ids=np.arange(3),matrix=self.matrix,graph=self.graph,rng=np.random.default_rng(1))
            loss.backward()
            snapshots.append((before,t,torch.get_rng_state(),np.random.get_state()))
        self.assertEqual(snapshots[0][0],snapshots[1][0])
        torch.testing.assert_close(snapshots[0][1],snapshots[1][1])
        torch.testing.assert_close(snapshots[0][2],snapshots[1][2])
        np.testing.assert_array_equal(snapshots[0][3][1],snapshots[1][3][1])


if __name__ == "__main__":
    unittest.main()
