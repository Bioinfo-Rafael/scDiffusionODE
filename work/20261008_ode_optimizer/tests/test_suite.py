"""CPU regression and end-to-end smoke tests; artifacts stay inside this suite."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from common import *
import numpy as np
import torch
from training.data import StatefulBatches
from training.train_loop import TrainLoop, split_parameters


def tiny_config(name='lambda0'):
    c=config_for(name,['total_steps=3','device=cpu','batch_size=4','save_interval=2','log_interval=10',
        'cell_unet_hidden_num=[16,12,8,8]','analysis_cells=8','gradient_cells=4',
        'analysis_batch_size=4','analysis_timesteps=[0,500,999]','umap_real_cells=8',
        'num_samples=4','sample_batch_size=4','pca_components=3','neighbor_pcs=3','neighbors=3'])
    return c


def toy_model(c):
    mask=torch.eye(4)
    return legacy('models.factory').build_model_from_config(c,['a','b','c','d'],1000,'cpu',mask=mask)


def toy_data(c):
    x=np.random.default_rng(123).normal(size=(12,4)).astype(np.float32)
    return StatefulBatches(x,np.zeros(len(x),dtype=np.int64),c['batch_size'],c['seed'])


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp=tempfile.TemporaryDirectory(dir=SUITE/'tests')
        self.root=Path(self.temp.name)
        from guided_diffusion import dist_util,logger
        dist_util.dev=lambda:torch.device('cpu')
        logger.configure(dir=str(self.root/'logs'))

    def tearDown(self): self.temp.cleanup()

    def loop(self,path,c,resume=False):
        seed_all(c['seed'])
        return TrainLoop(model=toy_model(c),diffusion=diffusion_for(c),data=toy_data(c),
                         config=c,run_dir=path,resume=resume)

    def test_optimizer_partition_and_mpt_identity(self):
        c=tiny_config(); loop=self.loop(self.root/'run',c)
        cell,ode=split_parameters(loop.model)
        self.assertFalse(set(map(id,cell)) & set(map(id,ode)))
        self.assertEqual(set(map(id,cell+ode)),set(map(id,loop.mp_trainer.master_params)))
        self.assertEqual(set(map(id,cell)),set(map(id,loop.opt.param_groups[0]['params'])))
        self.assertEqual(set(map(id,ode)),set(map(id,loop.ode_opt.param_groups[0]['params'])))
        loop.hook.remove()

    def test_lambda_zero_cell_gradient_is_diffusion_only(self):
        c=tiny_config(); seed_all(c['seed']); model=toy_model(c); model.train()
        diffusion=diffusion_for(c); batch,_=next(toy_data(c)); t=torch.tensor([0,50,500,999])
        losses=diffusion.training_losses(model,batch,t)
        diffusion_loss=losses['loss'].mean()
        consistency=model.consistency_penalty_20260830()
        soft=model.ode_model.off_mask_penalty_base('l1')*c['off_mask_lambda']
        components=legacy('training.train_loop_20260830').loss_components_20260830(
            diffusion_loss,soft,c['ode_reg_lambda'],consistency,torch.ones(4),0.)
        cell,ode=split_parameters(model)
        diff_grad=torch.autograd.grad(diffusion_loss,cell,retain_graph=True)
        total_grad=torch.autograd.grad(components['total_loss'],cell,retain_graph=True)
        for left,right in zip(diff_grad,total_grad): torch.testing.assert_close(left,right,rtol=0,atol=0)
        zero_grad=torch.autograd.grad(components['cell_ode_consistency_final_weighted_20260830'],cell,retain_graph=True)
        self.assertTrue(all(torch.count_nonzero(g)==0 for g in zero_grad))
        ode_grad=torch.autograd.grad(components['total_loss'],ode,allow_unused=True)
        self.assertTrue(any(g is not None and bool(g.abs().sum()>0) for g in ode_grad))

    def test_checkpoint_resume_matches_uninterrupted(self):
        c=tiny_config('lambda0p1')
        full=self.loop(self.root/'full',c); full.run_loop()
        first=self.loop(self.root/'split',c); first.run_loop(stop_after=2)
        before=torch_load(first.folder/'trainer000002.pt')
        restored=self.loop(self.root/'split',c,resume=True)
        self.assertEqual(restored.step,2)
        self.assertEqual(restored.ode_opt.state_dict()['param_groups'],before['ode_optimizer']['param_groups'])
        self.assertEqual(restored.ode_opt.state_dict()['state'],{}) # momentum=0: empty is expected
        for key,value in before['cell_optimizer']['state'].items():
            for name,tensor in value.items():
                torch.testing.assert_close(restored.opt.state_dict()['state'][key][name],tensor)
        restored.run_loop()
        for a,b in zip(full.model.parameters(),restored.model.parameters()): torch.testing.assert_close(a,b,rtol=0,atol=0)
        for left,right in zip(full.ema_params,restored.ema_params):
            for a,b in zip(left,right): torch.testing.assert_close(a,b,rtol=0,atol=0)
        import pandas as pd
        a=pd.read_csv(full.folder/'training_metrics.csv'); b=pd.read_csv(restored.folder/'training_metrics.csv')
        pd.testing.assert_frame_equal(a,b,check_exact=True)
        self.assertEqual(a.training_step.tolist(),[1,2,3])
        self.assertAlmostEqual(a.cell_learning_rate.iloc[0],1e-4)
        self.assertAlmostEqual(restored.opt.param_groups[0]['lr'],0.)

    def test_actual_updates_include_weight_decay(self):
        c=tiny_config(); loop=self.loop(self.root/'run',c)
        batch,cond=next(loop.data)
        before=[p.detach().clone() for p in loop.ode_params]
        row=loop.run_step(batch,cond)
        expected=[]
        for old,p in zip(before,loop.ode_params):
            target=old-c['ode_lr']*(p.grad+c['ode_weight_decay']*old)
            torch.testing.assert_close(p,target)
            expected.append((p.detach()-old).square().sum())
        self.assertAlmostEqual(row['ode_update_norm'],float(torch.stack(expected).sum().sqrt()),places=10)
        loop.hook.remove()

    def test_equal_initialization_and_data_order(self):
        fingerprints=[]; orders=[]
        for name in CONDITIONS:
            c=tiny_config(name); seed_all(c['seed']); model=toy_model(c)
            fingerprints.append(legacy('analysis.gradients').parameter_fingerprint(model))
            data=toy_data(c); order=[]
            for _ in range(8):
                next(data); order.extend(data.last_indices.tolist())
            orders.append(order)
        self.assertEqual(len(set(fingerprints)),1)
        self.assertTrue(all(order==orders[0] for order in orders))

    def test_sampling_one_trajectory_endpoints_match_legacy(self):
        from sample import capture_trajectory
        c=tiny_config(); seed_all(c['seed']); model=toy_model(c).eval(); diffusion=diffusion_for(c)
        seed_all(98); expected_noise=torch.randn(4,4)
        expected_final,_=diffusion.p_sample_loop(model,(4,4),noise=expected_noise,clip_denoised=False,start_time=1000)
        seed_all(98); states=capture_trajectory(diffusion,model,(4,4),device=torch.device('cpu'))
        np.testing.assert_array_equal(states[0],expected_noise.numpy())
        np.testing.assert_array_equal(states[1000],expected_final.numpy())
        seed_all(98); noise=torch.randn(4,4); observed={}
        for i,out in enumerate(diffusion.p_sample_loop_progressive(model,(4,4),noise=noise,clip_denoised=False,start_time=1000),1):
            if i in STEPS: observed[i]=out['sample'].numpy().copy()
        for s in STEPS[1:]: np.testing.assert_array_equal(states[s],observed[s])

    def test_no_fp16_silent_misregistration(self):
        c=tiny_config(); c['use_fp16']=True
        with self.assertRaisesRegex(ValueError,'FP16'): self.loop(self.root/'bad',c)

    def test_dry_run_four_conditions_no_writes(self):
        batch='dry-run-test-do-not-create'
        result=subprocess.run([sys.executable,'-B',str(SCRIPTS/'launch.py'),'--batch-id',batch,'--dry-run'],capture_output=True,text=True,check=True)
        plan=json.loads(result.stdout)
        self.assertEqual(plan['conditions'],list(CONDITIONS))
        self.assertEqual(len(plan['commands']),11)
        self.assertFalse((SUITE/'runs'/batch).exists())

    def test_protected_existing_files(self):
        subprocess.run([sys.executable,'-B',str(SCRIPTS/'verify_protected.py')],check=True)


class PipelineSmoke(unittest.TestCase):
    """Opt-in, real CPU train/sample/analysis/UMAP pipeline on 4 genes and 12 cells."""
    @unittest.skipUnless(os.environ.get('ODE_OPTIMIZER_SMOKE')=='1','Set ODE_OPTIMIZER_SMOKE=1 for full CPU smoke')
    def test_four_condition_pipeline(self):
        import anndata as ad
        import pandas as pd
        torch.set_num_threads(1)
        root=SUITE/'validation/smoke'
        root.mkdir(parents=True,exist_ok=True)
        a=ad.AnnData(np.random.default_rng(7).normal(size=(12,4)).astype(np.float32),
                    obs=pd.DataFrame({'celltype':['toy']*12},index=[f'c{i}' for i in range(12)]),
                    var=pd.DataFrame({'gene_name':['a','b','c','d']},index=['a','b','c','d']))
        data=root/'toy.h5ad'; a.write_h5ad(data)
        edge=root/'edges.tsv'; edge.write_text('from\tto\na\tb\nc\td\n')
        # Factory's legacy edge parser is bypassed only by using its explicit mask
        # in unit tests. CLI smoke uses the real TSV loader and full data path.
        campaign=root/'campaign'
        import train,sample,analyze,plot_umap,plot_comparison
        for name in CONDITIONS:
            c=tiny_config(name); c.update(data_dir=str(data),edge_tsv_path=str(edge))
            run=campaign/name; write_json(run/'exp_config.json',c)
            args=['--config',str(run/'exp_config.json'),'--run-dir',str(run)]
            if (run/'checkpoints/latest.json').exists(): args+=['--resume']
            train.main(args); sample.main(['--run-dir',str(run),'--force'])
        analyze.main(['--campaign',str(campaign)])
        plot_umap.main(['--campaign',str(campaign),'--refit'])
        plot_comparison.main(['--campaign',str(campaign)])
        for name in CONDITIONS:
            figs=campaign/name/'analysis/figures'
            for index in range(1,13): self.assertTrue(list(figs.glob(f'{index:02d}_*.png')))
            for step in STEPS:
                self.assertTrue((campaign/'umap'/f'umap_{name}_s{step:04d}.png').is_file())
        final=campaign/'umap/umap_comparison_600_1000_all_conditions.png'
        self.assertGreater(final.stat().st_size,10000)
        from PIL import Image
        with Image.open(final) as im: self.assertGreater(im.height,im.width)
        before=sha256(campaign/'umap/coordinates.npz')
        plot_umap.main(['--campaign',str(campaign),'--redraw-only'])
        self.assertEqual(before,sha256(campaign/'umap/coordinates.npz'))
        diff=pd.read_csv(campaign/'summary/prediction_difference_vs_lambda0.csv')
        self.assertTrue((diff[diff.experiment=='lambda0'].prediction_mse==0).all())
        # All conditions use corresponding initial Gaussian cells.
        noises=[np.load(campaign/name/'samples/step_0000.npz')['cell_gen'] for name in CONDITIONS]
        self.assertTrue(all(np.array_equal(noises[0],x) for x in noises[1:]))
        info=[read_json(campaign/name/'model_info.json')['initial_parameter_sha256'] for name in CONDITIONS]
        self.assertEqual(len(set(info)),1)

if __name__=='__main__': unittest.main()
