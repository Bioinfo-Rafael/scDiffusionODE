#!/usr/bin/env python3
"""Legacy 01-12 per-run figures plus four-condition measurements/comparisons."""
import argparse
import numpy as np
from common import *


def per_run(run):
    import pandas as pd
    import torch
    c=validate(read_json(run/'exp_config.json')); device=device_for(c)
    root=run/'analysis'; csv_dir=root/'checkpoint_csv'; figs=root/'figures'
    csv_dir.mkdir(parents=True,exist_ok=True)
    x,genes,_=load_cells(c)
    indices=np.sort(np.random.default_rng(c['seed']).choice(len(x),min(c['analysis_cells'],len(x)),replace=False))
    clean=torch.from_numpy(x[indices]); np.save(root/'cell_indices.npy',indices)
    diffusion=diffusion_for(c); model=build(c,genes,device); cp=checkpoint(run)
    model.load_state_dict(torch_load(cp)); step=legacy('analysis.gradients').checkpoint_training_step(cp)
    meta=dict(experiment=c['experiment'],ode_type=c['ode_type'],prediction_target=prediction_target(c),
        cell_ode_reg_lambda_20260830=c['cell_ode_reg_lambda_20260830'],
        cell_ode_reg_schedule_20260830='constant',run_directory=str(run),
        checkpoint_path=str(cp),checkpoint_training_step=step,
        cell_ode_reg_lambda_effective_20260830=c['cell_ode_reg_lambda_20260830'])
    dm,om=legacy('analysis.metrics').evaluate_diffusion_timesteps(model,diffusion,clean,c['analysis_timesteps'],
        batch_size=c['analysis_batch_size'],seed=c['seed'],device=device,metadata=meta)
    history,fractions=legacy('analysis.loss_history').load_loss_history(run,c,rolling_window=100)
    history=history[history.training_step<=step]; fractions=fractions[fractions.training_step<=step]
    grads=[]
    selections=legacy('analysis.gradients').select_analysis_checkpoints(cp.parent.glob('model*.pt'))
    for selection in selections:
        if selection['checkpoint_training_step']>step: continue
        model.load_state_dict(torch_load(selection['checkpoint_path']))
        grads.append(legacy('analysis.gradients').analyze_gradients(model,diffusion,
            clean[:c['gradient_cells']],(0,500,999),cell_ode_lambda=c['cell_ode_reg_lambda_20260830'],
            ode_reg_lambda=c['ode_reg_lambda'],seed=c['seed'],device=device,metadata={**meta,**selection}))
    gradients=pd.concat(grads,ignore_index=True)
    for name,frame in [('diffusion_metrics_by_timestep',dm),('cell_ode_metrics_by_timestep',om),
                       ('gradient_metrics',gradients)]:
        frame['measurement_source']='checkpoint_posthoc_no_optimizer_step'
        frame.to_csv(csv_dir/f'{name}.csv',index=False)
    history.to_csv(root/'loss_history_from_training.csv',index=False)
    fractions.to_csv(root/'loss_fraction_from_training.csv',index=False)
    legacy('analysis.plotting').plot_run_figures(dm,om,history,fractions,gradients,figs,rolling_window=100)
    write_json(root/'analysis.json',dict(**meta,total_steps=c['total_steps'],
        lr_schedule=f"linear decay over {c['lr_anneal_steps']} updates",seed=c['seed'],
        analysis_timesteps=c['analysis_timesteps'],legacy_figures='01 through 12 reused unchanged',
        training_source=str(cp.parent/'training_metrics.csv'),
        posthoc_warning='Checkpoint gradients are evaluations, not historical optimizer gradients or updates'))


def line_grid(frame, x, columns, output, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(columns),1,figsize=(11,3.3*len(columns)),squeeze=False)
    for ax,column in zip(axes.flat,columns):
        for name in CONDITIONS:
            part=frame[frame.experiment==name]
            if len(part): ax.plot(part[x],part[column],label=f'lambda={CONDITIONS[name]:g}',linewidth=1)
        ax.set(xlabel=x,ylabel=column); ax.grid(alpha=.25); ax.legend()
    fig.suptitle(title); fig.tight_layout(); output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=150); plt.close(fig)


def prediction_differences(runs, folder):
    import pandas as pd
    import torch
    baseline=next(r for r in runs if r.name=='lambda0')
    c=read_json(baseline/'exp_config.json'); device=device_for(c)
    x,genes,_=load_cells(c)
    indices=np.sort(np.random.default_rng(c['seed']).choice(len(x),min(c['analysis_cells'],len(x)),replace=False))
    clean=torch.from_numpy(x[indices]); diffusion=diffusion_for(c)
    # Persist exact noisy inputs and timesteps, allowing independent reruns.
    gen=torch.Generator().manual_seed(c['seed']+1729)
    fixed={t:diffusion.q_sample(clean,torch.full((len(clean),),t,dtype=torch.long),
             noise=torch.randn(clean.shape,generator=gen)) for t in c['analysis_timesteps']}
    np.savez_compressed(folder/'fixed_prediction_inputs.npz',cell_indices=indices,
                         **{f'x_t_{t}':v.numpy() for t,v in fixed.items()})
    def predict(run):
        model=build(c,genes,device); cp=checkpoint(run)
        model.load_state_dict(torch_load(cp)); model.eval(); values={}
        with torch.no_grad():
            for t,inputs in fixed.items():
                chunks=[]
                for part in inputs.split(c['analysis_batch_size']):
                    ts=diffusion._scale_timesteps(torch.full((len(part),),t,dtype=torch.long,device=device)).unsqueeze(1)
                    chunks.append(model.ml_model(part.to(device),ts).cpu())
                values[t]=torch.cat(chunks)
        return values,cp
    ref,ref_cp=predict(baseline); rows=[]
    final_steps={legacy('analysis.gradients').checkpoint_training_step(checkpoint(r)) for r in runs}
    if len(final_steps)!=1: raise ValueError('Prediction comparison requires equal checkpoint training steps')
    for run in runs:
        pred,cp=predict(run)
        np.savez_compressed(folder/f'predictions_{run.name}.npz',**{f't_{t}':v.numpy() for t,v in pred.items()})
        for t in fixed:
            delta=(pred[t]-ref[t]).double()
            rows.append(dict(experiment=run.name,lambda_value=CONDITIONS[run.name],diffusion_timestep=t,
                prediction_mse=float(delta.square().mean()),prediction_l2_mean=float(delta.norm(dim=1).mean()),
                relative_l2=float(delta.norm()/(ref[t].double().norm()+1e-12)),checkpoint=str(cp),
                baseline_checkpoint=str(ref_cp),checkpoint_training_step=next(iter(final_steps)),
                measurement_source='checkpoint_fixed_identical_inputs_eval',cells=len(clean)))
    frame=pd.DataFrame(rows); frame.to_csv(folder/'prediction_difference_vs_lambda0.csv',index=False)
    line_grid(frame,'diffusion_timestep',['prediction_mse','prediction_l2_mean','relative_l2'],
              folder/'prediction_difference_vs_lambda0.png','CellUnet prediction difference vs lambda=0; fixed x_t and t')


def parameter_distributions(runs, folder):
    import pandas as pd
    from analysis import parameter_plots as old
    old.CONDITIONS=tuple(old.Condition(f'{CONDITIONS[r.name]:g}',r.name,'runs') for r in runs)
    # Preserve initial raw + intermediate/final EMA columns, including a non-5k final checkpoint.
    available=[]
    for run in runs:
        c=read_json(run/'exp_config.json'); cp=checkpoint(run)
        final_step=legacy('analysis.gradients').checkpoint_training_step(cp)
        available.append({int(p.stem.rsplit('_',1)[1]) for p in cp.parent.glob(f"ema_{c['ema_rate']}_*.pt")
                          if int(p.stem.rsplit('_',1)[1])<=final_step})
    steps=sorted(set.intersection(*available)-{0})
    old.CHECKPOINTS=(old.Checkpoint('initial raw\n0',0,False),)+tuple(old.Checkpoint(f'EMA\n{s}',s,True) for s in steps)
    snapshots={}; rows=[]
    for run in runs:
        c=read_json(run/'exp_config.json'); root=checkpoint(run).parent
        for spec in old.CHECKPOINTS:
            path=root/old._checkpoint_filename(spec,c['ema_rate'])
            cats,_,_=old.load_snapshot_categories(path,positive_epsilon=c['positive_epsilon'],include_individual_cellunet=False)
            snapshots[(run.name,spec.training_step)]=cats
            for key,values in cats.items():
                rows.append(dict(experiment=run.name,checkpoint_training_step=spec.training_step,
                    category=key,checkpoint=str(path),measurement_source='checkpoint_parameter_distribution',**old.summarize(values)))
    folder.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(folder/'parameter_distributions.csv',index=False)
    # Save histogram bin/count data as well as moments, so grids can be redrawn.
    histogram=[]
    for category in (*old.CATEGORY_ORDER,*old.CELLUNET_AGGREGATE_ORDER):
        values=[v[category] for v in snapshots.values()]; edges=old._histogram_edges(values,100)
        for (name,step),snapshot in snapshots.items():
            counts,_=np.histogram(snapshot[category],edges)
            for lo,hi,count in zip(edges[:-1],edges[1:],counts):
                histogram.append(dict(experiment=name,checkpoint_training_step=step,category=category,left=lo,right=hi,count=int(count)))
        old.plot_category_grid(snapshots,category,folder/f'{category}.png',bins=100,dpi=100,shared_edges=edges)
    pd.DataFrame(histogram).to_csv(folder/'parameter_histogram_bins.csv',index=False)


def summarize(runs, folder):
    import pandas as pd
    folder.mkdir(parents=True,exist_ok=True)
    frames=[]; summary=[]
    for run in runs:
        c=read_json(run/'exp_config.json'); cp=checkpoint(run)
        f=pd.read_csv(cp.parent/'training_metrics.csv'); f=f[f.training_step<=legacy('analysis.gradients').checkpoint_training_step(cp)]
        f['experiment']=run.name; frames.append(f)
        d=pd.read_csv(run/'analysis/checkpoint_csv/diffusion_metrics_by_timestep.csv')
        o=pd.read_csv(run/'analysis/checkpoint_csv/cell_ode_metrics_by_timestep.csv')
        summary.append(dict(experiment=run.name,lambda_value=CONDITIONS[run.name],prediction_target=prediction_target(c),
            total_steps=c['total_steps'],lr_anneal_steps=c['lr_anneal_steps'],
            checkpoint_training_step=legacy('analysis.gradients').checkpoint_training_step(cp),
            cell_target_mse=float(d.cell_target_mse_mean.mean()),cell_ode_mse=float(o.cell_ode_mse_mean.mean()),
            cell_target_pearson=float(d.cell_target_pearson_global.mean()),
            cell_ode_pearson=float(o.cell_ode_pearson_global.mean())))
    frame=pd.concat(frames,ignore_index=True); frame.to_csv(folder/'training_measurements.csv',index=False)
    line_grid(frame,'training_step',['cell_update_norm','ode_update_norm'],folder/'parameter_update_norm.png','Actual parameter updates measured during training (includes weight decay)')
    line_grid(frame,'training_step',['cell_gradient_norm','ode_gradient_norm'],folder/'gradient_norm.png','Gradients measured during training (before optimizer weight decay)')
    line_grid(frame,'training_step',['diffusion_loss','ode_offmask_base_raw','ode_regularization_final_weighted',
              'cell_ode_consistency_raw_20260830','cell_ode_consistency_final_weighted_20260830','total_loss'],folder/'loss_curves.png','Training loss components: raw and weighted')
    line_grid(frame,'training_step',['ode_output_norm'],folder/'ode_output_norm.png','Training: mean per-cell ODE output L2 norm')
    sf=pd.DataFrame(summary); sf.to_csv(folder/'condition_summary.csv',index=False)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,key in zip(axes.flat,('cell_target_mse','cell_ode_mse','cell_target_pearson','cell_ode_pearson')):
        ax.bar([f"lambda={CONDITIONS[n]:g}" for n in sf.experiment],sf[key]); ax.set_title(key)
    fig.suptitle('Four conditions: checkpoint metrics averaged over identical diffusion timesteps')
    fig.tight_layout(); fig.savefig(folder/'condition_summary.png',dpi=150); plt.close(fig)
    write_json(folder/'comparison_protocol.json',dict(training_lengths=summary,
        historical_caveat='20260830 includes 30000- and 100000-step runs; linear LR horizons differ. No historical result is treated as a matched control.',
        measured_updates='from training CSV, never reconstructed from EMA/checkpoint differences'))
    if len(runs)==4:
        prediction_differences(runs,folder)
        parameter_distributions(runs,folder/'parameters')


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--run-dir',action='append')
    p.add_argument('--campaign'); a=p.parse_args(argv)
    if a.campaign:
        runs=campaign_runs(a.campaign,require_all=False)
    elif a.run_dir:
        runs=[inside(r) for r in a.run_dir]
    else: p.error('--campaign or --run-dir is required')
    for r in runs: per_run(r)
    output=inside(a.campaign)/'summary' if a.campaign else runs[0]/'analysis/summary'
    summarize(runs,output)

if __name__=='__main__': main()
