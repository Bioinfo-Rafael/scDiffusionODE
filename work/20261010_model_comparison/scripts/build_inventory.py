"""Regenerate audit tables from reviewed source references; never imports model copies."""
from pathlib import Path
import ast
import csv
import json
import subprocess

HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[1]
rows=[]
def add(name,family,path,cls,formula,*,time='no',experts='1',gate='none',mask='target_source; soft off-mask',status='implemented',reason='direct import; ordered Mouse IDs + explicit mask',kind='model',hybrid='ODE-only / Hybrid500 / additive',adapter='adapters/model.py'):
    p=ROOT/path
    assert p.is_file(),p
    tree=ast.parse(p.read_text());node=next((n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==cls),None)
    assert node is not None,(path,cls)
    rows.append(dict(name=name,family=family,classification=kind,source_file=path,symbol=cls,line=node.lineno,
        forward_formula=formula,input='x: [cells,genes], optional diffusion t',output='same gene order [cells,genes]',
        time_dependence=time,ode_experts=experts,gating=gate,grn=mask,cellunet_composition=hybrid,
        direct_import='yes',mouse_compatibility=status,status=status,required_adapter=reason,adapter=adapter))
F='work/20260830/models/ode_fields_20260830.py'
add('Simple softplus','autonomous ODE',F,'SimpleSoftplus20260830','softplus((W x+b)/sqrt(G)) - softplus(gamma)*x')
add('Hill after linear','autonomous ODE',F,'HillAfterLinear20260830','V*z^2/(K^2+z^2)-delta*x; z=softplus(Wx+b), delta=softplus(raw_delta)+eps')
add('Centered signed Hill','autonomous ODE',F,'CenteredSignedHill20260830','b_i+sum_j A_ij*tanh(alpha_ij*(log(max(x_j,eps))-log(theta_ij)))-delta_i*x_i')
add('Shifted Hill rho','autonomous ODE',F,'ShiftedHillRho20260830','b_i+sum_j A_ij*expm1(rho_ij)*sigmoid(2*(log(max(x_j,eps))-log(theta_ij)))-delta_i*x_i')
add('CellUNet','denoiser','guided_diffusion/cell_model.py','Cell_Unet','time-conditioned residual MLP encoder/decoder + additive skips + linear output',time='yes',experts='0',mask='none',hybrid='standalone / CellUNet branch')
add('GeneODE 4/21','autonomous ODE','ODE/ode_20260421_regODEMLratio.py','GeneODE','softplus((x W+b)/sqrt(G))-softplus(gamma)*x',mask='source_target; unweighted off-mask + optional cached ratio',reason='mapped TSV + replace constructor sub_gene-index mask with full-order source_target mask; original forward unchanged')
for name,cls,formula in [
 ('LowRank','LowRankField','softplus(U(x,t)V(x,t)^T x+b)-softplus(gamma)*x'),
 ('LinComb','LinCombField','sum_k a_k(x,t)*softplus(W_k x+b_k)-softplus(gamma)*x'),
 ('MatSum','MatSumField','softplus(sum_k a_k(x,t) A_k x+b)-softplus(gamma)*x'),
 ('LoRA','LoRAField','softplus((W0+sum_k a_k(x,t) U_k V_k^T)x+b)-softplus(gamma)*x')]:
 add(name,'state/time-conditioned field','ODE/ode_20260609_mathmlp.py',cls,formula,time='yes',experts='rank=16' if name=='LowRank' else 'K=8 bases/experts',gate='MLP raw coefficients' if name!='LowRank' else 'MLP factors',mask='target_source; unweighted; LowRank caches batch subsample W',reason='9/17 external off_mask_lambda=5; clear LowRank cache between inactive batches')
add('Configurable LinComb','gated field','ODE/ode_20260707_lincomb.py','ConfigurableLinCombField','sum_k a_k*softplus(W_k x+b_k)-softplus(gamma)*x',time='yes',experts='K=8',gate='raw or softmax; sparse(raw) or entropy(softmax)',mask='target_source; internal off_mask_lambda plus gate penalties')
for name,cls,formula in [('Hill after linear experts','HillAfterLinearField','sum_k softmax(g(x,t))_k*[V_k*z_k^2/(K_k^2+z_k^2)-delta*x]'),('RACIPE','RacipeField','G_i*product_j[(1-h_j)+exp(r_ij)*h_j]-delta_i*x_i; h=Hill(softplus(x/scale),K); log production clipped [-20,20]'),('Exponential','ExpField','exp(clamp(Wx+b,-20,20))-delta*x')]:
 add(name,'single or gated ODE','work/20260803_ODE_hill_exp/models/ode_fields.py',cls,formula,time='gate only',experts='1 or K=8',gate='none or softmax')
for name,cls,formula in [('Centered Hill K8','CenteredSignedHillField','sum_k a_k(x,t) * [b_ki+sum_j A_kij*tanh(alpha_kij*log(xpos_j/theta_kij))-delta_i*x_i]'),('Shifted Hill K8','ShiftedHillRhoField','sum_k a_k(x,t)*[b_ki+sum_j A_kij*expm1(rho_kij)*Hill(xpos_j,theta_kij,n=2)-delta_i*x_i]')]:
 add(name,'gated direct Hill','work/20260816/models/ode_fields.py',cls,formula,time='gate only',experts='K=8 enforced',gate='softmax')
for name,cls in [('Centered Hill 8/17','CenteredSignedHillField'),('Shifted Hill 8/17','ShiftedHillRhoField')]:
 add(name,'single direct Hill','work/20260817_singleODE/models/ode_fields.py',cls,'same direct-Hill equation; explicit K=1, no gate network',kind='source variant',status='compatible / not yet implemented',reason='8/30 exact requested source implemented; 8/17 class has distinct state layout/provenance and needs explicit registry key')
add('DirectMessageODE','sparse neural field','work/20260917/src/fields.py','DirectMessageODE','-softplus(gamma_i)*x_i + sum_(j->i) w_ji*phi(x_i,x_j,t)',time='yes',mask='hard sparse support; absent-edge penalty exactly zero')
add('MultiHopGraphFilterODE','sparse neural field','work/20260917/src/fields.py','MultiHopGraphFilterODE','f_theta(x_i,(Px)_i,(P^2x)_i,(P^3x)_i,t); P row-normalized incoming GRN',time='yes',mask='hard sparse support; absent-edge penalty exactly zero')
for name,path,cls,formula,time,experts in [
 ('Original GeneODE','ODE/ode_analysis.py','GeneODE','softplus(x_sub W+b)-gamma*x_sub; zero outside selected subgenes','no','1'),
 ('Sigmoid GeneODE 11/06','ODE/ode_analysis1106.py','GeneODE','sigmoid((xW+b)*scale)-gamma*x','no','1'),
 ('Normalized GeneODE 4/13','ODE/ode_analysis20260413.py','GeneODE','softplus(xW+b)-softplus(gamma)*x','no','1'),
 ('FactorMLP GeneODE 4/16','ODE/ode_analysis20260416.py','GeneODE','softplus(x A(x)B(x)^T/sqrt(rank)+b)-softplus(gamma)*x','state only','rank'),
 ('Time FactorMLP GeneODE','ODE/ode_analysis20260416.py','GeneODE_Time','time-condition x before A,B and production; decay uses original x','yes','rank'),
 ('MixtureOfSoftplusODE','ODE/ode_analysis20260420.py','MixtureOfSoftplusODE','sum_k softmax(AlphaNet(x,t))_k*softplus(x W_k+b_k)-softplus(gamma)*x','optional time','configurable K'),
 ('MatrixDictionaryODE','ODE/ode_analysis20260420.py','MatrixDictionaryODE','softplus(x*(sum_k alpha_k A_k)+b)-softplus(gamma)*x','optional time','configurable K'),
 ('LowRankResidualDictionaryODE','ODE/ode_analysis20260420.py','LowRankResidualDictionaryODE','softplus(xW0+sum_k alpha_k*x U_k V_k^T+b)-softplus(gamma)*x','optional time','configurable K/rank')]:
 add(name,'historical field',path,cls,formula,time=time,experts=experts,mask='source_target; constructor/index/cache policy differs',status='requires adapter',
     reason='needs dedicated full-ID mask/state layout validation and t-less-call adapter for older GeneODE; cannot alias to June fields; hard-mask behavior must be audited separately')
for name,path,cls,formula in [
 ('Standard / norm / ratio / scale / TS','ODE/ode_20260609_hybrid5x3.py','UnifiedODEMLHybrid','r F+(1-r) C; r=1-t/999 or sigmoid((Ts-t)/tau); mode optionally normalizes/scales branches'),
 ('Hybrid500','work/20260915_x0predict/models/__init__.py','SingleODEHybrid500','max(0,1-t/500)*F + (1-max(0,1-t/500))*C'),
 ('Additive Hybrid500','work/20260916_x0predict_hybrid_additive/models.py','AdditiveHybrid500','C + max(0,1-t/500)*F'),
 ('Consistency wrapper','work/20260830/models/cellunet_ode_regularized_20260830.py','CellUNetODERegularized20260830','output=C; training cache=mean_gene((C-F)^2); eval does not mix F')]:
 add(name,'composition',path,cls,formula,time='yes',experts='field-dependent',kind='configuration/composition',hybrid=formula,mask='field-dependent')
for name,path,cls,formula in [
 ('Normalized historical blend','ODE/ode_analysis20260413.py','ODE_ML_Hybrid','norm * (r*unit(F)+(1-r)*unit(C))'),
 ('Learned historical scale','ODE/ode_analysis20260413.py','ODE_ML_HybridLearnedScale','s(detached x,F,C,t) * (r*unit(F)+(1-r)*unit(C))'),
 ('HybridNormModes','ODE/ode_20260609_hybridnorm.py','ODE_ML_HybridNorm','none: blend; ratio_reg: blend + cached log norm penalty; normed: exp(log_scale)*blend(units)'),
 ('MathML Hybrid','ODE/ode_20260609_mathmlp.py','MathML_Hybrid','r F(x,t)+(1-r) C(x,t); cache log norm-ratio penalty')]:
 add(name,'historical composition',path,cls,formula,time='yes',kind='configuration/composition',status='compatible / not yet implemented',reason='direct class can be added; current extension uses explicitly named UnifiedODEMLHybrid modes, not historical checkpoint aliases')
for name,cls,formula,path in [
 ('FreeAffine forward','FreeAffineForward','dX=((W-0.5I)X+b) ds+dB; dense matrix-exponential Gaussian transition','free_affine.py'),
 ('Stationary QD forward','StationaryQDForward','dX=-(Q+D)X ds+sqrt(2D)dB; Q skew, D positive, low-rank structured operator','stationary_qd.py')]:
 add(name,'learnable diffusion process','work/20260903_learnable_forward/diffusion/'+path,cls,formula,time='physical diffusion time',experts='not applicable',mask='forward drift GRN penalty',status='requires adapter',hybrid='LearnableForwardModel + CellUNet epsilon denoiser',
     reason='separate experiment family: nonstandard q, ELBO/boundary loss, shared physical timestep and reverse sampler; incompatible with fixed DDPM swap',adapter='not implemented')
add('Cell classifier','classifier','guided_diffusion/cell_model.py','Cell_classifier','time-conditioned class logits',time='yes',experts='0',mask='none',status='not applicable',kind='auxiliary network',reason='output is class logits, not gene-space field',hybrid='classifier guidance auxiliary')
add('CellUNetVelocity','analysis','work/20260929noneqThermo/cellunet_adapter.py','CellUNetVelocity','V_t=raw START_X CellUNet(clean X,t)',time='yes',experts='0',mask='none',status='not applicable',kind='analysis',reason='restoration/output interpretation only; model already counted as CellUNet')

fields=list(rows[0]);
with (HERE/'MODEL_INVENTORY.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
registry=dict(schema='source_reuse_registry_v1',models=rows,valid_statuses=['implemented','compatible / not yet implemented','requires adapter','incompatible','missing source','not applicable'],
 incompatible_combinations=[
 dict(combination='learnable-forward + native DDPM',status='incompatible',reason='q, score, loss, physical time and reverse transition change'),
 dict(combination='raw-count preprocessing + normalized pilot',status='incompatible',reason='different expression scale; requires separate data protocol'),
 dict(combination='8/16 direct Hill + K1',status='incompatible',reason='source enforces K8; use 8/17 or 8/30 class explicitly'),
 dict(combination='softmax LinComb + sparse coefficient penalty',status='incompatible',reason='7/07 requires sparse only with raw gate; softmax has entropy penalty'),
 dict(combination='state/time-dependent W reported as autonomous GRN ODE',status='incompatible',reason='MLP coefficients depend on input/conditioning timestep'),
 dict(combination='legacy bare EMA + assumed new Mouse gene order',status='incompatible',reason='missing full new-data/cell-order provenance; never infer from tensor shape')],
 runnable_configs={group:[p.stem for p in sorted((HERE/'configs'/group).glob('*.json'))] for group in ['pilot','extended']})
registry['training_methods'] = [
 dict(name=name, status=status, source=source, reason=reason)
 for name,status,source,reason in [
 ('START_X reconstruction','implemented','work/20260915_x0predict/training/objectives.py','native START_X MSE + imported soft constraint'),
 ('EPSILON reconstruction','implemented','work/20260913_2step/training/objectives.py','native EPSILON MSE; separate extension config'),
 ('consistency','implemented','work/20260830/training/train_loop_20260830.py','exact composer; START_X target adaptation'),
 ('frozen reconstruction two-step','implemented','work/20260915_x0predict/models/__init__.py','shared new-data final A01 EMA; imported freeze/optimizer/EMA'),
 ('Sinkhorn gene-space OT','implemented','work/20260915_x0predict/training/objectives.py','START_X and EPSILON extensions keep source target conversions distinct'),
 ('independent PCA OT','requires adapter','work/20260916_x0predict_hybrid_additive/training/objectives.py','fixed PCA/target cache and independent row sampler require new-data provenance'),
 ('trajectory occupation OT','requires adapter','work/20260913_2step/training/trajectory.py','new-data x50 cache, trajectory configuration and independent streams'),
 ('kNN replacement loss','requires adapter','work/20260917/src/knn.py','new-data graph cache and index-aware batch sampler'),
 ('dual AdamW/SGD','requires adapter','work/20261008_ode_optimizer/training/train_loop.py','two optimizer states and resume adapter; pilot explicitly requests AdamW'),
 ('learnable-forward ELBO','requires adapter','work/20260903_learnable_forward/diffusion/training_diffusion.py','separate forward/reverse process family, not fixed-DDPM objective'),
 ('log-linear consistency schedule','compatible / not yet implemented','work/20260830/training/train_loop_20260830.py','source schedule can be bound; pilot requires constant four lambdas'),
 ]]
(HERE/'registry.json').write_text(json.dumps(registry,indent=2)+'\n')
md=['# Model inventory','',f'Audit: 2026-10-10 JST. Local HEAD equals fetched origin/main: `{(HERE/"audit/git_commit.txt").read_text().strip()}`.',
 'Forward equations below were checked in source bodies, not inferred from directory names. `softplus` parameter transforms, matrix orientation, expert counts and cache penalties are distinct. All implemented models keep their original classes in existing directories.',
 '',f'{len(rows)} model/composition/analysis entries. See `registry.json` for compatibility and executable config names; `audit/source_index.json` for all scanned Python files, hashes, classes, methods and line numbers.','']
for row in rows:
 md += ['## '+row['name'],'',f"- Family/classification: {row['family']} / {row['classification']}",f"- Source: `{row['source_file']}:{row['line']}::{row['symbol']}`",f"- Forward: `{row['forward_formula']}`",f"- Input/output: {row['input']} → {row['output']}",f"- Time: {row['time_dependence']}; experts: {row['ode_experts']}; gating: {row['gating']}",f"- GRN: {row['grn']}; CellUNet: {row['cellunet_composition']}",f"- Direct import: {row['direct_import']}; new Mouse status: **{row['status']}**",f"- Adaptation/reason: {row['required_adapter']}",'']
md += ['## Findings that directory names obscure','',
 '- 4/16 Transformer directories implement FactorMLP-generated low-rank fields; the inspected code does not contain transformer attention.',
 '- Early 11/06 GeneODE docstrings say exponential production, but `forward` uses sigmoid. 4/21 uses scaled softplus. These are not aliases.',
 '- June LinComb `compute_W` is a visualization proxy: mixing nonlinear expert outputs is not equal to one softplus of a mixed matrix.',
 '- 8/16 enforces eight gated experts; 8/17 explicitly enforces one; 8/30 supplies the four requested autonomous single fields.',
 '- DirectMessage/MultiHop have hard graph support and exactly zero off-support penalty; metadata must not claim an active soft-edge penalty.',
 '- Native CellUNet defines a dropout member but its forward does not call it; no BatchNorm buffers. Freeze is still checked over all state_dict parameters and buffers.',
 '- No named mandatory model source is missing on the inspected main. Git history includes `ed376ec` for 8/17. README-only aliases are not treated as independent implementations; unaudited historical branches are not assumed available.',
 '- Real Mouse data/checkpoints are absent locally. Mouse compatibility means executable adapter + synthetic validation; real coverage/metrics remain unmeasured.', '']
(HERE/'MODEL_INVENTORY.md').write_text('\n'.join(md))

# Machine-readable, source-linked coverage for every historical work directory.
index=json.loads((HERE/'audit/source_index.json').read_text())
notes={
'20260122_1_ReconstructUMAP':'Sigmoid GeneODE + CellUNet; gene/dataset/sampling-UMAP settings.',
'20260123_pancreas':'Same 11/06 classes on pancreas; dataset variant.',
'20260215_embryonic':'Same 11/06 sigmoid GeneODE hybrid, EPSILON legacy TrainLoop, 2000 CLI-default anneal steps; embryonic data variant.',
'20260413_Normalized_Embryonic':'Softplus GeneODE with positive decay; branch unit normalization.',
'20260416_Transformer':'FactorMLP low-rank state-conditioned field, not attention.',
'20260420_Normalized_LearnScale':'4/13 GeneODE + detached-summary SmallScaleNet; learned global branch magnitude.',
'20260420_TransformerTimeEmbedding':'GeneODE_Time: time-conditioned input before FactorMLP.',
'20260421_MoE':'MixtureOfSoftplus / MatrixDictionary / LowRankResidualDictionary via AlphaNet.',
'20260421_RegODEMLratio':'Scaled softplus GeneODE; log norm-ratio cached in off_mask_penalty.',
'20260609_Hybrid5x3':'Unified Hybrid across five fields × normalization modes; 30k default; scale-model extension.',
'20260609_HybridNormModes':'ratio_reg / none / normed_learned_scale configurations, not three independent ODEs.',
'20260609_MathMLPHybrid':'Dynamic LowRank, raw LinComb, MatSum, LoRA; time features; static versus cached penalties.',
'20260707_lincomb':'raw/softmax gating, sparse/entropy; TS spectral estimate and sigmoid regime scheduler; TrainLoop reuse.',
'20260801':'Launcher/settings for 7/07 TS/reversed coefficients; integrated UMAP and weight-curve analysis.',
'20260802':'Normalization-mode launcher reusing 7/07 implementations; no new field equation.',
'20260803_ODE_hill_exp':'Single/gated Hill-after-linear, RACIPE, clipped exp; factory and legacy train/sample.',
'20260803_distribution':'Distribution analysis and protection verifier; no trainable model.',
'20260804_raw_count_lincomb':'Raw-input variants of softplus/Hill/exp K8; imports 8/03 trainer/factory; distinct data protocol.',
'20260816':'K8 centered/shifted direct Hill, softmax experts and target/source masks.',
'20260817_singleODE':'K1 direct Hill; no MLP or expert gate; source present in main and history.',
'20260817_vector_field_analysis':'Existing vector-field adapters, Dynamo reconstruction and erythroid UMAP; analysis only.',
'20260821_noeqThermo':'Landscapes and probability-current analysis for existing fields; no denoiser training.',
'20260830':'K1 four-field factory, CellUNet-only output + consistency; EPSILON, 100k; detailed loss/grad/UMAP analysis.',
'20260903_learnable_forward':'FreeAffine / StationaryQD forward processes, ELBO or epsilon surrogate, custom reverse sampler; separate family.',
'20260911':'Posthoc trajectories/breakpoints and independent/joint UMAP for existing checkpoints; no new model.',
'20260913_2step':'EPSILON Stage1/Stage2, frozen CellUNet, reconstruction/Sinkhorn/trajectory occupation OT + Hybrid500.',
'20260915_x0predict':'START_X Stage1/Stage2, exact 8/30 K1 fields, Hybrid500; native DDPM; gene-space OT.',
'20260916_x0predict_hybrid_additive':'Additive wrapper, reconstruction or independent-target PCA OT, envelope gradient; 10k stop/30k anneal.',
'20260917':'June LowRank/MatSum/LoRA + DirectMessage/MultiHop; reconstruction or replacing kNN loss; frozen additive branch.',
'20260921':'Projection and nearest-neighbor/geometric diagnostics of existing models; no new trainable field.',
'20260929noneqThermo':'Raw START_X CellUNet velocity adapter for no-equilibrium thermodynamic analysis.',
'20261008_ode_optimizer':'8/30 loss reused; CellUNet AdamW + ODE SGD; 4 lambdas; optional START_X support. Pilot changes both to AdamW.',
'20261009_newBenchmark':'Full-population normalized linear HVG selection, sparse Stage1 START_X, native DDPM, clean-X t49 proxy; old full evaluation.',
'20261009_reproduce':'Official stochastic scVelo per raw erythroid fold, official-raw profile; not a full-trained normalized baseline.'}
coverage=[]
for d in sorted((ROOT/'work').iterdir()):
 if not d.is_dir() or d==HERE:continue
 assert d.name in notes,d.name
 paths=[r for r in index if r['path'].startswith('work/'+d.name+'/')]
 configs=[]
 for p in sorted(d.rglob('*.json')):
  if any(x in p.parts for x in ('runs','results','audit','validation','data')):continue
  if 'config' not in str(p):continue
  try:obj=json.loads(p.read_text())
  except ValueError:continue
  configs.append(dict(path=str(p.relative_to(ROOT)),values=obj))
 coverage.append(dict(directory=d.name,conclusion=notes[d.name],python_files=[r['path'] for r in paths],configs=configs))
(HERE/'audit/work_directory_audit.json').write_text(json.dumps(coverage,indent=2)+'\n')
lines=['# Repository audit coverage','',f'{len(coverage)} historical work directories; {len(index)} Python files indexed (excluding datasets, virtual environments, vendor and generated runs).',
'Each directory below was classified using imports, class/forward/loss/update bodies and its configuration/launcher. Python index is a coverage aid, not a claim that every plotting/test line was manually read.','',
'| Directory | Source finding | Python files |','|---|---|---|']
for r in coverage:lines.append(f"| `{r['directory']}` | {r['conclusion']} | {len(r['python_files'])} |")
lines += ['', 'Per-file symbols/hashes: `source_index.json`. Exact config/launcher values: `work_directory_audit.json`. Existing user edits are listed in `initial_git_status.txt` and included in the local source baseline; no reset/stash was performed.']
(HERE/'audit/WORK_DIRECTORY_AUDIT.md').write_text('\n'.join(lines)+'\n')
print(len(rows),'inventory entries;',len(coverage),'work directories')
