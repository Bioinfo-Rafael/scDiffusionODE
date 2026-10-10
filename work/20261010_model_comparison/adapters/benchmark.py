"""Subprocess contract only. All fold geometry and four metrics remain upstream."""
from __future__ import annotations
from ..common import *
BENCH=ROOT/'data_preparation/20261007/benchmark'
DEFAULT_REFERENCE=ROOT/'data_preparation/20261007/data/benchmark/references/normalized-3fold'
METRICS=('cbdir','icvcoh','cto','tsc')


def command(prediction, genes, output, method, reference=None):
    return [sys.executable,str(BENCH/'run.py'),'--evaluation','3fold',
            '--prediction',str(Path(prediction).resolve()),'--genes',str(Path(genes).resolve()),
            '--method',method,'--reference',str(Path(reference or DEFAULT_REFERENCE).resolve()),
            '--output-dir',str(Path(output).resolve())]


def prepare_reference(c,reference):
    reference=Path(reference)
    if (reference/'manifest.json').exists():return
    require(not reference.exists(),'incomplete benchmark reference exists; inspect before retry')
    subprocess.run([sys.executable,str(BENCH/'prepare.py'),'--evaluation','3fold','--profile','normalized',
                    '--input',str(root_path(c['input'])),'--output',str(reference)],check=True,cwd=ROOT)


def evaluate(c, run, *, fields=None, reference=None, prepare=False, prediction=None, genes=None, label=None, dry_run=False):
    run=Path(run);reference=Path(reference or DEFAULT_REFERENCE)
    if prediction:
        require(genes is not None and label, 'external prediction requires --genes and --label')
        entries=[(label,Path(prediction),Path(genes))]
    else:
        selected=fields or branch_names(c)
        require(set(selected)<=set(branch_names(c)),'unknown branch')
        entries=[(f'{field}_t{c["field_timestep"]}',run/'predictions'/f'{field}_t{c["field_timestep"]}.h5ad',run/'predictions/gene_ids.txt') for field in selected]
    results=[]
    for field,pred,gene in entries:
        require(Path(field).name==field,'invalid output label')
        out=confined(run/'metrics'/field)
        cmd=command(pred,gene,out,c['condition']+'_'+field,reference)
        if dry_run:results.append(cmd);continue
        if prepare:prepare_reference(c,reference)
        manifest=read_json(reference/'manifest.json')
        require(manifest['status']=='complete' and manifest['profile']=='normalized', 'requires normalized completed three-fold reference')
        if not prediction:
            exported=read_json(pred.with_suffix('.json'))
            require(exported['output_sha256']==sha256(pred),'exported prediction changed')
            require(manifest['input']['sha256']==exported['input_sha256'],
                    'benchmark reference and training source input differ')
        stamp=dict(prediction_sha256=sha256(pred),genes_sha256=sha256(gene),reference_sha256=sha256(reference/'manifest.json'),
                   inference_protocol='full-trained',profile='normalized',evaluation='3fold',field=field)
        sidecar=run/'metrics'/(field+'_request.json')
        if out.exists():
            require(sidecar.exists() and read_json(sidecar)==stamp,'existing evaluation provenance differs')
            require(read_json(out/'metadata.json')['status']=='complete','incomplete evaluation; preserve it and use new --label/output run')
            results.append(out);continue
        write_json(sidecar,stamp)
        logfile=run/'logs'/(field+'_benchmark.log');logfile.parent.mkdir(parents=True,exist_ok=True)
        try:
            with logfile.open('w') as log:
                subprocess.run(cmd,check=True,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            require(read_json(out/'metadata.json')['status']=='complete','benchmark did not complete')
        except Exception as exc:
            write_json(run/'metrics'/(field+'_failure.json'),dict(error=str(exc),log=str(logfile),command=cmd))
            raise
        results.append(out)
    return results
