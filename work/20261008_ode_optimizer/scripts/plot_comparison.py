#!/usr/bin/env python3
"""Redraw four conditions x s=600/1000 from the shared saved coordinates."""
import argparse
import numpy as np
from common import *
from plot_umap import draw_panel, SELECTION_VERSION

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--campaign',required=True); a=p.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=inside(a.campaign)/'umap'
    meta=read_json(folder/'embedding.json')
    if meta.get('settings',{}).get('selection_version')!=SELECTION_VERSION:
        raise ValueError('Refit with all Erythropoietic real cells before making comparison')
    if meta['fit_count']!=1 or meta['coordinate_sha256']!=sha256(folder/'coordinates.npz'):
        raise ValueError('Shared coordinate provenance is invalid')
    with np.load(folder/'coordinates.npz') as data:
        fig,axes=plt.subplots(4,2,figsize=(12,19),sharex=True,sharey=True)
        for row,(name,value) in enumerate(CONDITIONS.items()):
            for col,step in enumerate((600,1000)):
                draw_panel(axes[row,col],data['real'],data[f'{name}__{step}'],
                           f'{name} (lambda={value:g}) | sampling s={step}',data['limits'])
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=2,markerscale=4)
        fig.tight_layout(rect=(0,0,1,.98))
        fig.savefig(folder/'umap_comparison_600_1000_all_conditions.png',dpi=160)
        plt.close(fig)

if __name__=='__main__': main()
