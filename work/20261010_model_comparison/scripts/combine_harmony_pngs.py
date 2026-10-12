"""Tile existing pilot Harmony PNGs without recomputing any plots."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def combine(run_root, condition_set='pilot', normal=False, allow_missing=False):
    run_root=Path(run_root)
    suite=Path(__file__).resolve().parents[1]
    conditions=sorted(p.stem for p in (suite/'configs'/condition_set).glob('*.json'))
    filename='joint_umap.png' if normal else 'joint_umap_harmony.png'
    paths=[run_root/name/'figures/generative'/filename for name in conditions]
    missing=[str(p) for p in paths if not p.is_file()]
    if missing and not allow_missing:raise FileNotFoundError('Wait until all plots finish. Missing PNGs:\n'+'\n'.join(missing))
    images=[]
    for path in paths:
        if not path.is_file():
            images.append(None);continue
        with Image.open(path) as im:images.append(im.convert('RGB'))
    # Preserve every source pixel; use the largest source dimensions as a tile.
    width=max((im.width for im in images if im is not None),default=640)
    height=max((im.height for im in images if im is not None),default=480)
    gap=20;header=40;columns=4;rows=(len(images)+columns-1)//columns
    canvas=Image.new('RGB',(columns*(width+gap)+gap,rows*(height+header+gap)+gap),'white')
    draw=ImageDraw.Draw(canvas)
    for i,(name,im) in enumerate(zip(conditions,images)):
        x=gap+(i%columns)*(width+gap);y=gap+(i//columns)*(height+header+gap)
        draw.text((x+10,y+10),name,fill='black')
        if im is None:
            draw.text((x+10,y+header+20),'UNAVAILABLE: source UMAP missing',fill='red')
        else:
            canvas.paste(im,(x+(width-im.width)//2,y+header))
    out=run_root/'comparison'/filename.replace('.png','_all.png')
    out.parent.mkdir(parents=True,exist_ok=True)
    temporary=out.with_suffix('.tmp.png');canvas.save(temporary);temporary.replace(out)
    out.with_suffix('.json').write_text(json.dumps(dict(condition_set=condition_set,
        available=[name for name,path in zip(conditions,paths) if path.is_file()],
        unavailable=[name for name,path in zip(conditions,paths) if not path.is_file()]),indent=2)+'\n')
    print(out,flush=True)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--condition-set',choices=['pilot','ode_sgd','model_expansion'],default='pilot')
    parser.add_argument('--normal',action='store_true')
    parser.add_argument('--allow-missing',action='store_true')
    args=parser.parse_args()
    combine(args.run_root,args.condition_set,args.normal,args.allow_missing)
