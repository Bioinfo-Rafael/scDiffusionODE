#!/usr/bin/env python3
"""Build a Japanese visual guide from existing outputs; standard library only, no recomputation."""
import argparse
import csv
from html import escape
import json
from pathlib import Path
from urllib.parse import quote
from figure_guide import describe_figure, describe_file


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def relative_url(path, root):
    return quote(path.relative_to(root).as_posix(), safe='/')


def link(path, root, label=None):
    return f'<a href="{relative_url(path, root)}">{escape(label or path.name)}</a>'


def card(path, root):
    title, axes, note = describe_figure(path)
    if not path.is_file():
        return f'<article class="missing"><h4>{escape(title)}</h4><p>未生成・この実行にはありません。</p><code>{escape(str(path.relative_to(root)))}</code></article>'
    url = relative_url(path, root)
    return f'''<article><h4>{escape(title)}</h4>
<a href="{url}" target="_blank" rel="noopener"><img src="{url}" loading="lazy" alt="{escape(title)}"></a>
<p>{escape(axes)}</p><p class="note">{escape(note)}</p>
<code>{escape(path.relative_to(root).as_posix())}</code></article>'''


def cards(paths, root):
    return '<div class="cards">' + ''.join(card(path, root) for path in paths) + '</div>'


def data_links(paths, root):
    existing = [p for p in paths if p.is_file()]
    return '<ul class="data">' + ''.join(f'<li>{link(p, root)} — {escape(describe_file(p))}</li>' for p in existing) + '</ul>'


def group_browser(folder, root, title):
    """Only the selected celltype's figures are visible, rather than hundreds at once."""
    groups = sorted((p for p in folder.glob('*') if p.is_dir()), key=lambda p: (p.name != 'all', p.name)) if folder.is_dir() else []
    if not groups:
        return '<p class="note">この集団別図はまだ生成されていません。</p>'
    coefficient_groups = {}
    statistics = folder.parent / 'coefficient_statistics.csv'
    if statistics.is_file():
        with statistics.open() as stream:
            rows = list(csv.DictReader(stream))
        names = sorted({r['group'] for r in rows if r.get('scope') == 'celltype'})
        coefficient_groups = {f'{i:02d}': name for i, name in enumerate(names)}
    pieces = [f'<details><summary>{escape(title)}</summary><div class="group-browser"><label>集団 <select class="group-select">']
    for p in groups:
        info = read_json(p / 'definition.json')
        label = '全細胞' if p.name == 'all' else info.get('group', coefficient_groups.get(p.name.split('_', 1)[0], p.name))
        pieces.append(f'<option value="{escape(p.name)}">{escape(str(label))}</option>')
    pieces.append('</select></label>')
    for i, p in enumerate(groups):
        hidden = ' hidden' if i else ''
        pieces.append(f'<div data-group="{escape(p.name)}"{hidden}>')
        pieces.append(cards(sorted(p.glob('*.png')), root))
        pieces.append(data_links(sorted(p.glob('*.csv')) + sorted(p.glob('*.npz')), root))
        pieces.append('</div>')
    return ''.join(pieces) + '</div></details>'


def timestep_section(step, root, first):
    coeff = step / 'coefficients'; matrices = step / 'cell_matrices'
    result = [f'<div class="timestep" data-step="{escape(step.name)}"{("" if first else " hidden")}>']
    result.append('<h2>まずこの順番で見る</h2><p>青い見出しごとに問いを1つずつ確認します。図をクリックすると元PNGを開きます。</p>')
    result.append('<h3>1. 係数は正負のどちらで、どれくらい大きい？</h3>')
    result.append(cards([coeff / 'signed_mean.png', coeff / 'mean_absolute.png'], root))
    result.append('<h3>2. 総量を除くと、8成分の割合は変わる？</h3>')
    result.append(cards([coeff / 'mean_cell_composition.png'], root))
    result.append('<h3>3. 8次元全体にcelltype差が見える？</h3>')
    result.append(cards([coeff / 'embeddings/signed/pca_celltype.png', coeff / 'embeddings/signed/umap_celltype.png'], root))
    result.append('<h3>4. celltype内より、celltype間の違いが大きい？</h3>')
    result.append(cards([matrices / 'W_eta_squared_histogram.png'], root))
    result.append(data_links([coeff / 'variance_decomposition.csv', coeff / 'variance_decomposition_within_stage.csv',
                              coeff / 'coefficient_statistics.csv'], root))
    result.append('<h3>5. 具体的にどの遺伝子の重みが大きい？</h3><p>incoming＝標的として受ける絶対重みの和、outgoing＝制御元として送る絶対重みの和です。各モデルのgenes.csvでIDから遺伝子名を確認できます。</p>')
    result.append(data_links([matrices / 'top30_effective_genes.csv', matrices / 'effective_gene_rankings.csv',
                              matrices / 'genes_varying_by_celltype.csv'], root))
    result.append('<h2>目的別の詳細</h2>')
    result.append(group_browser(coeff / 'groups', root, '平均だけでは見えない係数分布：celltypeを選ぶ'))
    result.append('<details><summary>構成比の別定義・28組の散布図・既存の発現UMAP</summary>')
    result.append(cards([coeff / 'composition_of_mean_absolute.png', coeff / 'pairs_balanced/all_28_pairs.png',
                          coeff / 'expression_umap_coefficients.png'], root))
    pairs = sorted((coeff / 'pairs_balanced').glob('a[1-8]_a[1-8].png'))
    result.append('<p>個別ペア：' + ' / '.join(link(p, root, p.stem) for p in pairs) + '</p></details>')
    for representation, label in [('signed', '符号と大きさを含むa'), ('signed_L1', '総量を除いた符号付きa'), ('absolute_L1', '符号と総量を除いた構成比')]:
        folder = coeff / 'embeddings' / representation
        result.append(f'<details><summary>PCA・UMAP・cluster分割表：{label}</summary>')
        result.append(data_links([folder / 'clustering_metrics.csv', folder / 'pca_variance.csv', folder / 'pca_loadings.csv'], root))
        result.append(cards(sorted(folder.glob('*.png')), root))
        result.append('</details>')
    result.append(group_browser(matrices / 'groups', root, '細胞依存行列の平均・std・std/mean：celltypeを選ぶ'))
    result.append('<details><summary>係数に基底・発現を掛けた寄与と、成分除去の影響</summary>')
    result.append(cards([matrices / 'component_frobenius.png', matrices / 'component_x_L2.png', matrices / 'component_ablation_L2.png'], root))
    result.append('</details><details><summary>抽出した細胞のグラフスペクトル</summary>')
    graph = matrices / 'sampled_graphs'
    result.append(cards([graph / 'ordinary_celltype_medians.png', graph / 'normalized_celltype_medians.png'], root))
    result.append(data_links([graph / 'sampled_cells.csv'], root))
    result.append('<p>細胞別・celltype別の詳細図は、ページ末尾の全ファイル検索で <code>sampled_graphs</code> を検索してください。</p></details></div>')
    return ''.join(result)


def write_report(root):
    root = Path(root).expanduser().resolve()
    if (root / 'analysis').is_dir():
        root = root / 'analysis'
    models = [root / model for model in ('matsum', 'lora') if (root / model).is_dir()]
    if not models:
        raise ValueError(f'MatSum/LoRA解析結果がありません: {root}')
    manifest = read_json(root / 'manifest.json')
    config = read_json(root / 'resolved_config.json')
    status = manifest.get('status', '未確認')
    fixture = manifest.get('mode') == 'smoke' or any(read_json(p / 'model_provenance.json').get('fixture') for p in models)
    sections = ['<header><p class="eyebrow">MatSum / LoRA · 結果の読み方</p><h1>どの図で、何を確かめるか</h1>',
                f'<p>実行状態：<strong>{escape(status)}</strong> ／ 重み：{escape(str(config.get("weights", "未確認")))} ／ 固有空間の本数q：{escape(str(config.get("spectral_dimensions", "未確認")))}</p>',
                f'<p class="path">{escape(str(root))}</p></header>']
    if fixture:
        sections.append('<div class="warning">人工データ・ランダム重みの動作確認です。学習済みD16/D20の結果ではありません。</div>')
    if status != 'complete':
        sections.append('<div class="warning">この実行は完了を確認できていません。存在する図だけを案内しています。未生成の図から結果を推測しないでください。</div>')
    sections.append('''<section class="terms"><h2>最初に：a・B・C・Wは別のもの</h2>
<dl><dt>a_k(x,t)</dt><dd>細胞ごとに変わる8個の混合係数。正負があります。</dd>
<dt>B_k</dt><dd>全細胞共通の8基底。MatSumはA_k、LoRAはU_k V_kᵀ。</dd>
<dt>C_ck = a_ck B_k</dt><dd>細胞の係数を1つの基底に掛けた成分行列。</dd>
<dt>W_c = W0 + Σ C_ck</dt><dd>8成分を合成した細胞別の重み。MatSumのW0は0。</dd></dl>
<p>「8基底間のstd」はk方向、「細胞間のstd」はc方向の計算です。tは拡散ステップで、発生段階stageとは別です。</p></section>''')
    options = ''.join(f'<option value="{p.name}">{p.name.upper()}</option>' for p in models)
    sections.append(f'<nav><label>モデル <select id="model-select">{options}</select></label><a href="#catalog">全ファイルを説明文で検索</a></nav>')
    for i, model in enumerate(models):
        sections.append(f'<section data-model="{model.name}"{("" if i == 0 else " hidden")}>')
        steps = sorted(p for p in model.glob('t[0-9]*') if p.is_dir())
        if steps:
            options = ''.join(f'<option value="{p.name}">{p.name}</option>' for p in steps)
            sections.append(f'<label>拡散ステップ <select class="step-select">{options}</select></label>')
            sections.extend(timestep_section(step, root, j == 0) for j, step in enumerate(steps))
        else:
            sections.append('<p>係数・細胞別解析の図はまだありません。</p>')
        sections.append('<details><summary>全細胞共通の8基底：重み・ばらつき・グラフ・類似度</summary><p>ここはcelltype差ではなく、モデル内部の8基底どうしを比較する部分です。</p>')
        sections.append(cards(sorted((model / 'basis').glob('*.png')), root))
        sections.append(data_links([model / 'basis/spectral_diagnostics.csv', model / 'basis/shared_matrices.npz', model / 'genes.csv'], root))
        sections.append('</details></section>')
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'index.html')
    rows = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        rows.append(f'<tr><td>{link(path, root, relative)}</td><td>{escape(describe_file(path))}</td></tr>')
    sections.append(f'''<details id="catalog"><summary>全ファイル辞典（{len(files)}件）：ファイル名・意味・celltypeで検索</summary>
<label>検索 <input id="file-search" type="search" placeholder="例：UMAP、std、Erythroid、固有、gene"></label>
<p>PNG＝見る図、CSV/CSV.GZ＝数値表、NPZ＝再解析用配列、JSON/LOG＝条件・実行記録です。</p>
<div class="table-wrap"><table><thead><tr><th>相対パス</th><th>何を描く・保存するか</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></details>''')
    sections.append('<footer>このページは既存ファイルへの案内です。解析の再実行・ファイル移動・削除・統計的結論の自動生成は行っていません。別のPCで開くときは、index.htmlと下位フォルダを一緒にコピーしてください。</footer>')
    page = '<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MatSum / LoRA 結果ガイド</title>'
    page += '<style>' + CSS + '</style><body><main>' + ''.join(sections) + '</main><script>' + JS + '</script></body></html>'
    target = root / 'index.html'
    temporary = root / '.index.html.tmp'
    temporary.write_text(page, encoding='utf-8')
    temporary.replace(target)
    return target


CSS = '''
:root{font-family:system-ui,-apple-system,"Noto Sans JP",sans-serif;color:#1c3044;background:#f1f4f7;line-height:1.65}
*{box-sizing:border-box}body{margin:0}main{max-width:1280px;margin:auto;padding:32px 24px}h1{font-size:30px;margin:8px 0}h2{font-size:22px}h3{color:#145c80;margin-top:32px;border-left:4px solid #168d9d;padding-left:12px}h4{margin:0 0 12px;font-size:17px}
.eyebrow{color:#277586;font-weight:700}.path,code{font-size:12px;overflow-wrap:anywhere}.note{color:#5c6877;font-size:13px}.warning{background:#fff0d9;border-left:5px solid #dc8d0c;padding:16px;margin:16px 0}
.terms,article,details{background:white;border:1px solid #dce3e9;border-radius:10px;padding:20px}.terms{margin:24px 0}.terms dl{display:grid;grid-template-columns:minmax(140px,220px) 1fr;gap:10px}.terms dt{font-weight:700}.terms dd{margin:0}
nav{display:flex;gap:24px;align-items:center;background:#e5eef3;padding:16px;margin:20px 0;border-radius:8px}select,input{font:inherit;padding:8px;border:1px solid #b5c6d3;border-radius:5px;background:white;max-width:100%}select{margin-left:8px}a{color:#156c96}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,430px),1fr));gap:18px;margin:18px 0}article img{width:100%;height:auto;border:1px solid #edf0f3}article p{margin:10px 0}.missing{background:#f6f7f8;color:#7a808a}details{margin:18px 0}summary{cursor:pointer;font-weight:700}summary::marker{color:#168d9d}.data{padding-left:24px}.data li{margin:10px 0}.group-browser>label{display:block;margin-top:16px}[hidden]{display:none!important}.table-wrap{overflow:auto;max-height:600px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px;border-bottom:1px solid #e3e9ee;text-align:left;vertical-align:top}td:first-child{max-width:400px;overflow-wrap:anywhere}th{position:sticky;top:0;background:#edf3f6}footer{font-size:13px;color:#647280;margin:28px 0}@media(max-width:650px){main{padding:20px 12px}nav{flex-wrap:wrap}.terms dl{display:block}.terms dd{margin-bottom:12px}}
'''

JS = '''
document.getElementById('model-select').addEventListener('change', e => {
  document.querySelectorAll('[data-model]').forEach(p => p.hidden = p.dataset.model !== e.target.value);
});
document.querySelectorAll('.step-select').forEach(select => select.addEventListener('change', e => {
  select.closest('[data-model]').querySelectorAll('[data-step]').forEach(p => p.hidden = p.dataset.step !== e.target.value);
}));
document.querySelectorAll('.group-select').forEach(select => select.addEventListener('change', e => {
  select.closest('.group-browser').querySelectorAll('[data-group]').forEach(p => p.hidden = p.dataset.group !== e.target.value);
}));
document.getElementById('file-search').addEventListener('input', e => {
  const query = e.target.value.toLocaleLowerCase();
  document.querySelectorAll('#catalog tbody tr').forEach(row => row.hidden = !row.textContent.toLocaleLowerCase().includes(query));
});
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='解析結果ディレクトリ、jobディレクトリ、または--all時のoutputs親ディレクトリ')
    parser.add_argument('--all', action='store_true', help='指定ディレクトリ内の全解析runに後付けする（再計算なし）')
    args = parser.parse_args()
    if args.all:
        roots = sorted({p.parent for name in ('matsum', 'lora') for p in args.output.rglob(name) if p.is_dir()})
        if not roots:
            parser.error('解析結果のmatsum/loraディレクトリが見つかりません')
        for root in roots:
            print(write_report(root))
    else:
        print(write_report(args.output))


if __name__ == '__main__':
    main()
