"""Japanese descriptions of existing figures. No inference or re-computation."""
from pathlib import Path
import re

# title, axes/marks, interpretation boundary
GUIDE = {
    'signed_mean': ('係数の符号付き平均', '横軸＝celltype、縦軸＝平均a_k。色＝8成分。正と負を上下に積み上げます。', '正負の細胞が混在すると平均で相殺されます。棒の総高さは係数の総量ではありません。'),
    'mean_absolute': ('⚠ 係数の大きさ：絶対値を取ってから平均', '横軸＝celltype、縦軸＝mean(|a_k|)。色＝8成分。棒の高さ＝平均L1総量。', '符号を捨てています。出力への寄与率ではなく、係数の大きさです。'),
    'mean_cell_composition': ('⚠ 各細胞を総量1に揃えた平均構成比', '横軸＝celltype、縦軸＝mean(|a_k|/Σ|a|)。色＝8成分。各棒の合計＝1。', '細胞を等しく扱った構成比です。符号と総量は失われます。全ゼロ細胞は除外。'),
    'composition_of_mean_absolute': ('⚠ 絶対値平均を総量1に揃えた構成比', '横軸＝celltype、縦軸＝mean(|a_k|)/Σmean(|a|)。色＝8成分。', '係数総量の大きい細胞を強く反映します。「各細胞を先に正規化」とは異なります。'),
    'signed_histograms': ('各係数の符号付き分布', '8パネル＝a1〜a8。横軸＝係数値、縦軸＝確率密度。青＝対象集団、灰＝全細胞。', '細胞数ではなく密度を比較します。正負の混在や裾の広がりを見ます。'),
    'absolute_histograms': ('各係数の絶対値分布', '8パネル＝|a1|〜|a8|。横軸＝絶対値、縦軸＝確率密度。青＝対象集団、灰＝全細胞。', '符号を捨てた分布です。'),
    'L1_histogram': ('細胞ごとの係数総量', '横軸＝Σ_k|a_k|、縦軸＝確率密度。青＝対象集団、灰＝全細胞。', '8成分の内訳を畳み込んだ総量です。'),
    'all_28_pairs': ('8係数の全28組を一覧で見る', '対角＝各成分のhistogram。下三角＝横列のa_jと縦行のa_iの散布図。点＝抽出細胞、色＝celltype。', '表示はcelltypeごとの上限数で抽出。細胞数比率を再現した図ではありません。'),
    'expression_umap_coefficients': ('既存の発現UMAP上で係数を見る', '座標＝元データの発現UMAP、点＝細胞。8パネルの色＝それぞれの符号付きa_k。', '係数から新しく作ったUMAPとは別です。'),
    'basis_heatmaps_full': ('全細胞共通の8基底 B_k', '8パネル＝B1〜B8。縦軸＝標的遺伝子、横軸＝制御元遺伝子。赤＝正、青＝負、白＝0。', '細胞間で共有する行列です。celltype差を直接描いた図ではありません。'),
    'basis_heatmaps_zoom': ('共有基底 B_k の上位遺伝子を拡大', '縦＝標的、横＝制御元。8基底で同じ遺伝子集合・順序・色の尺度。', '重みの大きさから選んだ遺伝子の部分行列です。選択IDはzoom_gene_ids.csv。'),
    'W0_heatmap': ('共有する基準行列 W0', '縦＝標的、横＝制御元。赤＝正、青＝負。', 'LoRAの共有項です。MatSumではW0=0なので全白になります。'),
    'mean_C_full': ('係数付き各成分 C_k の集団平均', '8パネル。縦＝標的、横＝制御元。色＝mean_c(a_ck) B_kij。', '同じkでは基底の倍率と符号だけが変わります。新しい行列形状を学習した意味ではありません。'),
    'mean_C_zoom': ('係数付き各成分 C_k の集団平均：拡大', '上位遺伝子の部分行列。色＝mean_c(a_ck) B_kij、赤＝正、青＝負。', '全celltypeで同じ遺伝子と色尺度を使用しています。'),
    'mean_W_full': ('合成重み W の集団平均', '縦＝標的、横＝制御元。色＝W0+Σ_k mean_c(a_ck) B_kij。', '8成分を混ぜた結果です。符号相殺があるため平均絶対重みとは異なります。'),
    'mean_W_zoom': ('合成重み W の集団平均：拡大', '上位遺伝子の部分行列。縦＝標的、横＝制御元、赤＝正、青＝負。', 'celltype間の合成パターンを比較する図です。因果的な制御の証明ではありません。'),
    'W_eta_squared_histogram': ('Wの変動のうちcelltype別平均差が占める割合', '横軸＝η²=celltype間分散/全体分散、縦軸＝密度。1要素＝1遺伝子ペア。', '1に近い要素ほどcelltype内よりcelltype間の平均差が支配的。stage・sampleの交絡は除去していません。'),
    'component_frobenius': ('係数と基底の大きさを合わせた成分規模', '横軸＝celltype、色＝成分k、縦軸＝平均 |a_ck| ||B_k||F の積み上げ。', '成分ノルムの和は合成行列Wのノルムには一致しません。'),
    'component_x_L2': ('入力発現に対する活性化前の成分寄与', '横軸＝celltype、色＝成分k、縦軸＝平均 ||a_ck B_k x_c||2 の積み上げ。', '係数だけでなく基底と入力発現を反映します。最終出力への寄与率ではありません。'),
    'component_ablation_L2': ('成分を1つ除いたときの出力変化', '横軸＝celltype、色＝成分k、縦軸＝平均 ||f−f^(-k)||2。', '他の係数を固定した除去効果です。非線形なので8成分の効果の和は総出力に一致しません。'),
}

METRICS = {
    'low_eigenvalue_rmsd': ('小さい固有値のRMSD', '小さいほどスペクトルが近い。固有値が同じでも同じグラフとは限りません。'),
    'eigenvector_product_frobenius': ('固有ベクトル部分空間の重なり', '||Qi.T Qj||F。大きいほど近い類似度で、距離ではありません。'),
    'eigenspace_similarity': ('正規化した固有ベクトル部分空間の類似度', '0〜1、大きいほど近い。自己比較は1。'),
    'eigenspace_distance': ('固有ベクトル部分空間の距離', '0〜1、小さいほど近い。自己比較は0。q境界の縮退はspectral_diagnostics.csvで確認。'),
}


def describe_figure(path):
    path = Path(path)
    name = path.stem
    if name in GUIDE:
        return GUIDE[name]
    if name.startswith('across_basis_'):
        metric = name[len('across_basis_'):-len('_histogram')]
        return (f'8基底間の要素ごとの {metric}', f'横軸＝{metric}、縦軸＝密度。各遺伝子ペア(i,j)を固定し、8個のB_k方向で計算。',
                '細胞間のばらつきではありません。std/meanは平均ゼロ近傍を除外、負値や極端値を取り得ます。')
    if re.fullmatch(r'[CW]_(std|std_over_mean|std_over_abs_mean)_histogram', name):
        matrix, metric = name[0], name[2:-len('_histogram')]
        return (f'{matrix}の細胞間 {metric}', f'各遺伝子ペアを固定して細胞方向に計算。横軸＝{metric}、縦軸＝密度。青＝対象集団、灰＝全細胞。',
                'Cは8パネル、Wは1パネル。Cのstd/|mean|は非ゼロ要素で同じ値になる数学的構造があります。比率の除外数はratio_validity.csv。')
    if name.startswith(('pca_', 'umap_')):
        method, label = name.split('_', 1)
        rep = {'signed': '符号付きa', 'signed_L1': '符号付きa/L1総量', 'absolute_L1': '|a|/L1総量'}.get(path.parent.name, path.parent.name)
        return (f'{rep}の{method.upper()}：{label}で色分け',
                f'点＝細胞、色＝{label}。横・縦＝PC1/PC2。' if method == 'pca' else f'点＝細胞、色＝{label}。横・縦＝UMAPの2座標。',
                'PCAは中心化のみ。UMAPの見た目の分離だけで差の大きさを断定しません。原データはこのフォルダのCSV。')
    match = re.fullmatch(r'cluster_k(\d+)_(counts|within_cluster|within_celltype)', name)
    if match:
        k, variant = match.groups()
        measure = {'counts': '細胞数', 'within_cluster': '各行（cluster）の和が1になる割合', 'within_celltype': '各列（celltype）の和が1になる割合'}[variant]
        return (f'clusterとcelltypeの関係：k={k}', f'縦軸＝cluster、横軸＝celltype、色＝{measure}。',
                '元8次元でK-meansした結果。正解・不正解のconfusion matrixではなく分割表です。最初のseedを表示。')
    match = re.fullmatch(r'a([1-8])_a([1-8])', name)
    if match:
        i, j = match.groups()
        return (f'係数a{i}とa{j}の関係', f'横軸＝符号付きa{i}、縦軸＝符号付きa{j}。点＝抽出細胞、色＝celltype。', '表示用にcelltype別抽出。使用IDはscatter_cell_ids.csv。')
    if name.startswith('signed_matrix_cosine_'):
        similarity = name.endswith('similarity')
        return ('元の符号付き基底行列のcosine比較', '両軸＝B1〜B8。色＝cosine類似度（−1〜1）。' if similarity else '両軸＝B1〜B8。色＝1−cosine（0〜2）。',
                '類似度は大きいほど近く、距離は小さいほど近い。対称化前の元行列を比較します。')
    for variant in ('ordinary', 'normalized'):
        label = '通常ラプラシアン' if variant == 'ordinary' else '正規化ラプラシアン'
        if name == f'{variant}_celltype_medians':
            return (f'celltype別Wスペクトルの比較：{label}', '横軸＝小さい順の固有値番号、縦軸＝固有値。線＝各celltypeの抽出細胞の中央値。', '少数の抽出細胞による探索図です。元の符号・方向は対称化と絶対値化で失われます。')
        if name.startswith(variant + '_spectrum_'):
            return (f'8基底の固有値スペクトル：{label}', '横軸＝小さい順の固有値番号、縦軸＝固有値。8本の線＝B1〜B8。lowは先頭q本、fullは全固有値。',
                    '通常Lは重みの総量も反映。正規化Lは行列全体の定数倍に不変。')
        for metric, (title, note) in METRICS.items():
            if name == f'{variant}_{metric}':
                return (f'{title}：{label}', '両軸＝B1〜B8、色＝比較指標。先頭q本の固有値または固有ベクトルを使用。', note)
        if path.parent.name == 'sampled_graphs' and name.endswith('_' + variant):
            if name.startswith('cell_'):
                return (f'抽出1細胞の8成分C_k：{label}', '横軸＝固有値番号、縦軸＝固有値。線＝8成分。cell番号はsampled_cells.csvの行順（0始まり）。',
                        '通常Lは|a_k|倍の基底スペクトル。正規化Lはa_k≠0なら基底と同じです。')
            return (f'celltype内の抽出Wスペクトル：{label}', '横軸＝固有値番号、縦軸＝固有値。灰線＝抽出細胞、青線＝中央値、帯＝25〜75%区間。', '少数抽出です。細胞IDはsampled_cells.csv。集団の推定精度を保証する図ではありません。')
    return ('個別の出力図', '軸・色は図中のラベルを確認してください。', 'このファイル名の専用説明は未登録です。')


TABLES = {
    'coefficient_statistics.csv': '全体・celltype別の各a成分の平均、絶対値平均、分散、std、正負割合。',
    'variance_decomposition.csv': '各a成分の全体・celltype内・celltype間分散とη²。',
    'variance_decomposition_within_stage.csv': 'stage別に分けた係数の分散分解。',
    'clustering_metrics.csv': '全指定k/seedのARI、AMI、silhouette、seed間一致。',
    'top30_effective_genes.csv': 'celltype別Wの平均incoming/outgoingで上位30位以内の遺伝子。順位の列で並べ替えて比較。',
    'effective_gene_rankings.csv': '全遺伝子のcelltype別Wスコア・順位。各細胞の絶対重みを計算してから平均。',
    'component_gene_rankings.csv.gz': '各C_kについての全体・celltype別の遺伝子スコアと順位。',
    'genes_varying_by_celltype.csv': 'celltype平均outgoingスコアのstd/rangeが大きい遺伝子。',
    'mean_W_gene_rankings.csv': '|集団平均W|からの遺伝子スコア。細胞ごとの絶対重み平均とは異なる。',
    'ratio_validity.csv': 'std/meanのゼロ近傍閾値・未定義要素数。',
    'spectral_diagnostics.csv': '先頭q本の境界での縮退、ゼロ固有値・孤立点数。',
    'genes.csv': '行列の遺伝子ID順序とgene_nameなどの元注釈。',
    'sampled_cells.csv': '固有値解析に抽出した細胞IDと注釈。cell_0000は先頭行。',
    'cell_coefficients.csv.gz': '全細胞の8係数・構成比・L1総量と注釈。',
    'cell_effects.csv.gz': '全細胞の成分規模、入力への寄与、成分除去効果。',
    'matrix_moments.npz': 'Cの平均・stdを再構成できる係数平均・std・共分散・絶対値平均。基底はshared_matrices.npz。',
    'shared_matrices.npz': '共有B[8,d,d]、W0[d,d]、遺伝子ID順序。',
    'across_eight_basis_statistics.npz': '8基底方向に計算した各遺伝子ペアのmean・std・比率。',
    'W_variance_decomposition.npz': '各遺伝子ペアのWの全体・celltype内・celltype間分散とη²。',
}


def describe_file(path):
    path = Path(path)
    if path.suffix == '.png':
        return ' '.join(describe_figure(path))
    if path.name in TABLES:
        return TABLES[path.name]
    if path.suffix == '.csv' and path.with_suffix('.png').exists():
        return '対応するPNGの元数値。' + describe_figure(path.with_suffix('.png'))[0]
    if path.name == 'coordinates_and_clusters.csv.gz':
        return '細胞IDごとのPCA/UMAP座標、全k/seedのclusterラベル。'
    if path.name.startswith('pca_'):
        return 'PCAのloading（各係数から各PCへの重み）または寄与率。'
    if path.suffix == '.npz':
        return '再解析用の数値配列。グラフ隣接行列、固有値・固有ベクトルまたは抽出細胞のスペクトル。'
    if path.suffix == '.json':
        return '計算条件・細胞数・検証結果・由来の記録。図そのものではありません。'
    if path.suffix == '.log' or path.name == 'error.txt':
        return '処理の進捗またはエラー記録。'
    return '補助データ。ファイルの列名とREADMEの出力一覧を参照。'
