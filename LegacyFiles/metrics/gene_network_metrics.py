import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import igraph as ig

# 1) adata → 重み付きグラフ（頂点に embedding を保存）
def make_weighted_graph_from_adata(adata, n_components=50):
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X   # (cells × genes)
    genes = np.array(adata.var_names)
    Xt = X.T  # (genes × cells)

    pca = PCA(n_components=n_components)
    emb = pca.fit_transform(Xt)   # (genes × 50)

    sim = cosine_similarity(emb)  # 類似度を重みとして使う（[-1,1]）

    g = ig.Graph.Weighted_Adjacency(sim.tolist(), mode="undirected", attr="weight", loops=False)
    g.vs["name"] = genes.tolist()
    # ← これが無いと後で KeyError になる
    g.vs["embedding"] = [e.tolist() for e in emb]

    return g

def adjacency_correlation(g1, g2):
    # 共通遺伝子でサブグラフ化＆順序合わせ
    names1 = np.array(g1.vs["name"]); names2 = np.array(g2.vs["name"])
    common = np.intersect1d(names1, names2)
    if common.size == 0:
        raise ValueError("共通遺伝子がありません。")
    g1s = g1.subgraph([g1.vs.find(name=n).index for n in common])
    g2s = g2.subgraph([g2.vs.find(name=n).index for n in common])

    A1 = np.array(g1s.get_adjacency(attribute="weight").data, dtype=float)
    A2 = np.array(g2s.get_adjacency(attribute="weight").data, dtype=float)
    iu = np.triu_indices(len(common), k=1)
    v1 = A1[iu]; v2 = A2[iu]
    # 相関（スパースなら0クリップや閾値後が安定）
    corr = np.corrcoef(v1, v2)[0,1]
    return float(corr)