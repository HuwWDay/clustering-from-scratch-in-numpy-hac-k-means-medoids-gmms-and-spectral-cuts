"""
Clustering from Scratch in NumPy: HAC, K-Means++, Medoids, GMMs and Spectral Cuts scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Clustering from Scratch in NumPy (Murphy, Probabilistic Machine Learning, chapter 21).

Runs every method on the data behind the book's figures: external scores on Figure 21.1, single,
complete and average link on the real yeast expression data, K-means and K-means++, vector
quantization of a 200 x 320 image, K-medoids on DNA reads by edit distance, Gaussian mixtures with four
covariance types, hard EM as K-means, BIC vs distortion vs silhouette for choosing K, spectral
clustering on rings and spirals, Bernoulli mixtures on binarized digits, and a final bake-off."""
import base64
import time
import numpy as np
from sklearn.datasets import load_digits, load_iris, load_wine

from solution import (pairwise_distances, contingency_table, pair_counts, cluster_scores, agglomerate, cut_tree,
                      single_link_mst, same_hierarchy, max_diameter, linkage_report, assign_clusters, kmeans_lloyd,
                      kmeanspp_init, kmeans_restarts, vq_bits, block_vq, edit_distance, edit_distance_matrix,
                      kmedoids_voronoi, pam, gmm_e_step, gmm_m_step, fit_gmm, em_kmeans_limit, gmm_n_params, bic_score,
                      silhouette_samples, k_selection_curves, gaussian_affinity, graph_laplacians, spectral_clustering,
                      ncut_value, escape_probabilities, best_bipartition, fit_bernoulli_mixture, METHODS, cluster_with,
                      clustering_bakeoff)

# 310 yeast genes x 7 time points (DeRisi et al. 1997, as distributed with the book's probml-data), in thousandths
YEAST_B64 = "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"


def book_blobs():
    from sklearn.datasets import make_blobs
    X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
    X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
    X2, y2 = make_blobs(n_samples=1000, centers=np.array([[-4, 1], [-4, 3], [-4, -2]]),
                        cluster_std=np.array([0.5, 0.5, 0.5]), random_state=7)
    return np.r_[X1, X2], np.r_[y1, y2 + 2]


def rings(n_per=500, bandwidth=0.1, seed=0):
    r = np.random.RandomState(seed)
    X, y = [], []
    for k in range(2):
        theta = 2 * np.pi * r.uniform(size=n_per)
        rho = k + 1 + r.randn(n_per) * bandwidth
        X.append(np.c_[rho * np.cos(theta), rho * np.sin(theta)])
        y.append(np.full(n_per, k))
    return np.vstack(X), np.concatenate(y)


def spirals(n_per=500, bandwidth=0.1, seed=0):
    r = np.random.RandomState(seed)
    w = np.arange(1, n_per + 1) / n_per
    A = np.c_[(4 * w + 1) * np.cos(2 * np.pi * w) + r.randn(n_per) * bandwidth,
              (4 * w + 1) * np.sin(2 * np.pi * w) + r.randn(n_per) * bandwidth]
    return np.vstack([A, -A]), np.r_[np.zeros(n_per, dtype=int), np.ones(n_per, dtype=int)]


def grid25(seed=0, side=5, n=40, s=0.3):
    r = np.random.RandomState(seed)
    c = np.array([[i, j] for i in range(side) for j in range(side)], dtype=float) * 4.0
    return np.vstack([ck + s * r.randn(n, 2) for ck in c]), np.repeat(np.arange(side * side), n)


def blobs_with_outliers(n_out, seed=0):
    r = np.random.RandomState(seed)
    c = np.array([[0, 0], [6, 0], [3, 5]], dtype=float)
    X = np.vstack([c[k] + r.randn(100, 2) for k in range(3)])
    O = r.uniform(-1, 1, size=(n_out, 2))
    O = 60 * O / np.maximum(np.linalg.norm(O, axis=1, keepdims=True), 1e-12)
    return np.vstack([X, O]), np.repeat(np.arange(3), 100)


def read_families(n_families=8, n_per=20, length=16, rate=0.3, seed=0):
    r = np.random.RandomState(seed)
    alphabet = "ACGT"
    reads, y = [], []
    for f in range(n_families):
        ancestor = [alphabet[i] for i in r.randint(4, size=length)]
        for _ in range(n_per):
            s = []
            for c in ancestor:
                u = r.uniform()
                if u < rate / 3:
                    continue
                if u < 2 * rate / 3:
                    s.append(alphabet[r.randint(4)])
                    s.append(c)
                elif u < rate:
                    s.append(alphabet[r.randint(4)])
                else:
                    s.append(c)
            reads.append("".join(s))
            y.append(f)
    return reads, np.array(y)


def china_image():
    from sklearn.datasets import load_sample_image
    rgb = load_sample_image("china.jpg").astype(float)
    return np.round(rgb @ np.array([0.299, 0.587, 0.114]))[::2, ::2][:200, :320]


def yeast():
    return np.frombuffer(base64.b64decode(YEAST_B64), dtype="<i2").reshape(310, 7) / 1000.0


def k_task(seed, K_true, shape):
    r = np.random.RandomState(seed)
    while True:
        c = r.uniform(-12, 12, size=(K_true, 2))
        d = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1)) + 100 * np.eye(K_true)
        if d.min() > 5.0:
            break
    X, y = [], []
    for k in range(K_true):
        n = 400 // K_true
        Z = r.randn(n, 2)
        if shape == "elongated":
            th = r.uniform(0, np.pi)
            A = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ np.diag([1.6, 0.35])
            Z = Z @ A.T
        else:
            Z = Z * 0.7
        X.append(c[k] + Z)
        y.append(np.full(n, k))
    return np.vstack(X), np.concatenate(y)


def main():
    t0 = time.time()
    print("== Part 1. Scoring a clustering (Figure 21.1, 17 objects) ==")
    pred = np.r_[np.zeros(6, int), np.ones(6, int), np.full(5, 2)]
    true = np.r_[[0] * 5, [1], [0], [1] * 4, [2], [0] * 2, [2] * 3]
    print("  contingency table", contingency_table(pred, true).tolist(), "pairs (TP, FP, FN, TN)", pair_counts(contingency_table(pred, true)))
    print("  clustering  ", {k: round(v, 3) for k, v in cluster_scores(pred, true).items()})
    print("  singletons  ", {k: round(abs(v), 3) for k, v in cluster_scores(np.arange(17), true).items()}, "(purity 1 and ARI 0)")

    print("\n== Part 2. Hierarchical clustering ==")
    X5 = np.array([[1, 2], [2.5, 4.5], [2, 2], [4, 1.5], [4, 2.5]])
    print("  Figure 21.2 single-link heights:", agglomerate(pairwise_distances(X5, "cityblock"), "single")[:, 2].tolist())
    Xy = yeast()
    print(f"  Figure 21.4, {len(Xy)} yeast genes, correlation distance, cut into 16 clusters:")
    Dc = pairwise_distances(Xy, "correlation")
    for m, r in linkage_report(Dc, 16).items():
        print(f"    {m:9s} top height {r['top_height']:.3f}  largest cluster {r['largest']:3d}  singletons {r['singletons']:2d}  widest cluster {r['max_diameter']:.3f}")
    De = pairwise_distances(Xy)
    print("  Same tree after squaring the distances?", {m: same_hierarchy(agglomerate(De, m), agglomerate(De ** 2, m)) for m in ("single", "complete", "average")})
    Dr = pairwise_distances(rings()[0])
    t = time.time(); Z1 = agglomerate(Dr, "single"); t1 = time.time() - t
    t = time.time(); Z2 = single_link_mst(Dr); t2 = time.time() - t
    print(f"  1000 points: agglomerate {t1:.2f}s, MST route {t2:.3f}s, same heights: {bool(np.allclose(Z1[:, 2], Z2[:, 2]))}")

    print("\n== Part 3. K-means ==")
    X, y = book_blobs()
    for K in (4, 5):
        rr = np.array(kmeans_restarts(X, K, 200, np.random.default_rng(0), init="random")["all_J"])
        pp = np.array(kmeans_restarts(X, K, 200, np.random.default_rng(0))["all_J"])
        print(f"  book data K={K}, 200 starts: random mean {rr.mean():.1f} best {rr.min():.2f}, K-means++ mean {pp.mean():.1f} best {pp.min():.2f}")
    print("    local optima at K=4 (random starts):", sorted(set(np.round(kmeans_restarts(X, 4, 100, np.random.default_rng(1), init='random')['all_J'], 1).tolist())), "(book: 2800.54 and 2978.45)")
    for seed in range(3):
        Xg, yg = grid25(seed)
        rr = np.array(kmeans_restarts(Xg, 25, 100, np.random.default_rng(seed), init="random")["all_J"])
        pp = np.array(kmeans_restarts(Xg, 25, 100, np.random.default_rng(seed))["all_J"])
        print(f"  25 tight clusters, data seed {seed}: random mean {rr.mean():.0f} best {rr.min():.1f}, K-means++ mean {pp.mean():.0f} best {pp.min():.1f}")

    print("\n== Part 4. Vector quantization (200 x 320 grey image, 512,000 raw bits) ==")
    img = china_image()
    for b, K in [(1, 2), (1, 4), (1, 16), (2, 16), (2, 256), (4, 16), (4, 256), (5, 256), (8, 256)]:
        o = block_vq(img, b, K, np.random.default_rng(0))
        print(f"  {b}x{b} blocks K={K:3d}: codes {o['bpp_codes']:.3f} bpp, total {o['bpp_total']:.3f} bpp, PSNR {o['psnr']:.2f} dB")

    print("\n== Part 5. K-medoids ==")
    print("  DNA read families, 20 random starts each (no feature vectors, only edit distance):")
    for seed in range(3):
        reads, yr = read_families(seed=seed)
        D = edit_distance_matrix(reads)
        v, p = [], []
        for s in range(20):
            m0 = np.random.default_rng(s).choice(len(reads), 8, replace=False)
            _, z, h = kmedoids_voronoi(D, m0)
            v.append((h[-1], cluster_scores(z, yr)["ari"]))
            _, z, h = pam(D, m0)
            p.append((h[-1], cluster_scores(z, yr)["ari"]))
        v, p = np.array(v), np.array(p)
        hac = {m: round(cluster_scores(cut_tree(agglomerate(D, m), n_clusters=8), yr)["ari"], 2) for m in ("complete", "average")}
        print(f"    data seed {seed}: Voronoi cost {v[:, 0].mean():.1f} ARI {v[:, 1].mean():.2f}; PAM cost {p[:, 0].mean():.1f} (range {p[:, 0].min():.0f}-{p[:, 0].max():.0f}) ARI {p[:, 1].mean():.2f}; HAC {hac}")
    print("  Three blobs plus far outliers, mean ARI on the blobs over 5 seeds:")
    for n_out in (0, 1, 3, 6, 12):
        a, b = [], []
        for seed in range(5):
            Xo, yo = blobs_with_outliers(n_out, seed)
            rng = np.random.default_rng(seed)
            a.append(cluster_scores(kmeans_restarts(Xo, 3, 10, rng)["z"][:300], yo)["ari"])
            Do = pairwise_distances(Xo)
            best = min((pam(Do, rng.choice(len(Xo), 3, replace=False)) for _ in range(3)), key=lambda s: s[2][-1])
            b.append(cluster_scores(best[1][:300], yo)["ari"])
        print(f"    {n_out:2d} outliers: K-means {np.mean(a):.2f}, K-medoids {np.mean(b):.2f}")

    print("\n== Part 6. Gaussian mixtures (book data, K = 5, Figure 21.14) ==")
    for t in ("full", "tied", "diag", "spherical"):
        g = fit_gmm(X, 5, t, np.random.default_rng(0), n_init=10)
        print(f"  {t:9s} ll/N {g['ll'] / len(X):.3f}  params {gmm_n_params(5, 2, t)}  BIC {bic_score(g['ll'], 5, 2, len(X), t):.1f}  ARI {cluster_scores(g['z'], y)['ari']:.3f}")
    print(f"  K-means   ARI {cluster_scores(kmeans_restarts(X, 5, 20, np.random.default_rng(0))['z'], y)['ari']:.3f}")
    mu0 = kmeanspp_init(X, 5, np.random.default_rng(3))
    m1, z1, _ = kmeans_lloyd(X, mu0)
    m2, z2 = em_kmeans_limit(X, mu0)
    print(f"  Hard EM vs Lloyd from one start: same labels {bool(np.array_equal(z1, z2))}, largest mean difference {np.abs(m1 - m2).max():.1e}")
    for s in (5.0, 1.0, 0.5, 0.1, 0.05):
        ms, zs = em_kmeans_limit(X, mu0, sigma=s, max_iter=400)
        print(f"    soft EM sigma = {s:4.2f}: mean difference {np.abs(ms - m1).max():.4f}, labels differ {np.mean(zs != z1):.4f}, distortion {assign_clusters(X, ms)[1]:.0f}")
    print("  BIC over K = 2..8, higher is better:")
    for t in ("full", "tied", "diag", "spherical"):
        s = [bic_score(fit_gmm(X, K, t, np.random.default_rng(K), n_init=5)["ll"], K, 2, len(X), t) for K in range(2, 9)]
        print(f"    {t:9s} {np.round(s, 0).astype(int).tolist()} picks K = {int(np.argmax(s)) + 2}")

    print("\n== Part 7. Choosing K (Figures 21.11 and 21.13) ==")
    p = np.random.default_rng(0).permutation(len(X))
    c = k_selection_curves(X[p[:1000]], X[p[1000:]], range(2, 9), np.random.default_rng(0))
    print("  K              ", c["K"])
    print("  val distortion ", np.round(c["val_distortion"], 2).tolist())
    print("  BIC            ", np.round(c["bic"], 0).astype(int).tolist())
    print("  silhouette     ", np.round(c["silhouette"], 3).tolist())
    print("  picks          ", c["pick"], "(true K = 5)")
    Dx = pairwise_distances(X)
    print("  all 2000 points, silhouette at K = 3..7:", [round(float(silhouette_samples(Dx, kmeans_restarts(X, K, 20, np.random.default_rng(0))["z"]).mean()), 4) for K in range(3, 8)])
    print("  20 generated datasets per shape with true K in 2..6, hits out of 20:")
    for shape in ("spherical", "elongated"):
        hits = {"val_distortion": 0, "bic": 0, "silhouette": 0}
        for seed in range(20):
            K_true = 2 + seed % 5
            Xk, _ = k_task(seed, K_true, shape)
            q = np.random.RandomState(seed).permutation(len(Xk))
            n = len(Xk) // 2
            ck = k_selection_curves(Xk[q[:n]], Xk[q[n:]], range(2, 9), np.random.default_rng(seed), n_init=3)
            for k in hits:
                hits[k] += int(ck["pick"][k] == K_true)
        print(f"    {shape:9s} {hits}")

    print("\n== Part 8. Spectral clustering (Figure 21.19) ==")
    for name, (Xs, ys) in (("rings", rings()), ("spirals", spirals())):
        ari = lambda z: cluster_scores(z, ys)["ari"]
        Ds = pairwise_distances(Xs)
        print(f"  {name}: K-means {abs(ari(kmeans_restarts(Xs, 2, 10, np.random.default_rng(0))['z'])):.3f}, full GMM {abs(ari(fit_gmm(Xs, 2, 'full', np.random.default_rng(0), 5)['z'])):.3f}, complete link {abs(ari(cut_tree(agglomerate(Ds, 'complete'), n_clusters=2))):.3f}, single link {ari(cut_tree(single_link_mst(Ds), n_clusters=2)):.3f}")
        print("    spectral ARI by sigma:", {s: round(abs(ari(spectral_clustering(gaussian_affinity(Xs, s), 2, np.random.default_rng(0))["z"])), 3) for s in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0)})
    W = np.zeros((7, 7))
    for a, b, w in [(0, 1, 1), (0, 2, 1), (1, 2, 1), (3, 4, 1), (3, 5, 1), (4, 5, 1), (0, 3, 0.05), (4, 6, 0.02)]:
        W[a, b] = W[b, a] = w
    zc, vc = best_bipartition(W, "cut")
    zn, vn = best_bipartition(W, "ncut")
    print(f"  Two triangles + pendant: min cut {zc.tolist()} value {vc:.3f}; min Ncut {zn.tolist()} value {vn:.5f}")
    print("  Random-walk check: p(out|S) + p(S|out) =", round(sum(escape_probabilities(W, zn == 1)), 5), "= 2 x Ncut of eq 21.29 =", round(2 * vn, 5))

    print("\n== Part 9. Binary data and the bake-off ==")
    d = load_digits()
    Xb, yd = (d.data >= 8).astype(float), d.target
    Xg = d.data.astype(float)
    for K in (10, 20):
        rows = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            rows.append([cluster_scores(q, yd)[m] for q in (fit_bernoulli_mixture(Xb, K, rng, 5)["z"], kmeans_restarts(Xb, K, 5, rng)["z"], kmeans_restarts(Xg, K, 5, rng)["z"]) for m in ("ari", "nmi", "purity")])
        rows = np.array(rows)
        print(f"  digits K={K}, 5 seeds, ARI/NMI/purity: Bernoulli mixture {rows[:, :3].mean(0).round(3).tolist()} (ARI sd {rows[:, 0].std():.3f}), K-means on bits {rows[:, 3:6].mean(0).round(3).tolist()} (ARI sd {rows[:, 3].std():.3f}), K-means on grey {rows[:, 6:9].mean(0).round(3).tolist()}")
    print("  Bernoulli mixture BIC by K:", {K: round(fit_bernoulli_mixture(Xb, K, np.random.default_rng(0), 3)["ll"] - 0.5 * (K * 64 + K - 1) * np.log(len(Xb))) for K in (5, 10, 20, 30, 40)})
    iris, wine = load_iris(), load_wine()
    idx = np.random.default_rng(0).permutation(1797)[:600]
    sub = np.random.default_rng(0).permutation(len(X))[:500]
    reads, yr = read_families()
    Xr, yr2 = rings(250)
    Xs, ys = spirals(250)
    datasets = {
        "book blobs": {"X": X[sub], "y": y[sub], "K": 5},
        "rings": {"X": Xr, "y": yr2, "K": 2},
        "spirals": {"X": Xs, "y": ys, "K": 2},
        "iris": {"X": iris.data, "y": iris.target, "K": 3},
        "wine": {"X": (wine.data - wine.data.mean(0)) / wine.data.std(0), "y": wine.target, "K": 3},
        "digits": {"X": Xg[idx], "y": yd[idx], "K": 10},
        "digits, binary": {"X": Xb[idx], "y": yd[idx], "K": 10},
        "DNA reads": {"D": edit_distance_matrix(reads), "y": yr, "K": 8},
    }
    tables = [clustering_bakeoff(datasets, METHODS, seed) for seed in range(3)]
    print("  Mean ARI over 3 seeds (-- means the method cannot run on that data):")
    print(" " * 18 + " ".join(f"{m[:8]:>8s}" for m in METHODS))
    for name in datasets:
        cells = []
        for m in METHODS:
            v = [t[name][m] for t in tables]
            cells.append("      --" if v[0] is None else f"{abs(np.mean(v)) if abs(np.mean(v)) < 0.005 else np.mean(v):8.3f}")
        print(f"  {name:16s}" + " ".join(cells))
    print("  Largest seed-to-seed range per dataset:", {name: round(max(max(t[name][m] for t in tables) - min(t[name][m] for t in tables) for m in METHODS if tables[0][name][m] is not None), 3) for name in datasets})
    print(f"\ntotal {time.time() - t0:.0f}s")


main()

