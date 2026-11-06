# Clustering from Scratch in NumPy: HAC, K-Means++, Medoids, GMMs and Spectral Cuts

Build every clustering method in Murphy's Probabilistic Machine Learning in NumPy, run each on the data behind the book's figures, and find out which one to trust. Score clusterings with purity, the Rand index, adjusted Rand and NMI, reproducing the 17-object example exactly. Grow dendrograms on real yeast gene expression and watch single link chain 235 of 310 genes into one cluster while complete link stays compact. Implement Lloyd's algorithm and K-means++ and land on the book's distortion of 2800.54. Compress an image with vector quantization and see the codebook cost half the bits. Cluster DNA reads by edit distance with K-medoids, where no mean exists. Fit Gaussian mixtures with four covariance types, prove on the data that hard EM is K-means, and pick K with BIC while validation distortion only ever asks for more clusters. Then cut graphs: spectral clustering recovers rings and spirals that K-means cannot, as long as you get the kernel width right. A final bake-off runs nine methods on eight datasets, and no method wins them all.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** pairwise_distances
- [x] **2.** contingency_table
- [x] **3.** pair_counts
- [x] **4.** cluster_scores
- [x] **5.** agglomerate
- [x] **6.** cut_tree

---

Built on Deep-ML.
