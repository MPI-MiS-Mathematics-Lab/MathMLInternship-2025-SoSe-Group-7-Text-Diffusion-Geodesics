#%%
"""
Text Mining and Diffusion Analysis Pipeline
This module performs exploratory data analysis on text data using diffusion geometry
and manifold learning techniques. It processes text documents through TF-IDF vectorization,
applies SVD dimensionality reduction, and constructs diffusion maps to analyze the
geometric structure of the text corpus.
Key Features:
- TF-IDF vectorization with preprocessing
- SVD-based dimensionality reduction and variance analysis
- Non-negative matrix factorization for probabilistic interpretation
- Entropy-based similarity kernel construction
- Markov chain diffusion process modeling
- Von Neumann entropy computation across diffusion times
- Diffusion distance calculation for manifold geodesics
- K-nearest neighbors graph construction for connectivity
- Geodesic path finding using Dijkstra's algorithm
- Network visualization with spring layout and minimum spanning tree
The pipeline is designed to discover semantic relationships in text data by modeling
the corpus as a diffusion process on a manifold, enabling the computation of
meaningful distances and paths between documents based on their content similarity.
Dependencies:
- Scientific computing: numpy, scipy, pandas
- Machine learning: scikit-learn, sentence-transformers
- Text processing: gensim, datasets
- Visualization: matplotlib, networkx
- Optimization: numba (for fast distance computations)
- Dimensionality reduction: umap
- Clustering: hdbscan, bertopic
Input: CSV file containing text documents with columns ['text', 'url', 'topic']
Output: Diffusion maps, entropy measures, geodesic paths, and network visualizations
"""
import gc
import os
import pickle
import re

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy
from bertopic import BERTopic
from datasets import load_dataset
from gensim.parsing.preprocessing import preprocess_string
from hdbscan import HDBSCAN
from mpl_toolkits.mplot3d import Axes3D
from numba import njit, prange
from scipy.linalg import fractional_matrix_power
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import torch
from tqdm import tqdm
from umap import UMAP
from joblib import Parallel, delayed


#%%
#corpus_file = "wiki_ml.csv"
#corpus_file = "../data/wiki_ml_zeroshot.csv"
#df_corpus = pd.read_csv(corpus_file, index_col=0)

corpus_file = "../data/wiki_ml_zeroshot.parquet"
df_corpus = pd.read_parquet(corpus_file)


#%%
# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_df=0.5, 
    min_df=5, 
    preprocessor=preprocess_string,
    tokenizer=lambda x: x
)
tfidf_matrix = vectorizer.fit_transform(df_corpus["text"])
print(tfidf_matrix.shape)

#%%
# Apply SVD
n_markov_components = 5000
svd = TruncatedSVD(n_components=min(n_markov_components, tfidf_matrix.shape[1]-1), random_state=42)
svd_matrix = svd.fit_transform(tfidf_matrix)

# Calculate cumulative explained variance
cumulative_variance = np.cumsum(svd.explained_variance_ratio_)

# Plot cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs Number of SVD Components')
plt.grid(True)
plt.show()

print(f"Variance explained by first 100 components: {cumulative_variance[99]:.4f}")

# Plot eigenvalues in log-log scale
plt.figure(figsize=(10, 6))
plt.loglog(range(1, len(svd.singular_values_) + 1), svd.singular_values_**2)
plt.xlabel('Component Index')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues vs Component Index (Log-Log Scale)')
plt.grid(True)
plt.show()

#%%
# Create non-negative SVD matrix
svd_matrix_nonneg = np.zeros_like(svd_matrix)
epsilon = 1e-5  # Small value to avoid division by zero

for i in range(svd_matrix.shape[1]):
    column = svd_matrix[:, i]
    pos_norm = np.linalg.norm(column[column > 0])
    neg_norm = np.linalg.norm(column[column < 0])
    
    if pos_norm >= neg_norm:
        svd_matrix_nonneg[:, i] = np.maximum(column, epsilon)
    else:
        svd_matrix_nonneg[:, i] = np.maximum(-column, epsilon)

print(f"Original SVD matrix shape: {svd_matrix.shape}")
print(f"Non-negative SVD matrix shape: {svd_matrix_nonneg.shape}")
print(f"Min value in non-negative matrix: {svd_matrix_nonneg.min()}")

#%%
# Normalize columns to sum to one
svd_matrix_normalized = svd_matrix_nonneg / np.sum(svd_matrix_nonneg, axis=0)
# TODO: try to combine both directions#
# kernel matrix with pairwise cross entropy
kernel_matrix = - svd_matrix_normalized @ np.log2(svd_matrix_normalized.T)

# Get diagonal values of the kernel matrix
df_corpus["svd_entropy"] = np.diag(kernel_matrix) / np.log2(kernel_matrix.shape[0])
print("Correlation(svd_entropy, text_length) =", df_corpus["svd_entropy"].corr(df_corpus["text"].map(len)))
plt.figure(figsize=(10, 6))
plt.hist(df_corpus["svd_entropy"], bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('SVD Entropy')
plt.ylabel('Frequency')
plt.title('Histogram of SVD Entropy')
plt.grid(True, alpha=0.3)
plt.show()

#%%
@njit(parallel=True)
def compute_diffusion_distances(M_t, stationary_distribution=np.array([])):
    n = M_t.shape[0]
    diffusion_distances = np.zeros((n, n))
    for i in prange(n):
        for j in range(i + 1, n):
            diff = M_t[i] - M_t[j]
            if stationary_distribution.any():
                distance = np.sum((diff**2) / stationary_distribution)
            else:
                distance = np.sum(diff**2)
            diffusion_distances[i, j] = distance
            diffusion_distances[j, i] = distance  # Symmetric matrix
    return diffusion_distances

#%%
#sigma_values = [1.0, 1.2, 1.3, 1.4, 1.5]
#t_values = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
sigma_values = [1.2]
t_values = [1.3]

# Store results for grid plotting
grid_results = {}
P_t_matrices = {}

for sigma in tqdm(sigma_values, desc="Sigma values"):
    similarity_matrix = np.exp(-kernel_matrix / sigma**2)
    # set diagonal to zero
    np.fill_diagonal(similarity_matrix, 0)

    # Create diffusion matrix
    similarity_matrix += 1e-6  # small constant to avoid zero columns
    markov_chain = similarity_matrix / similarity_matrix.sum(axis=0)
    
    # set very small values to zero for numerical stability
    #markov_chain[np.isclose(markov_chain, 0, atol=1e-5)] = 0
    # Normalize columns again after zeroing small values
    #markov_chain = markov_chain / markov_chain.sum(axis=0)

    # Compute eigenvectors to get stationary distribution
    #eigenvalues, eigenvectors = np.linalg.eig(markov_chain)
    #stationary_distribution = np.real(eigenvectors[:, np.argmax(np.real(eigenvalues))])
    #stationary_distribution = np.sqrt(stationary_distribution)
    stationary_distribution = np.array([])

    L = markov_chain - np.eye(markov_chain.shape[0])
    
    for t in tqdm(t_values, desc=f"t values (sigma={sigma})", leave=False):
        print(f"Computing for sigma={sigma}, t={t}...")
        print("Computing matrix exponential multiplication...")
        # Move to CUDA and compute matrix exponential
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        L_torch = torch.tensor(L, dtype=torch.float64, device=device)
        P_t_torch = torch.linalg.matrix_exp(t * L_torch)
        P_t_matrix = P_t_torch.cpu().numpy()
        print("Computing diffusion distances...")
        # Compute pairwise diffusion distances
        diffusion_distances = compute_diffusion_distances(P_t_matrix, stationary_distribution)
        diffusion_distances = np.sqrt(diffusion_distances)

        print("Creating KNN graph...")
        # Create KNN graph
        k = 5
        n = diffusion_distances.shape[0]
        knn_graph = np.zeros((n, n))

        for i in range(n):
            distances = diffusion_distances[i]
            nearest_indices = np.argsort(distances)[1:k+1]
            knn_graph[i, nearest_indices] = distances[nearest_indices]
            knn_graph[nearest_indices, i] = distances[nearest_indices]

        sparse_knn_graph = csr_matrix(knn_graph)
        G = nx.Graph(sparse_knn_graph)
        pos_2d = nx.spring_layout(G, iterations=50)

        # Store results
        grid_results[(sigma, t)] = (G, pos_2d)
        
        gc.collect()
        print("Done.")

#%%
# Save P_t_matrices to disk
with open("../data/P_t_matrices.pkl", "wb") as f:
    pickle.dump(P_t_matrices, f)
print("Saved P_t_matrices to ../data/P_t_matrices.pkl")

#%%
# Create grid plot
n_rows = len(sigma_values)
n_cols = len(t_values)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))

node_colors = df_corpus["svd_entropy"].values
vmin, vmax = node_colors.min(), node_colors.max()

for i, sigma in enumerate(sigma_values):
    for j, t in enumerate(t_values):
        ax = axes[i, j]
        G, pos_2d = grid_results[(sigma, t)]
        nx.draw(G, pos_2d, node_size=10, node_color=node_colors, cmap='coolwarm', 
                edge_color='gray', alpha=0.5, ax=ax, with_labels=False)
        ax.set_title(f"σ={sigma}, t={t}", fontsize=10)
        if j == 0:
            ax.set_ylabel(f"σ={sigma}", fontsize=12)
        if i == 0:
            ax.set_xlabel(f"t={t}", fontsize=12)
            ax.xaxis.set_label_position('top')

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=vmin, vmax=vmax))
fig.colorbar(sm, ax=axes, label='SVD Entropy', shrink=0.6)
fig.suptitle("Markov Graph Spring Layouts: Parameter Grid", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("../figures/parameter_grid.png", dpi=150, bbox_inches='tight')
plt.show()

# %%
nx.draw(G, pos_2d, node_size=10, node_color=node_colors, cmap='coolwarm', 
                edge_color='gray', alpha=0.5, with_labels=False)
plt.tight_layout()
plt.show()

# %%
