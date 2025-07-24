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
from tqdm import tqdm
from umap import UMAP


#%%
plot_markov_graph_spring_layout = False

#corpus_file = "wiki_ml.csv"
corpus_file = "wiki_ml_zeroshot.csv"

df_corpus = pd.read_csv(corpus_file, index_col=0)

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
sigma = 1.5  # Adjust sigma as needed for kernel smoothing
similarity_matrix = np.exp(-kernel_matrix / sigma**2)
# Set diagonal to zero to avoid self-similarity
similarity_matrix -= np.diag(np.diag(similarity_matrix))
print(similarity_matrix.min(), similarity_matrix.max())
plt.hist(similarity_matrix.flatten(), bins=100)
plt.xlabel('Similarity Value')
plt.ylabel('Frequency')
plt.title('Histogram of Similarity Values')
plt.grid(True, alpha=0.3)
plt.show()

#%%
# create diffusion matrix
markov_chain = similarity_matrix / similarity_matrix.sum(axis=0)

# compute eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(
    markov_chain.T)
eigenvalues, eigenvectors = np.real(eigenvalues), np.real(eigenvectors)
eigv_inv = scipy.linalg.pinv(eigenvectors)

# Sort eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]
eigv_inv = eigv_inv[sorted_indices, :]

#%%
# Plot eigenvalues to determine the number of components to use
plt.figure(figsize=(10, 6))
plt.plot(eigenvalues[1:])
plt.xlabel('Component Index')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues vs Component Index')
plt.grid(True)
plt.show()

#%%
# Compute von Neumann entropy for multiple diffusion times t
t_values = np.linspace(1, 3, 17)
von_neumann_entropies = []

for t in tqdm(t_values):
    continue
    # Compute the diffused Markov matrix for time t
    eigv_power = np.diag(np.where(eigenvalues[:n_markov_components] > 0, eigenvalues[:n_markov_components]**t, eigenvalues[:n_markov_components]))
    M_t = eigv_inv[:n_markov_components].T @ eigv_power @ eigenvectors[:, :n_markov_components].T
    M_t = M_t @ M_t.T

    plt.imshow(M_t, cmap='viridis')
    plt.show()

    # Normalize eigenvalues to form a probability distribution
    eigenvalues_t = np.linalg.eigvalsh(M_t)
    eigenvalues_t = np.real(eigenvalues_t) + 1e-10  # Ensure eigenvalues are real
    eigenvalues_t = np.abs(eigenvalues_t) / np.sum(eigenvalues_t)

    # Compute von Neumann entropy
    entropy = -np.sum(eigenvalues_t * np.log(eigenvalues_t))
    if np.isnan(entropy) or np.isinf(entropy):
        print(f"Warning: Entropy calculation resulted in NaN or Inf for t={t}. Skipping this value.")
        continue
    von_neumann_entropies.append(entropy)
    if t == t_values[-1]:
        # Plot von Neumann entropy vs diffusion time
        plt.figure(figsize=(10, 6))
        plt.plot(t_values, von_neumann_entropies, marker='o')
        plt.semilogy()
        plt.xlabel('Diffusion Time (t)')
        plt.ylabel('Von Neumann Entropy')
        plt.title('Von Neumann Entropy vs Diffusion Time')
        plt.grid(True)
        plt.show()

#%%
# find points lying on the geodesic path, i.e. interpolating on the manifold
# - got to the nearest point that is still closer to the reference point than the current point?

# - build k nearest neighbors graph with small k, large enough to connect
# - find shortest path with A*

# Calculate pairwise diffusion distances using einsum


# Choose a diffusion time t 1, 1.5, 2, 2.5
# higher diffusion time will seperate high entropy points more from low entropy points,
# results in longer paths between points with low and high entropy
t = 1.3
# Compute the diffused Markov matrix for time t
#M_t = eigv_inv[:n_markov_components].T @ np.diag(eigenvalues[:n_markov_components])**t @ eigenvectors[:, :n_markov_components].T

if t % 1 != 0:
    # only raise positive eigenvalues to power t
    #eigv_power = np.diag(np.where(eigenvalues[:n_markov_components] > 0, eigenvalues[:n_markov_components]**t, eigenvalues[:n_markov_components]))
    # Use fractional matrix power for non-integer t
    #eigv_power = np.abs(fractional_matrix_power(np.diag(eigenvalues[:n_markov_components]), t))
    # use real part of fractional matrix power to avoid complex numbers
    #eigv_power = np.real(fractional_matrix_power(np.diag(eigenvalues[:n_markov_components]), t))
    eigv_power = np.diag(np.real(np.complex128(eigenvalues[:n_markov_components])**t))
else:
    eigv_power = np.diag(eigenvalues[:n_markov_components])**t
M_t = eigv_inv[:n_markov_components].T @ eigv_power @ eigenvectors[:, :n_markov_components].T
M_t = M_t.T
print(M_t.sum(axis=1), M_t.min(), M_t.max())

#%%
x, y, z = M_t[:, 1:4][:, 0], M_t[:, 1:4][:, 1], M_t[:, 1:4][:, 2]
plt.figure(figsize=(10, 8))
plt.scatter(x, y, alpha=0.6, s=30)
plt.xlabel('First Diffusion Component')
plt.ylabel('Second Diffusion Component')
plt.title(f'2D Scatter Plot of Diffusion Map Embedding (t={t})')
plt.grid(True, alpha=0.3)
plt.show()

#%%
@njit(parallel=True)
def compute_diffusion_distances(M_t, stationary_distribution):
    n = M_t.shape[0]
    diffusion_distances = np.zeros((n, n))
    for i in prange(n):
        for j in range(i + 1, n):
            diff = M_t[i] - M_t[j]
            distance = np.sum((diff**2) / stationary_distribution)
            diffusion_distances[i, j] = distance
            diffusion_distances[j, i] = distance  # Symmetric matrix
    return diffusion_distances

# Compute pairwise diffusion distances
diffusion_distances = compute_diffusion_distances(M_t, eigenvectors[:, 0]**2)
diffusion_distances = np.sqrt(diffusion_distances)  # Take square root to get distances

# Print a sample diffusion distance
print(f"Sample diffusion distance: {diffusion_distances[0, 1]}")

plt.figure(figsize=(10, 8))
plt.imshow(diffusion_distances, cmap='viridis')
plt.colorbar(label='Diffusion Distance')
plt.title(f'Pairwise Diffusion Distances (t={t})')
plt.show()

#%%
# Create KNN graph
k = 5
n = diffusion_distances.shape[0]
knn_graph = np.zeros((n, n))

# For each point, find its k nearest neighbors
for i in range(n):
    # Get distances from point i to all other points
    distances = diffusion_distances[i]
    # Find indices of k nearest neighbors (excluding itself)
    nearest_indices = np.argsort(distances)[1:k+1]
    # Add edges to KNN graph
    knn_graph[i, nearest_indices] = distances[nearest_indices]
    knn_graph[nearest_indices, i] = distances[nearest_indices]  # Make the graph undirected

# Convert to sparse matrix for efficiency
sparse_knn_graph = csr_matrix(knn_graph)

# Create a NetworkX graph
G = nx.Graph(sparse_knn_graph)

#%%
# Create compact 2D spring layout visualization
pos_2d = nx.spring_layout(G, iterations=50, seed=42)

# Draw edges and nodes
plt.figure(figsize=(10, 8))
nx.draw_networkx_edges(G, pos_2d, alpha=0.3, width=0.5, edge_color='gray')
plt.title(f'2D KNN Graph (k={k}, t={t})')
plt.axis('off')
plt.show()

plt.figure(figsize=(10, 8))
nx.draw_networkx_edges(G, pos_2d, alpha=0.3, width=0.5, edge_color='gray')
scatter = nx.draw_networkx_nodes(G, pos_2d, node_size=20, 
                                node_color=df_corpus["svd_entropy"], 
                                cmap='coolwarm', alpha=0.8)
plt.colorbar(scatter, label='SVD Entropy')
plt.title(f'2D KNN Graph (k={k}, t={t})')
plt.axis('off')
plt.show()

plt.figure(figsize=(10, 8))
nx.draw_networkx_edges(G, pos_2d, alpha=0.3, width=0.5, edge_color='gray')
scatter = nx.draw_networkx_nodes(G, pos_2d, node_size=20, 
                                node_color=np.log(df_corpus["text"].str.len()), 
                                cmap='coolwarm', alpha=0.8)
plt.colorbar(scatter, label='Log Text Length')
plt.title(f'2D KNN Graph (k={k}, t={t})')
plt.axis('off')
plt.show()

plt.figure(figsize=(10, 8))
nx.draw_networkx_edges(G, pos_2d, alpha=0.3, width=0.5, edge_color='gray')
scatter = nx.draw_networkx_nodes(G, pos_2d, node_size=20, 
                                node_color=df_corpus["topic"],
                                alpha=0.8)
plt.title(f'2D KNN Graph (k={k}, t={t})')
plt.axis('off')
plt.show()

#%%
df_corpus[["text", "svd_entropy"]].sample(50).sort_values(by="svd_entropy")

#%%
# Define start and end points (you can change these)
# Find the index of the document containing "Attention Is All You Need"
df_corpus = df_corpus.reset_index(drop=True)
search_term = "Introducing matrices"
#search_term = "Multicollinearity"
df_corpus["match_position"] = df_corpus["text"].str.find(search_term)

# Select the document where the match is closest to the beginning of the text
start_point = df_corpus.loc[df_corpus["match_position"] >= 0, "match_position"].idxmin()
print(f"Start point index: {start_point}, Text: {df_corpus.loc[start_point, 'text'][:100]}...")

search_term = "LSTM"
df_corpus["match_position"] = df_corpus["text"].str.find(search_term)

# Select the document where the match is closest to the beginning of the text
end_point = df_corpus.loc[df_corpus["match_position"] >= 0, "match_position"].idxmin()
print(f"End point index: {end_point}, Text: {df_corpus.loc[end_point, 'text'][:100]}...")

# Find shortest path using Dijkstra's algorithm
try:
    path = nx.dijkstra_path(G, start_point, end_point, weight='weight')
    path_length = nx.dijkstra_path_length(G, start_point, end_point, weight='weight')
    print(f"Geodesic path from point {start_point} to point {end_point}:")
    print(path)
    print(f"Path length: {path_length:.4f}")
except nx.NetworkXNoPath:
    print(f"No path found between {start_point} and {end_point}. Try increasing k.")

df_temp = df_corpus.loc[df_corpus.index[path], ["url", "text", "svd_entropy"]].reset_index(drop=True)
df_temp["text"] = df_temp["text"].str[:125].str.replace("#", "").str.replace("%", "").str.replace("&", "").str.replace("$", "").str.replace("_", "").str.replace("{", "").str.replace("}", "").str.replace("\\", "").str.replace("^", "").str.replace("~", "").str.replace("\n", " ") + " ..."
df_temp["url"] = df_temp["url"].str.replace("https://", "").str.replace("http://", "").str.replace("%", "")
df_temp["svd_entropy"] = df_temp["svd_entropy"].round(4)
plt.figure(figsize=(12, 6))
df_temp["svd_entropy"].plot(marker='o', linestyle='-', color='blue', alpha=0.7)
plt.xlabel('Document Index')
plt.ylabel('SVD Entropy')
plt.title('SVD Entropy along Geodesic Path')
plt.xticks(df_temp.index, df_temp["text"].str[:50].replace('\n', ' '), rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
df_temp

#%%
df_temp[["text", "svd_entropy"]].reset_index().to_latex("geodesic_path.tex", index=False, escape=False,
                 column_format='|c|p{12cm}|c|',
                 header=["Index", "Text (first 125 chars)", "SVD Entropy"],
                 caption="Geodesic Path from Start to End Point",
                 label="tab:geodesic_path",
                 position="htbp")

#%%
# Plot the spring layout with the geodesic path highlighted
plt.figure(figsize=(12, 10))

# Draw all edges in light gray
nx.draw_networkx_edges(G, pos_2d, alpha=0.8, width=0.5, edge_color='lightgray')

# Draw all nodes colored by SVD entropy
scatter = nx.draw_networkx_nodes(G, pos_2d, node_size=20, 
                                node_color=df_corpus["svd_entropy"], 
                                cmap='coolwarm', alpha=0.6)

# Highlight the path nodes in larger size
if 'path' in locals():
    path_nodes = [pos_2d[node] for node in path]
    path_x = [pos[0] for pos in path_nodes]
    path_y = [pos[1] for pos in path_nodes]
    
    # Draw path as connected line
    plt.plot(path_x, path_y, 'r-', linewidth=3, alpha=0.8, label='Geodesic Path')
    
    # Highlight start and end points
    plt.scatter([path_x[0]], [path_y[0]], s=100, c='green', marker='o', 
               label='Start Point', edgecolors='black', linewidth=2, zorder=5)
    plt.scatter([path_x[-1]], [path_y[-1]], s=100, c='red', marker='s', 
               label='End Point', edgecolors='black', linewidth=2, zorder=5)
    
    # Highlight intermediate path nodes
    if len(path_x) > 2:
        plt.scatter(path_x[1:-1], path_y[1:-1], s=60, c='orange', marker='^', 
                   label='Path Nodes', edgecolors='black', linewidth=1, zorder=4)

plt.colorbar(scatter, label='SVD Entropy')
plt.title(f'2D KNN Graph with Geodesic Path (k={k}, t={t})')
plt.legend()
plt.axis('off')
plt.show()


#%%
k = 5
n = 300
knn_probabilities = np.zeros((diffusion_distances.shape[0], diffusion_distances.shape[0]))
sample_indices = np.random.choice(range(diffusion_distances.shape[0]), size=n, replace=False)

# For each point, find its k nearest neighbors
for i in sample_indices:
    # Get distances from point i to all other points
    distances = diffusion_distances[i]
    # Find indices of k nearest neighbors (excluding itself)
    nearest_indices = np.argsort(distances)[1:k+1]
    # Add edges to KNN graph
    knn_probabilities[i, nearest_indices] = M_t[i, nearest_indices]
    knn_probabilities[nearest_indices, i] = M_t[nearest_indices, i]  # Make the probabilities undirected

sparse_knn_probabilities = csr_matrix(knn_probabilities)

# Create a NetworkX graph from the sparse matrix
G_prob = nx.DiGraph(sparse_knn_probabilities)
# Remove nodes that are not connected to any other nodes
isolated_nodes = list(nx.isolates(G_prob))
G_prob.remove_nodes_from(isolated_nodes)
print(f"Removed {len(isolated_nodes)} isolated nodes")

# Count the number of connected components in the graph
num_components = nx.number_weakly_connected_components(G_prob)
print(f"Number of weakly connected components in G_prob: {num_components}")

# Get the sizes of each component
component_sizes = [len(c) for c in nx.weakly_connected_components(G_prob)]
print(f"Component sizes: {sorted(component_sizes, reverse=True)}")

# Keep only the largest connected component
largest_component = max(nx.weakly_connected_components(G_prob), key=len)
G_prob = G_prob.subgraph(largest_component).copy()
print(f"Kept largest component with {len(largest_component)} nodes")

# Compute the minimum spanning arborescence (directed minimum spanning tree)
G_msp = nx.maximum_spanning_arborescence(G_prob)
# Plot the minimum spanning tree using graphviz layout

pos_graphviz = nx.nx_agraph.graphviz_layout(G_msp)

#%%
plt.figure(figsize=(12, 10))
# Draw the minimum spanning tree
nx.draw_networkx_edges(G_msp, pos_graphviz, width=0.1, alpha=0.6, edge_color='black')
scatter = nx.draw_networkx_nodes(G_msp, pos_graphviz, node_size=10,
                                node_color=df_corpus.loc[list(G_msp.nodes), "svd_entropy"], 
                                cmap='coolwarm', alpha=0.8)
plt.colorbar(scatter, label='SVD Entropy')
plt.title('Minimum Spanning Tree with Graphviz Layout')
plt.axis('off')
plt.show()

#%%