#%%
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
from tqdm import tqdm
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.linalg import fractional_matrix_power

from datasets import load_dataset
import re
from numba import njit, prange
import gc
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short
import zlib
import pickle
import os
from functools import lru_cache

#%%
ds = load_dataset("open-web-math/open-web-math")

#%%
df_math = ds["train"].to_pandas()

#%%
# Extract domain from URLs using regex
#df_math["domain"].value_counts(ascending=False)
df_math["domain"] = df_math["url"].str.extract(r"https?://(?:www\.)?([^/]+)")

#%%
filter_url = "wikipedia.org"
# filter only wikipedia urls
df_corpus = df_math.loc[df_math["url"].str.contains(filter_url, case=False, na=False)].copy()
# remove duplicate urls
df_corpus = df_corpus.loc[~df_corpus["url"].duplicated(keep="first")]
df_corpus["publication_info"] = df_corpus["url"].str.extract(r"(?:publication|machine-learning-glossary-and-terms)/(.*)")
# for wikipedia, filter out non article content
df_corpus = df_corpus.loc[~df_corpus["url"].str.strip("https://").str.strip("http://").str.contains(":", case=False)]

#%%
# Filter df_corpus by a list of strings, keeping rows where at least one string is contained in the "text" column
filter_strings = [
    "linear algebra", 
    "machine learning", 
    "neural network", 
    "deep learning",
    "probability theory", 
    "information theory",
    "statistical learning"
] 
pattern = '|'.join(re.escape(s) for s in filter_strings)  # Create regex pattern from the list
df_corpus = df_corpus[df_corpus["text"].str.contains(pattern, case=False, na=False)]
df_corpus = df_corpus.loc[~df_corpus["url"].duplicated(keep="first")]
print(df_corpus.shape)

#%%
del df_math, ds
#gc.collect()

#%%
# Define preprocessing pipeline
CUSTOM_FILTERS = [
    strip_tags,
    strip_punctuation,
    strip_multiple_whitespaces,
    strip_numeric,
    remove_stopwords,
    lambda x: strip_short(x, minsize=3)
]

# Preprocess texts
print("Preprocessing texts...")
df_corpus["preprocessed_text"] = df_corpus["text"].apply(
    lambda x: " ".join(preprocess_string(x, CUSTOM_FILTERS))
)

@lru_cache(maxsize=1000)
def lempel_ziv_compress(text):
    """Compress text using Lempel-Ziv (zlib) and return compressed length."""
    return len(zlib.compress(text.encode('utf-8')))

def compression_divergence(text1, text2):
    """Compute normalized compression distance between two texts."""
    c1 = lempel_ziv_compress(text1)
    c2 = lempel_ziv_compress(text2)
    c12 = lempel_ziv_compress(text1 + text2)
    c21 = lempel_ziv_compress(text2 + text1)
    
    cd12 = (c21 - c2) / (c1 + c2)
    cd21 = (c12 - c1) / (c1 + c2)
    return max(0, cd12), max(0, cd21)  # Ensure non-negative

# Cache file path
cache_file = "/home/oliver/text-learning/nb/kernel_matrix_cache.pkl"

# Check if cached results exist
if os.path.exists(cache_file):
    print("Loading cached kernel matrix...")
    with open(cache_file, 'rb') as f:
        kernel_matrix = pickle.load(f)
    print(f"Loaded kernel matrix shape: {kernel_matrix.shape}")
else:
    print("Computing pairwise Lempel-Ziv distances...")
    n_docs = len(df_corpus)
    kernel_matrix = np.zeros((n_docs, n_docs))

    # Compute pairwise distances with progress bar
    for i in tqdm(range(n_docs)):
        for j in range(i, n_docs):
            if i == j:
                kernel_matrix[i, j] = 0
            else:
                kernel_matrix[i, j], kernel_matrix[j, i] = compression_divergence(
                    df_corpus.iloc[i]["preprocessed_text"],
                    df_corpus.iloc[j]["preprocessed_text"]
                )
    
    # Save results to cache
    print("Saving kernel matrix to cache...")
    with open(cache_file, 'wb') as f:
        pickle.dump(kernel_matrix, f)

print(f"Kernel matrix shape: {kernel_matrix.shape}")
print(f"Distance range: [{kernel_matrix.min():.4f}, {kernel_matrix.max():.4f}]")

df_corpus["lziv_complexity"] = df_corpus["preprocessed_text"].apply(lambda x: lempel_ziv_compress(x)) / df_corpus["preprocessed_text"].str.len()
df_corpus["lziv_complexity"].hist(bins=50)

#%%
sigma = 0.3  # Adjust sigma as needed for kernel smoothing
similarity_matrix = np.exp(-kernel_matrix / sigma**2)
# Set diagonal to zero to avoid self-similarity
similarity_matrix -= np.diag(np.diag(similarity_matrix))
print(similarity_matrix.min(), similarity_matrix.max())
plt.hist(similarity_matrix.flatten(), bins=100)

#%%
# create diffusion matrix
markov_chain = similarity_matrix / similarity_matrix.sum(axis=0)

# compute eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(
    markov_chain.T)
eigenvalues, eigenvectors = np.real(eigenvalues), np.real(eigenvectors)
eigv_inv = scipy.linalg.pinv(eigenvectors)

# Sort eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]
eigv_inv = eigv_inv[sorted_indices, :]

n_markov_components = 2500

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
t = 1.1
# Compute the diffused Markov matrix for time t
#M_t = eigv_inv[:n_markov_components].T @ np.diag(eigenvalues[:n_markov_components])**t @ eigenvectors[:, :n_markov_components].T

if t % 1 != 0:
    # only raise positive eigenvalues to power t
    #eigv_power = np.diag(np.where(eigenvalues[:n_markov_components] > 0, eigenvalues[:n_markov_components]**t, eigenvalues[:n_markov_components]))
    # Use fractional matrix power for non-integer t
    #eigv_power = np.abs(fractional_matrix_power(np.diag(eigenvalues[:n_markov_components]), t))
    # use real part of fractional matrix power to avoid complex numbers
    eigv_power = np.real(fractional_matrix_power(np.diag(eigenvalues[:n_markov_components]), t))
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
pos_2d = nx.spring_layout(G, k=1, iterations=50)

# Draw edges and nodes
plt.figure(figsize=(10, 8))
nx.draw_networkx_edges(G, pos_2d, alpha=0.3, width=0.5, edge_color='gray')
scatter = nx.draw_networkx_nodes(G, pos_2d, node_size=20, 
                                node_color=np.log(df_corpus["lziv_complexity"]), 
                                cmap='coolwarm', alpha=0.8)
plt.colorbar(scatter, label='Log Lempel-Ziv Complexity')
plt.title(f'2D KNN Graph (k={k})')
plt.axis('off')
plt.show()

plt.figure(figsize=(10, 8))
nx.draw_networkx_edges(G, pos_2d, alpha=0.3, width=0.5, edge_color='gray')
scatter = nx.draw_networkx_nodes(G, pos_2d, node_size=20, 
                                node_color=np.log(df_corpus["text"].str.len()), 
                                cmap='coolwarm', alpha=0.8)
plt.colorbar(scatter, label='Log Text Length')
plt.title(f'2D KNN Graph (k={k})')
plt.axis('off')
plt.show()

#%%
df_corpus[["text", "lziv_complexity"]].sample(25).sort_values(by="lziv_complexity")

#%%
# Define start and end points (you can change these)
# Find the index of the document containing "Attention Is All You Need"
df_corpus = df_corpus.reset_index(drop=True)
search_term = "Perceptron"
df_corpus["match_position"] = df_corpus["text"].str.find(search_term)

# Select the document where the match is closest to the beginning of the text
start_point = df_corpus.loc[df_corpus["match_position"] >= 0, "match_position"].idxmin()
print(f"Start point index: {start_point}, Text: {df_corpus.loc[start_point, 'text'][:100]}...")

search_term = "Transformer"
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

df_temp = df_corpus.loc[df_corpus.index[path], ["url", "text", "lziv_complexity"]].copy()
df_temp["lziv_complexity"].copy().reset_index(drop=True).plot()
df_temp

#%%
G_msp = nx.minimum_spanning_tree(G)
# Plot the minimum spanning tree using graphviz layout
plt.figure(figsize=(12, 10))
pos_graphviz = nx.nx_agraph.graphviz_layout(G_msp)

# Draw the minimum spanning tree
nx.draw_networkx_edges(G_msp, pos_graphviz, alpha=0.6, width=1, edge_color='black')
scatter = nx.draw_networkx_nodes(G_msp, pos_graphviz, node_size=10, 
                                node_color=df_corpus["svd_entropy"], 
                                cmap='coolwarm', alpha=0.8)
plt.colorbar(scatter, label='SVD Entropy')
plt.title('Minimum Spanning Tree with Graphviz Layout')
plt.axis('off')
plt.show()

# %%
