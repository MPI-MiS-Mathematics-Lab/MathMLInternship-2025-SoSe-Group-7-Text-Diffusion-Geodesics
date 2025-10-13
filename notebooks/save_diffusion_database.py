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

from sentence_transformers import SentenceTransformer
import sqlite3
import json


#%%
plot_markov_graph_spring_layout = False

#corpus_file = "wiki_ml.csv"
#corpus_file = "../data/wiki_ml_zeroshot.csv"
#df_corpus = pd.read_csv(corpus_file, index_col=0)

corpus_file = "../data/wiki_ml_zeroshot.parquet"
df_corpus = pd.read_parquet(corpus_file)

#%%
# 3000 characters minimum. ca. 1 a4 page
df_corpus = df_corpus.loc[(~df_corpus["url"].duplicated(keep="first")) & (df_corpus["text"].str.len() >= 3000)]
print(df_corpus.shape)

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
sigma = 1.2  # Adjust sigma as needed for kernel smoothing
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
#
# --- compute embeddings, save corpus in database ---
#
# Initialize the embedding model
embedding_model = SentenceTransformer('thenlper/gte-small')

# Create database
conn = sqlite3.connect('../database/data.db')
c = conn.cursor()

# Create tables
c.execute('''CREATE TABLE documents (
             id INTEGER PRIMARY KEY,
             text TEXT,
             url TEXT,
             domain TEXT,
             svd_entropy REAL,
             topic INTEGER,
             embedding TEXT
             )''')

for i in tqdm(range(len(df_corpus))):
    text = df_corpus['text'].iloc[i]
    embedding = embedding_model.encode(text).tolist()
    c.execute('INSERT INTO documents (id, text, url, svd_entropy, topic, embedding) VALUES (?, ?, ?, ?, ?, ?)',
              (i,
               text, 
               df_corpus["url"].iloc[i], 
               df_corpus["svd_entropy"].iloc[i], 
               df_corpus["topic"].iloc[i], 
               json.dumps(embedding)))

#%%
#
# --- Compute diffusion distances for a specific diffusion times t ---
#

c.execute('''CREATE TABLE graph (
             id INTEGER PRIMARY KEY,
             diffusion_time REAL,
             data TEXT
             )''')

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

t_list = np.arange(1, 2, 0.1)

for t in tqdm(t_list):

    if t % 1 != 0:
        eigv_power = np.diag(np.real(np.complex128(eigenvalues[:n_markov_components])**t))
    else:
        eigv_power = np.diag(eigenvalues[:n_markov_components])**t
    M_t = eigv_inv[:n_markov_components].T @ eigv_power @ eigenvectors[:, :n_markov_components].T
    M_t = M_t.T
    print(M_t.sum(axis=1), M_t.min(), M_t.max())

    # Compute pairwise diffusion distances
    diffusion_distances = compute_diffusion_distances(M_t, eigenvectors[:, 0]**2)
    diffusion_distances = np.sqrt(diffusion_distances)  # Take square root to get distances

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

    # Create compact 2D spring layout visualization
    pos_2d = nx.spring_layout(G, iterations=200)

    # Create graph data dictionary
    graph_data = {
        'nodes': {},
        'edges': []
    }

    # Add nodes with positions and SVD entropy values
    for i, (node_id, (x, y)) in enumerate(pos_2d.items()):
        graph_data['nodes'][node_id] = {
            'x': float(x),
            'y': float(y),
            'svd_entropy': float(df_corpus['svd_entropy'].iloc[node_id])
        }

    # Add edges from the sparse KNN graph
    rows, cols = sparse_knn_graph.nonzero()
    for i, j in zip(rows, cols):
        if i < j:  # Avoid duplicate edges since graph is undirected
            graph_data['edges'].append((int(i), int(j)))


    # insert graph data into database
    c.execute('INSERT INTO graph (diffusion_time, data) VALUES (?, ?)',
          (t, json.dumps(graph_data)))


#%%
# commit changes and close the connection
conn.commit()
conn.close()

# %%
