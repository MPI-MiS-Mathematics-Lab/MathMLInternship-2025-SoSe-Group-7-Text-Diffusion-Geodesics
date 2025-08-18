#%%
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
from bertopic.representation import KeyBERTInspired
from datasets import load_dataset
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
ds = load_dataset("open-web-math/open-web-math")

#%%
df_math = ds["train"].to_pandas()

#%%
# Extract domain from URLs using regex
#df_math["domain"].value_counts(ascending=False)
df_math["domain"] = df_math["url"].str.extract(r"https?://(?:www\.)?([^/]+)")

#%%
filter_strings = [
    "wikipedia.org",
    "github.io",
    "nature.com",
    #"blogspot.com",
    #"wordpress.com",
]
pattern = '|'.join(re.escape(s) for s in filter_strings)  # Create regex pattern from the list
# filter only wikipedia urls
df_corpus = df_math.loc[df_math["url"].str.contains(pattern, case=False, na=False)].copy()

# remove duplicate urls
df_corpus = df_corpus.loc[~df_corpus["url"].duplicated(keep="first")]
#df_corpus["publication_info"] = df_corpus["url"].str.extract(r"(?:publication|machine-learning-glossary-and-terms)/(.*)")
# for wikipedia, filter out non article content
df_corpus = df_corpus.loc[~df_corpus["url"].str.strip("https://").str.strip("http://").str.contains(":", case=False)]

#%%
# Filter df_corpus by a list of strings, keeping rows where at least one string is contained in the "text" column
filter_strings = [
    "linear algebra", 
    "statistics", 
    "probability theory",
    "information theory",
    "machine learning", 
    "deep learning",
    "statistical learning",
    "reinforcement learning",
    "neural network"
]
pattern = '|'.join(re.escape(s) for s in filter_strings)  # Create regex pattern from the list
df_corpus = df_corpus[df_corpus["text"].str.contains(pattern, case=False, na=False)]
df_corpus = df_corpus.loc[~df_corpus["url"].duplicated(keep="first")]
print(df_corpus.shape)

#%%
# zero shot topics
topic_model = BERTopic(
    embedding_model="thenlper/gte-small",
    min_topic_size=15,
    zeroshot_topic_list=filter_strings,
    zeroshot_min_similarity=.85,
    representation_model=KeyBERTInspired()
)
topics, _ = topic_model.fit_transform(df_corpus["text"].tolist())

# filter corpus by topics
df_corpus["topic"] = topics
df_corpus[df_corpus["topic"].isin(range(len(filter_strings)))]

#%%
# Save the filtered corpus with topics
df_corpus[df_corpus["topic"].isin(range(len(filter_strings)))].to_csv("wiki_ml_zeroshot.csv")

#%%
# Model topics with BERTopic - with caching


cache_file = "topic_model_cache.pkl"

if os.path.exists(cache_file):
    print("Loading cached topic model...")
    with open(cache_file, 'rb') as f:
        cached_data = pickle.load(f)
    
    topic_model = cached_data['topic_model']
    topics = cached_data['topics']
    probs = cached_data['probs']
    embeddings = cached_data['embeddings']

    topic_model = BERTopic(zeroshot_topic_list=topic_model.zeroshot_topic_list,
                           embedding_model=topic_model.embedding_model,
                            umap_model=topic_model.umap_model,
                            hdbscan_model=topic_model.hdbscan_model,
                            vectorizer_model=topic_model.vectorizer_model,
                            top_n_words=topic_model.top_n_words,
                            verbose=topic_model.verbose)
    
    # Add topics to dataframe
    #df_corpus["topic"] = topics
    df_corpus["topic"] = topic_model.transform(df_corpus["text"].tolist())
    
    print("Loaded cached topic model successfully!")
else:
    print("Computing topic model (this may take a while)...")
    
    # Initialize sentence transformer for embeddings
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create embeddings for the text
    embeddings = sentence_model.encode(df_corpus["text"].tolist(), show_progress_bar=True)

    # Configure UMAP for dimensionality reduction
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

    # Configure HDBSCAN for clustering
    hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom')

    # Initialize BERTopic with custom models
    topic_model = BERTopic(
        embedding_model=sentence_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=TfidfVectorizer(max_df=0.5, min_df=5, stop_words='english'),
        top_n_words=10,
        verbose=True
    )

    # Fit the model
    topics, probs = topic_model.fit_transform(df_corpus["text"].tolist(), embeddings)

    # Add topics to dataframe
    df_corpus["topic"] = topics
    
    # Cache the results
    print("Caching topic model for future use...")
    cached_data = {
        'topic_model': topic_model,
        'topics': topics,
        'probs': probs,
        'embeddings': embeddings
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)
    
    print("Topic model cached successfully!")

# Display topic information
print(f"Number of topics found: {len(set(topics)) - (1 if -1 in topics else 0)}")
print(f"Number of outliers: {sum(1 for t in topics if t == -1)}")

# Show top topics
topic_info = topic_model.get_topic_info()
print("\nTop 50 topics:")
print(topic_info.head(50))

#%%
# filter topics
selected_topics = [1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 31, 32, 33, 34, 35, 42, 43, 45, 46, 47, 49, 50, 51, 52, 54, 55, 57, 59, 60, 61, 63, 67, 68, 70, 72, 73, 74, 76, 78, 79, 80, 81, 82, 84, 85, 89, 90, 94, 96, 98, 99, 100, 103, 104, 107, 112, 113, 114, 115, 120, 122, 123, 124, 125, 126, 128, 129, 130, 131, 132, 133, 135]
df_corpus = df_corpus[df_corpus["topic"].isin(selected_topics)]

#%%
df_corpus.to_csv("wiki_ml.csv")