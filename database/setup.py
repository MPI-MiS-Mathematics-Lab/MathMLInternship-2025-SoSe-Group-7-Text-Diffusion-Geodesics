from sentence_transformers import SentenceTransformer
import sqlite3
import json

# Initialize the embedding model
embedding_model = SentenceTransformer('thenlper/gte-small')

# Create database
conn = sqlite3.connect('database/data.db')
c = conn.cursor()

# Create tables
c.execute('''CREATE TABLE documents (
             id INTEGER PRIMARY KEY,
             text TEXT,
             url TEXT,
             svd_entropy REAL,
             topic INTEGER,
             embedding TEXT,
             domain TEXT  -- New field for document domain
             )''')

c.execute('''CREATE TABLE graph (
             id INTEGER PRIMARY KEY,
             diffusion_time REAL,
             data TEXT
             )''')

# Insert sample data
sample_graph = {
    'nodes': {
        1: {'x': 0, 'y': 0, 'svd_entropy': 0.5},
        2: {'x': 1, 'y': 1, 'svd_entropy': 0.6},
        3: {'x': 2, 'y': 2, 'svd_entropy': 0.7}
    },
    'edges': [(1, 2), (2, 3)]
}

c.execute('INSERT INTO graph (diffusion_time, data) VALUES (?, ?)',
          (1.0, json.dumps(sample_graph)))

# Add realistic embeddings for documents
sample_texts = [
    'Introduction to linear algebra',
    'Deep learning for image recognition',
    'Probability theory and statistics overview'
]

for i, text in enumerate(sample_texts, start=1):
    embedding = embedding_model.encode(text).tolist()
    c.execute('INSERT INTO documents (id, text, url, svd_entropy, topic, embedding) VALUES (?, ?, ?, ?, ?, ?)',
              (i, text, f'http://example.com/doc{i}', sample_graph['nodes'][i]['svd_entropy'], i, json.dumps(embedding)))

conn.commit()
conn.close()