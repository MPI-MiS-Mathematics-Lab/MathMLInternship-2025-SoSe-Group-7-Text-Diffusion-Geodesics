from flask import Flask, request, jsonify
import sqlite3
import json
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)

# Load the embedding model once at the start
embedding_model = SentenceTransformer('thenlper/gte-small')

# Database connection
def get_db_connection():
    conn = sqlite3.connect('database/data.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/graph', methods=['GET'])
def get_graph():
    diffusion_time = request.args.get('diffusion_time', type=float)
    conn = get_db_connection()
    graph = conn.execute('SELECT * FROM graph WHERE diffusion_time = ?', (diffusion_time,)).fetchone()
    if graph is None:
        conn.close()
        return jsonify({'error': 'No graph data found for the given diffusion time'}), 404

    graph_data = json.loads(graph['data'])

    # Fetch document data to enrich nodes with text
    documents = conn.execute('SELECT id, text FROM documents').fetchall()
    document_map = {doc['id']: doc['text'] for doc in documents}
    conn.close()

    # Add text to nodes
    for node_id, node in graph_data['nodes'].items():
        node['text'] = document_map.get(int(node_id), "")  # Default to empty string if not found

    return jsonify(graph_data)

@app.route('/document', methods=['GET'])
def get_document():
    node_id = request.args.get('node_id', type=int)
    conn = get_db_connection()
    document = conn.execute('SELECT * FROM documents WHERE id = ?', (node_id,)).fetchone()
    conn.close()
    return jsonify({
        'text': document['text'],
        'url': document['url'],
        'svd_entropy': document['svd_entropy']
    })

@app.route('/search', methods=['GET'])
def search_documents():
    query = request.args.get('query', type=str)
    min_entropy = request.args.get('min_entropy', type=float, default=0.0)
    max_entropy = request.args.get('max_entropy', type=float, default=1.0)

    # Compute query embedding
    query_embedding = embedding_model.encode(query)

    conn = get_db_connection()
    documents = conn.execute(
        'SELECT * FROM documents WHERE svd_entropy BETWEEN ? AND ?',
        (min_entropy, max_entropy)
    ).fetchall()

    # Compute cosine similarity between query and document embeddings
    results = []
    for doc in documents:
        doc_embedding = np.array(json.loads(doc['embedding']))
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        results.append({
            'id': doc['id'],
            'text': doc['text'],
            'url': doc['url'],
            'svd_entropy': doc['svd_entropy'],
            'similarity': similarity
        })

    # Sort results by similarity
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)
    conn.close()

    return jsonify(results)

@app.route('/geodesic', methods=['GET'])
def get_geodesic():
    start = request.args.get('start', type=int)
    end = request.args.get('end', type=int)
    conn = get_db_connection()
    graph = conn.execute('SELECT * FROM graph WHERE diffusion_time = 1.0').fetchone()
    conn.close()

    if graph is None:
        return jsonify({'error': 'No graph data available for geodesic computation'}), 404

    graph_data = json.loads(graph['data'])

    # Compute geodesic path using Dijkstra's algorithm
    import networkx as nx
    G = nx.Graph()
    for edge in graph_data['edges']:
        G.add_edge(int(edge[0]), int(edge[1]))  # Ensure node IDs are integers

    try:
        path = list(nx.shortest_path(G, source=start, target=end))  # Convert to list
        length = int(nx.shortest_path_length(G, source=start, target=end))  # Ensure integer
    except nx.NetworkXNoPath:
        return jsonify({'error': 'No path found between the nodes'}), 404
    except nx.NodeNotFound:
        return jsonify({'error': 'One or both nodes not found in the graph'}), 404
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

    return jsonify({'path': path, 'length': length})

@app.route('/diffusion_times', methods=['GET'])
def get_diffusion_times():
    conn = get_db_connection()
    diffusion_times = conn.execute('SELECT DISTINCT diffusion_time FROM graph').fetchall()
    conn.close()
    return jsonify([row['diffusion_time'] for row in diffusion_times])

if __name__ == '__main__':
    app.run(debug=True)