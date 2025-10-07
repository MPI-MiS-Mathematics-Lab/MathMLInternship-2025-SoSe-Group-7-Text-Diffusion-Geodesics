import streamlit as st
import requests
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

# Streamlit App
st.title("Text Diffusion Geodesics")

# Cache available diffusion times in session state
if 'available_diffusion_times' not in st.session_state:
    st.session_state['available_diffusion_times'] = requests.get("http://127.0.0.1:5000/diffusion_times").json()

available_diffusion_times = st.session_state['available_diffusion_times']

# Dropdown for diffusion time
diffusion_time = st.selectbox("Select Diffusion Time", available_diffusion_times)

# Cache graph data in session state to avoid redundant reloads
if 'cached_diffusion_time' not in st.session_state or st.session_state['cached_diffusion_time'] != diffusion_time:
    response = requests.get(f"http://127.0.0.1:5000/graph?diffusion_time={diffusion_time}")
    st.session_state['graph_data'] = response.json()
    st.session_state['cached_diffusion_time'] = diffusion_time

graph_data = st.session_state['graph_data']

# Cache geodesic path in session state
if 'geodesic_path' not in st.session_state:
    geodesic_response = requests.get("http://127.0.0.1:5000/geodesic")
    if geodesic_response.status_code == 200:
        geodesic_data = geodesic_response.json()
        st.session_state['geodesic_path'] = geodesic_data.get('path', [])
    else:
        st.session_state['geodesic_path'] = []

path = st.session_state['geodesic_path']

# Create a placeholder for the graph view
graph_placeholder = st.empty()

# Commented out the interactive graph rendering logic for now
# def render_graph(path=None):
#     fig = go.Figure()
#     for edge in graph_data['edges']:
#         x_coords = [graph_data['nodes'][str(edge[0])]['x'], graph_data['nodes'][str(edge[1])]['x']]
#         y_coords = [graph_data['nodes'][str(edge[0])]['y'], graph_data['nodes'][str(edge[1])]['y']]
#         fig.add_trace(go.Scatter(x=x_coords, y=y_coords, mode='lines', line=dict(color='gray', width=0.5), showlegend=False))
#
#     for node_id, node in graph_data['nodes'].items():
#         fig.add_trace(go.Scatter(
#             x=[node['x']],
#             y=[node['y']],
#             mode='markers',
#             marker=dict(size=10, color=node['svd_entropy'], colorscale='bluered'),
#             name=f"Node {node_id}",
#             text=f"Text: {node['text'][:100]}...",
#             showlegend=False
#         ))
#
#     if path:
#         for i in range(len(path) - 1):
#             start_node = graph_data['nodes'][str(path[i])]
#             end_node = graph_data['nodes'][str(path[i + 1])]
#             fig.add_trace(go.Scatter(
#                 x=[start_node['x'], end_node['x']],
#                 y=[start_node['y'], end_node['y']],
#                 mode='lines',
#                 line=dict(color='blue', width=4, dash='solid'),
#                 name='Geodesic Path',
#                 showlegend=False
#             ))
#
#     graph_placeholder.plotly_chart(fig)

# Cache the graph plot in session state
if 'cached_graph_plot' not in st.session_state or st.session_state['cached_diffusion_time'] != diffusion_time:
    fig, ax = plt.subplots(figsize=(10, 8))
    for edge in graph_data['edges']:
        x_coords = [graph_data['nodes'][str(edge[0])]['x'], graph_data['nodes'][str(edge[1])]['x']]
        y_coords = [graph_data['nodes'][str(edge[0])]['y'], graph_data['nodes'][str(edge[1])]['y']]
        ax.plot(x_coords, y_coords, color='gray', linewidth=0.5, alpha=0.5)

    # Optimize node plotting by using a single scatter plot
    node_positions = np.array([[node['x'], node['y']] for node in graph_data['nodes'].values()])
    node_colors = [node['svd_entropy'] for node in graph_data['nodes'].values()]
    scatter = ax.scatter(node_positions[:, 0], node_positions[:, 1], s=20, c=node_colors, cmap='coolwarm', alpha=0.8)

    # Highlight the geodesic path if available
    if path:
        for i in range(len(path) - 1):
            start_node = graph_data['nodes'][str(path[i])]
            end_node = graph_data['nodes'][str(path[i + 1])]
            ax.plot(
                [start_node['x'], end_node['x']],
                [start_node['y'], end_node['y']],
                color='red', linewidth=3, alpha=1.0, label='Geodesic Path' if i == 0 else ""
            )
        # Highlight start and end nodes
        start_node = graph_data['nodes'][str(path[0])]
        end_node = graph_data['nodes'][str(path[-1])]
        ax.scatter(start_node['x'], start_node['y'], s=150, c='green', marker='o', label='Start Node', edgecolors='black', linewidth=1.5)
        ax.scatter(end_node['x'], end_node['y'], s=150, c='blue', marker='s', label='End Node', edgecolors='black', linewidth=1.5)

    ax.set_title("Static Graph View with Geodesic Path")
    ax.axis('off')
    ax.legend()
    st.session_state['cached_graph_plot'] = fig

# Display the cached plot
st.pyplot(st.session_state['cached_graph_plot'])


# Document List with Search Bar
st.subheader("Document List")
query = st.text_input("Search Query")

# Replace entropy range slider with dropdowns
entropy_values = [round(x * 0.1, 1) for x in range(0, 11)]
min_entropy = st.selectbox("Minimum SVD Entropy", entropy_values, index=0)
max_entropy = st.selectbox("Maximum SVD Entropy", entropy_values, index=len(entropy_values) - 1)

# Dropdown for number of search results to display
num_results = st.selectbox("Number of Results to Display", [5, 10, 20, 50])

# Cache search results for entropy range and query in session state
search_cache_key = f"search_{query}_{min_entropy}_{max_entropy}"
if st.button("Search"):
    if search_cache_key not in st.session_state:
        search_response = requests.get(
            f"http://127.0.0.1:5000/search?query={query}&min_entropy={min_entropy}&max_entropy={max_entropy}"
        )
        st.session_state[search_cache_key] = search_response.json()

search_results = st.session_state.get(search_cache_key, [])

# Display cached search results with the selected number of results
if search_results:
    for doc in search_results[:num_results]:  # Limit results based on dropdown selection
        header = f"Document {doc['id']}: {doc['text'][:100]}..."  # Add first 100 characters to header
        with st.expander(header):
            st.write("**Source Link:**", doc['url'])
            st.write("**SVD Entropy:**", doc['svd_entropy'])
            st.write("**Text:**", doc['text'])

# Geodesic Path
st.subheader("Geodesic Path")
start_node = st.text_input("Start Node ID")
end_node = st.text_input("End Node ID")
if st.button("Find Geodesic Path"):
    # Fetch the geodesic path
    path_response = requests.get(f"http://127.0.0.1:5000/geodesic?start={start_node}&end={end_node}")
    path_data = path_response.json()

    if 'error' in path_data:
        st.error(path_data['error'])
        path = []  # Clear path if there's an error
    else:
        st.write("**Path Length:**", path_data['length'])
        path = path_data['path']

    # Refresh graph data and re-render immediately
    response = requests.get(f"http://127.0.0.1:5000/graph?diffusion_time={diffusion_time}")
    graph_data = response.json()
    # render_graph(path=path)  # Commented out the interactive graph rendering

    # Display the path as an interactive list of documents
    st.subheader("Documents in Geodesic Path")
    for node_id in path:
        node = graph_data['nodes'][str(node_id)]
        header = f"Node {node_id}: {node['text'][:100]}..."  # Add first 100 characters to header
        with st.expander(header):
            st.write("**SVD Entropy:**", node['svd_entropy'])
            st.write("**Text:**", node['text'])
