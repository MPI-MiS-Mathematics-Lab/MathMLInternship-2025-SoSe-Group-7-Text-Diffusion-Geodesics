import streamlit as st
import requests
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# Re-enable the interactive graph rendering logic with blue-red colorscale
@st.cache_data
def create_graph_figure(_graph_data, path=None):
    """Create the graph figure - cached to avoid re-rendering"""
    fig = go.Figure()
    
    # Batch all edges into a single trace for better performance
    edge_x = []
    edge_y = []
    for edge in _graph_data['edges']:
        x0, y0 = _graph_data['nodes'][str(edge[0])]['x'], _graph_data['nodes'][str(edge[0])]['y']
        x1, y1 = _graph_data['nodes'][str(edge[1])]['x'], _graph_data['nodes'][str(edge[1])]['y']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(color='gray', width=0.5),
        hoverinfo='none',
        showlegend=False
    ))

    # Convert nodes to arrays for efficient plotting
    node_ids = list(_graph_data['nodes'].keys())
    node_x = [_graph_data['nodes'][nid]['x'] for nid in node_ids]
    node_y = [_graph_data['nodes'][nid]['y'] for nid in node_ids]
    node_colors = [_graph_data['nodes'][nid]['svd_entropy'] for nid in node_ids]
    node_texts = [_graph_data['nodes'][nid]['text'][:100] + '...' for nid in node_ids]
    
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        marker=dict(size=8, color=node_colors, colorscale='RdBu_r'),
        text=node_texts,
        customdata=node_ids,
        hovertemplate='<b>Node %{customdata}</b><br>Text: %{text}<extra></extra>',
        showlegend=False
    ))

    if path:
        # Batch geodesic path into a single trace
        path_x = []
        path_y = []
        for i in range(len(path) - 1):
            start_node = _graph_data['nodes'][str(path[i])]
            end_node = _graph_data['nodes'][str(path[i + 1])]
            path_x.extend([start_node['x'], end_node['x'], None])
            path_y.extend([start_node['y'], end_node['y'], None])
        
        fig.add_trace(go.Scatter(
            x=path_x, y=path_y,
            mode='lines',
            line=dict(color='green', width=4),
            name='Geodesic Path',
            showlegend=True
        ))
        
        # Emphasize nodes on geodesic path
        path_node_x = [_graph_data['nodes'][str(nid)]['x'] for nid in path]
        path_node_y = [_graph_data['nodes'][str(nid)]['y'] for nid in path]
        path_node_texts = [_graph_data['nodes'][str(nid)]['text'][:100] + '...' for nid in path]
        path_node_entropies = [_graph_data['nodes'][str(nid)]['svd_entropy'] for nid in path]
        
        # Highlight all path nodes
        fig.add_trace(go.Scatter(
            x=path_node_x,
            y=path_node_y,
            mode='markers',
            marker=dict(size=15, color=path_node_entropies, colorscale='RdBu_r', 
                       line=dict(color='black', width=2)),
            text=path_node_texts,
            customdata=path,
            hovertemplate='<b>Path Node %{customdata}</b><br>Text: %{text}<extra></extra>',
            name='Path Nodes',
            showlegend=True
        ))
        
        # Highlight start node
        fig.add_trace(go.Scatter(
            x=[path_node_x[0]],
            y=[path_node_y[0]],
            mode='markers',
            marker=dict(size=20, color='green', symbol='star', 
                       line=dict(color='black', width=2)),
            text=[path_node_texts[0]],
            customdata=[path[0]],
            hovertemplate='<b>Start Node %{customdata}</b><br>Text: %{text}<extra></extra>',
            name='Start Node',
            showlegend=True
        ))
        
        # Highlight end node
        fig.add_trace(go.Scatter(
            x=[path_node_x[-1]],
            y=[path_node_y[-1]],
            mode='markers',
            marker=dict(size=20, color='blue', symbol='square', 
                       line=dict(color='black', width=2)),
            text=[path_node_texts[-1]],
            customdata=[path[-1]],
            hovertemplate='<b>End Node %{customdata}</b><br>Text: %{text}<extra></extra>',
            name='End Node',
            showlegend=True
        ))
    
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode='closest',
        plot_bgcolor='white'
    )

    return fig

# Create and cache the graph figure
graph_fig = create_graph_figure(graph_data, tuple(path) if path else None)
graph_placeholder.plotly_chart(graph_fig, use_container_width=True)


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
    search_response = requests.get(
        f"http://127.0.0.1:5000/search?query={query}&min_entropy={min_entropy}&max_entropy={max_entropy}"
    )
    st.session_state[search_cache_key] = search_response.json()
    st.session_state['last_search_key'] = search_cache_key

# Get search results from cache
search_results = st.session_state.get(st.session_state.get('last_search_key', search_cache_key), [])

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
        st.session_state['geodesic_path'] = []  # Clear path if there's an error
    else:
        st.write("**Path Length:**", path_data['length'])
        st.session_state['geodesic_path'] = path_data['path']  # Update cached path
        # Force re-render by clearing graph cache
        st.cache_data.clear()
        st.rerun()

# Display the geodesic path as an interactive list of documents (like search results)
if st.session_state.get('geodesic_path'):
    st.subheader("Documents in Geodesic Path")
    for idx, node_id in enumerate(st.session_state['geodesic_path']):
        node = graph_data['nodes'][str(node_id)]
        # Format similar to search results
        position_label = ""
        if idx == 0:
            position_label = " (START)"
        elif idx == len(st.session_state['geodesic_path']) - 1:
            position_label = " (END)"
        
        header = f"Step {idx + 1}{position_label} - Node {node_id}: {node['text'][:100]}..."
        with st.expander(header):
            st.write("**SVD Entropy:**", node['svd_entropy'])
            st.write("**Text:**", node['text'])

