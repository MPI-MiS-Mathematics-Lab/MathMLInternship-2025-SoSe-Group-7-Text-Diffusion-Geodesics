import streamlit as st
import requests
import plotly.graph_objects as go

# Streamlit App
st.title("Text Diffusion Geodesics")

# Dropdown for diffusion time
diffusion_time = st.selectbox("Select Diffusion Time", [1.0, 1.5, 2.0, 2.5, 3.0])

# Graph View
st.subheader("Graph View")
response = requests.get(f"http://127.0.0.1:5000/graph?diffusion_time={diffusion_time}")
graph_data = response.json()

# Retrieve geodesic path if available
geodesic_response = requests.get("http://127.0.0.1:5000/geodesic")
if geodesic_response.status_code == 200:
    geodesic_data = geodesic_response.json()
    path = geodesic_data.get('path', [])
else:
    path = []

# Create a placeholder for the graph view
graph_placeholder = st.empty()

# Initial Graph View
def render_graph(path=None):
    fig = go.Figure()
    for edge in graph_data['edges']:
        x_coords = [graph_data['nodes'][str(edge[0])]['x'], graph_data['nodes'][str(edge[1])]['x']]
        y_coords = [graph_data['nodes'][str(edge[0])]['y'], graph_data['nodes'][str(edge[1])]['y']]
        fig.add_trace(go.Scatter(x=x_coords, y=y_coords, mode='lines', line=dict(color='gray', width=0.5), showlegend=False))

    for node_id, node in graph_data['nodes'].items():
        fig.add_trace(go.Scatter(
            x=[node['x']],
            y=[node['y']],
            mode='markers',
            marker=dict(size=10, color=node['svd_entropy'], colorscale='bluered'),  # Updated colorscale
            name=f"Node {node_id}",
            text=f"Text: {node['text'][:100]}...",
            showlegend=False  # Disable legend for nodes
        ))

    if path:
        for i in range(len(path) - 1):
            start_node = graph_data['nodes'][str(path[i])]
            end_node = graph_data['nodes'][str(path[i + 1])]
            fig.add_trace(go.Scatter(
                x=[start_node['x'], end_node['x']],
                y=[start_node['y'], end_node['y']],
                mode='lines',
                line=dict(color='blue', width=4, dash='solid'),
                name='Geodesic Path',
                showlegend=False  # Disable legend for geodesic path
            ))

    graph_placeholder.plotly_chart(fig)

# Render the initial graph
render_graph()


# Document List with Search Bar
st.subheader("Document List")
query = st.text_input("Search Query")
min_entropy, max_entropy = st.slider("SVD Entropy Range", 0.0, 1.0, (0.0, 1.0))
if st.button("Search"):
    search_response = requests.get(
        f"http://127.0.0.1:5000/search?query={query}&min_entropy={min_entropy}&max_entropy={max_entropy}"
    )
    search_results = search_response.json()
    for doc in search_results:
        header = f"Document {doc['id']}: {doc['text'][:100]}..."  # Add first 100 characters to header
        with st.expander(header):
            st.write("**Text:**", doc['text'])
            st.write("**Source Link:**", doc['url'])
            st.write("**SVD Entropy:**", doc['svd_entropy'])

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
    render_graph(path=path)

    # Display the path as an interactive list of documents
    st.subheader("Documents in Geodesic Path")
    for node_id in path:
        node = graph_data['nodes'][str(node_id)]
        header = f"Node {node_id}: {node['text'][:100]}..."  # Add first 100 characters to header
        with st.expander(header):
            st.write("**Text:**", node['text'])
            st.write("**SVD Entropy:**", node['svd_entropy'])
