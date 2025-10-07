# data foundation

for the given document dataset (corpus), the following metrics will be given precomputed:

- gte-small embeddings for semantic search/retrieval ability (i.e. indexing)
- position and connectivity in the final spring embedding of the k nearest neighbor graph for different diffusion times t

# user interface (UI)

- document window: (show text content, source link, svd entropy, connected documents; buttons: select as start node, select as end node)
- interactive graph view: show the graph based on the spring embedding and connectivity; click on document -> show the selected document in the document window
- document list with search bar: retrieve based on embeddings, also give a slider to filter by a range of svd entropy (min and max value); click on document -> show the selected document in the document window
- dropdown menu to select from the available diffusion times t -> with changing t the corresponding spring graph layout needs to be loaded
- geodesics window: button to find the shortest path between start and end node, will show the list of documents in order of the geodesic path: click on document -> show the selected document in the document window

# tech stack

- evaluate yourself which frontend/backend frameworks or database are good fits for solving the stated problems, to solve as much as possible in pyhon is preferred
- this mainly aims to make the proof of concept of the idea explorable

# documentation

- clearly document the code with inline comments, assume that someone without experience in web developement has to understand the code
