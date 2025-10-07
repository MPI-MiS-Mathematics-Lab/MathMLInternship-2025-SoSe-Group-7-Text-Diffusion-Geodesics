# Copilot Instructions for Text Diffusion Geodesics

## Project Overview

This is a research project that applies **diffusion geometry** and **information theory** to analyze semantic relationships in text corpora. The core innovation is using cross-entropy kernels to capture asymmetric document relationships and computing geodesic paths through the resulting manifold.

## Key Architecture Patterns

### Data Pipeline Flow

1. **Data preprocessing** (`notebooks/preprocess_data.py`): Loads OpenWebMath dataset, filters for ML/math content using BERTopic zero-shot classification
2. **Main analysis** (`notebooks/diffusion_geodesics.py`): TF-IDF → SVD → entropy kernels → diffusion maps → geodesic computation
3. **Output**: LaTeX tables, visualizations, and network graphs showing semantic paths between documents

### Core Mathematical Pipeline

```python
# The fundamental transformation sequence:
TF-IDF → SVD (dimensionality reduction) → Non-negative matrix → Cross-entropy kernel → Diffusion process → Geodesic paths
```

## Critical Implementation Details

### SVD-Based Entropy Computation

- **Pattern**: Convert SVD components to non-negative by flipping negative columns based on norm comparison
- **Entropy calculation**: Use normalized columns for cross-entropy: `kernel = -P @ log2(P.T)`
- **Key insight**: Diagonal entropy values characterize document complexity on generality-specialization spectrum

### Diffusion Time Parameter

- **Critical parameter**: `t` (diffusion time) controls separation between high/low entropy documents
- **Non-integer t**: Use complex number fractional powers: `np.complex128(eigenvalues)**t` then take real part
- **Typical range**: 1.0-3.0, where higher values amplify entropy-based separation

### Graph Construction for Geodesics

- **K-NN approach**: Build sparse connectivity graph with `k=5` typically
- **Distance metric**: Diffusion distances computed with numba optimization for speed
- **Path finding**: NetworkX Dijkstra on weighted graph where edge weights are diffusion distances

## Data Conventions

### Expected Input Format

- **Primary**: Parquet files with columns `['text', 'url', 'topic']`
- **Text preprocessing**: Uses gensim preprocessing pipeline
- **Topic filtering**: BERTopic zero-shot classification with predefined ML/math topics

### Output Artifacts

- **Visualizations**: 2D spring layouts colored by SVD entropy
- **Tables**: LaTeX format geodesic paths showing entropy progression (`geodesic_path.tex`)
- **Networks**: Minimum spanning trees and KNN graphs for manifold structure

## Performance Optimization Patterns

### Memory Management

- Use `gc.collect()` after large matrix operations
- Prefer sparse matrices (`csr_matrix`) for large graphs
- Sample subsets for visualization (`n=300` typical for graph layouts)

### Computational Acceleration

- **Numba**: All distance computations use `@njit(parallel=True)` decorators
- **Matrix operations**: Leverage scipy sparse operations for large similarity matrices
- **Eigendecomposition**: Use real parts only, sort by absolute eigenvalue magnitude

## Debugging and Validation

### Common Issues

- **Connectivity**: Check `nx.number_weakly_connected_components()` - may need to increase k or take largest component
- **Entropy validation**: Correlation with text length should be positive but moderate
- **Path existence**: Use `nx.NetworkXNoPath` exception handling for disconnected graphs

### Key Diagnostic Plots

1. **Eigenvalue decay**: Log-log plot should show power-law behavior
2. **Cumulative variance**: First 100 SVD components should explain significant variance
3. **Entropy distribution**: Should be roughly normal, not bimodal
4. **Similarity matrix**: Check min/max values and histogram shape

## Research Context

This implements ideas from manifold learning applied to NLP, where documents are points on a manifold and geodesics represent optimal "learning paths" between concepts. The entropy measures capture document complexity while diffusion geometry reveals the intrinsic semantic structure.
