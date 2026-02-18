# Trident: Interactive 3D Topological Tanner Graph Visualization

This subfolder contains an advanced real-time visualization for exploring Quantum Low-Density Parity-Check (QLDPC) codes through their topological Tanner graph representations in 3D space.

## Overview

The **Trident** visualization demonstrates a different aspect of QLDPC theory compared to the existing interactive TQNN GUI - specifically focusing on the **geometric and topological constraints** that affect quantum error correction performance. While the main TQNN GUI explores semi-classical limits and classification robustness, Trident explores the fundamental graph-theoretic and topological foundations of quantum error correcting codes.

## Key Features

### 🌐 3D Topological Embedding
- **Surface Embedding**: Tanner graphs embedded on surfaces of different genus (torus, higher-genus surfaces)
- **Hyperbolic Geometry**: Real-time curvature adjustment affecting code distance bounds
- **Interactive 3D Navigation**: Rotate, zoom, and explore the graph structure from all angles
- **Surface Visualization**: Optional display of the underlying topological surface

### 📊 Real-Time Topological Analysis
- **Genus Control**: Dynamically change surface genus and observe effects on code parameters
- **Euler Characteristic**: Live calculation of χ = 2 - 2g topological invariant
- **Distance Bounds**: Real-time computation of theoretical distance limits
- **Network Statistics**: Degree distribution and connectivity analysis

### ⚡ Interactive Error Correction
- **Error Injection**: Click vertices to inject X, Z, or Y errors
- **Syndrome Propagation**: Watch belief propagation decoding in real-time
- **Performance Tracking**: Live metrics on error rates and correction efficiency
- **Auto-Correction Mode**: Continuous syndrome propagation and analysis

### 🎮 Interactive Controls
- **Genus Slider**: Adjust surface topology (genus 0-5)
- **Curvature Slider**: Control hyperbolic geometry parameters (-2.0 to 0.0)
- **Error Management**: Inject, propagate, and clear errors with button controls
- **Display Options**: Toggle surface mesh, edges, and vertex labels
- **Animation Controls**: Auto-rotation and auto-syndrome modes

## Theoretical Foundation

### Topological Quantum Error Correction
The visualization demonstrates key concepts from topological quantum error correction:

1. **Surface Codes on Higher Genus**: Shows how increasing genus affects error correction capabilities
2. **Hyperbolic Geometry**: Explores how negative curvature enables better code constructions
3. **Tanner Graph Topology**: Visualizes the bipartite graph structure relating data and check qubits
4. **Distance Bounds**: Real-time calculation of d ≤ √(n(g+1)) topological bounds

### Mathematical Framework

#### Topological Invariants
- **Euler Characteristic**: χ = 2 - 2g for surfaces of genus g
- **Distance Bounds**: Theoretical limits based on surface topology
- **Homological Structure**: Graph embeddings respecting topological constraints

#### QLDPC Properties
- **Sparse Structure**: Low-density parity check matrices with constant weight
- **Local Connectivity**: Each vertex connects to O(1) neighbors
- **Topological Protection**: Error correction via topological quantum field theory

## How to Run

### Prerequisites
```bash
pip install matplotlib numpy seaborn networkx scipy
```

### Execution
```bash
cd Code+/Trident
python interactive_3d_tanner_graph.py
```

## Usage Instructions

### Getting Started
1. **Launch**: Run the script to open the interactive 3D visualization
2. **Explore**: Use mouse to rotate and zoom the 3D graph
3. **Adjust Topology**: Use the genus slider to change surface topology
4. **Inject Errors**: Click vertices or use the "Inject Error" button
5. **Watch Correction**: Use "Propagate" to see syndrome decoding

### Advanced Features
- **Curvature Effects**: Adjust hyperbolic curvature to see geometric constraints
- **Auto Modes**: Enable auto-rotation and auto-syndrome for continuous demonstration
- **Performance Analysis**: Monitor evolution plots and network statistics
- **Surface Visualization**: Toggle surface mesh to see embedding manifold

### Understanding the Visualization

#### 3D Tanner Graph
- **Blue Circles**: Data qubits (can have errors)
- **Orange Squares**: Check vertices (syndrome detection)
- **Red Vertices**: Qubits with errors
- **Orange Checks**: Active syndrome (error detected)
- **Red Edges**: Connections involved in active syndromes

#### Performance Metrics Panel
- **Error Rate**: Fraction of vertices with errors
- **Distance Bound**: Theoretical maximum code distance
- **Topological Properties**: Genus, Euler characteristic, graph statistics

#### Evolution Plots
- **Genus History**: How topology changes over time
- **Distance Evolution**: Theoretical bounds vs. time
- **Error Rate Tracking**: Performance under noise

## Educational Value

This visualization demonstrates several advanced concepts:

### 1. Topological Quantum Error Correction
- How surface topology affects error correction capability
- Relationship between genus and code distance
- Topological protection mechanisms

### 2. Hyperbolic Geometry in Quantum Computing
- Effects of negative curvature on graph embeddings
- Geometric constraints on quantum codes
- Relationship between geometry and computational capacity

### 3. Graph Theory and Network Analysis
- Tanner graph structure and properties
- Degree distributions in LDPC codes
- Network connectivity and robustness

### 4. Real-Time Algorithm Visualization
- Belief propagation decoding process
- Syndrome propagation through graph structure
- Performance analysis under varying conditions

## Comparison with Existing Visualizations

| Aspect | Main TQNN GUI | Trident 3D Tanner |
|--------|---------------|-------------------|
| **Focus** | Semi-classical limit, classification | Graph topology, error correction |
| **Dimension** | 2D patterns, radial plots | 3D embedded graphs |
| **Interaction** | Pattern editing, noise injection | Topology control, geometric parameters |
| **Theory** | TQFT, topological robustness | QLDPC, surface codes, hyperbolic geometry |
| **Education** | Neural network emergence | Fundamental error correction principles |

## Technical Implementation

### Architecture
- **Object-Oriented Design**: Separate classes for graph logic and visualization
- **Real-Time 3D Rendering**: Efficient matplotlib 3D plotting with animation
- **Interactive Controls**: Professional widget integration with callbacks
- **Performance Optimization**: Smart redrawing and state management

### Key Algorithms
- **Surface Generation**: Parameterized surface meshes for different genus
- **Hyperbolic Embedding**: Poincaré disk model mapping to 3D
- **Belief Propagation**: Simplified message-passing for syndrome decoding
- **Topological Calculation**: Real-time computation of invariants

### Code Quality
- **Comprehensive Documentation**: Detailed docstrings and inline comments
- **Error Handling**: Robust exception handling and graceful degradation
- **Modular Design**: Clear separation of concerns and reusable components
- **Professional Standards**: Consistent with project quality expectations

## Research Connections

This visualization is inspired by recent breakthroughs in quantum error correction:

### Surface Codes and Topological Order
- Demonstrates principles from Kitaev's toric code
- Shows relationship between topology and error correction
- Visualizes concepts from topological quantum field theory

### QLDPC Code Constructions
- Based on recent linear distance scaling breakthroughs
- Incorporates hyperbolic geometry approaches
- Shows practical implementation of theoretical advances

### Geometric Approaches to Quantum Computing
- Demonstrates role of curvature in quantum codes
- Shows embedding constraints for realistic implementations
- Connects abstract theory to geometric intuition

## Future Extensions

Potential enhancements to the Trident visualization:

1. **Higher-Dimensional Embeddings**: 4D hyperbolic spaces
2. **Quantum Simulations**: Integration with actual quantum circuits
3. **Advanced Decoding**: More sophisticated error correction algorithms
4. **Multi-Surface Codes**: Code concatenation across different topologies
5. **Experimental Data**: Integration with real quantum hardware results

## Files in this Directory

- `interactive_3d_tanner_graph.py`: Main visualization script
- `README.md`: This documentation file
- `tanner_graph_examples.py`: Additional example scripts (planned)
- `topological_utilities.py`: Utility functions for topology calculations (planned)
- `plots/`: Directory for saved visualizations (auto-created)

---

*The Trident visualization represents a sophisticated exploration of topological quantum error correction, providing an interactive laboratory for understanding the deep connections between geometry, topology, and quantum computation.*