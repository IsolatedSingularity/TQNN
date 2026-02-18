"""
Topological Utilities for 3D Tanner Graph Visualization

This module provides utility functions for topological calculations and 
geometric operations used in the 3D Tanner graph visualization.

Functions:
- Surface generation for different genus
- Hyperbolic geometry calculations  
- Topological invariant computation
- Graph embedding algorithms
- Distance bound calculations
"""

import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import math


def generate_torus_surface(major_radius=2.0, minor_radius=1.0, nu=30, nv=20):
    """
    Generate a torus surface mesh
    
    Args:
        major_radius (float): Major radius of torus
        minor_radius (float): Minor radius of torus
        nu (int): Number of points in u direction
        nv (int): Number of points in v direction
    
    Returns:
        tuple: (X, Y, Z) coordinate arrays for surface mesh
    """
    u = np.linspace(0, 2*np.pi, nu)
    v = np.linspace(0, 2*np.pi, nv)
    U, V = np.meshgrid(u, v)
    
    X = (major_radius + minor_radius * np.cos(V)) * np.cos(U)
    Y = (major_radius + minor_radius * np.cos(V)) * np.sin(U)
    Z = minor_radius * np.sin(V)
    
    return X, Y, Z


def generate_genus_g_surface(genus, base_radius=2.0, nu=30, nv=20):
    """
    Generate surface of arbitrary genus using connected sum construction
    
    Args:
        genus (int): Genus of surface (0 = sphere, 1 = torus, etc.)
        base_radius (float): Base radius for construction
        nu (int): Number of points in u direction  
        nv (int): Number of points in v direction
    
    Returns:
        tuple: (X, Y, Z) coordinate arrays for surface mesh
    """
    if genus == 0:
        # Sphere
        u = np.linspace(0, 2*np.pi, nu)
        v = np.linspace(0, np.pi, nv)
        U, V = np.meshgrid(u, v)
        
        X = base_radius * np.sin(V) * np.cos(U)
        Y = base_radius * np.sin(V) * np.sin(U)
        Z = base_radius * np.cos(V)
        
    elif genus == 1:
        # Torus
        X, Y, Z = generate_torus_surface(base_radius, base_radius/2, nu, nv)
        
    else:
        # Higher genus surface (simplified approximation)
        # Use torus with genus-dependent modulation
        u = np.linspace(0, 2*np.pi, nu)
        v = np.linspace(0, 2*np.pi, nv)
        U, V = np.meshgrid(u, v)
        
        # Base torus
        major_radius = base_radius + 0.3 * genus
        minor_radius = base_radius / 2
        
        # Add genus-dependent handles
        genus_modulation = 0.2 * np.sin(genus * V) * np.cos(genus * U)
        
        X = (major_radius + minor_radius * np.cos(V) + genus_modulation) * np.cos(U)
        Y = (major_radius + minor_radius * np.cos(V) + genus_modulation) * np.sin(U)
        Z = minor_radius * np.sin(V) + 0.1 * genus * np.sin(2*V)
    
    return X, Y, Z


def hyperbolic_to_euclidean(theta, phi, curvature=-1.0):
    """
    Map hyperbolic coordinates to Euclidean 3D space using Poincaré model
    
    Args:
        theta (float): Angular coordinate
        phi (float): Radial coordinate in hyperbolic space
        curvature (float): Hyperbolic curvature parameter (negative)
    
    Returns:
        tuple: (x, y, z) Euclidean coordinates
    """
    # Poincaré disk model
    r = np.tanh(abs(curvature) * phi / 2)
    
    # Project to unit disk
    x_disk = r * np.cos(theta)
    y_disk = r * np.sin(theta)
    
    # Stereographic projection to 3D
    denom = 1 + x_disk**2 + y_disk**2
    x = 2 * x_disk / denom
    y = 2 * y_disk / denom
    z = (1 - x_disk**2 - y_disk**2) / denom
    
    return x, y, z


def calculate_euler_characteristic(genus):
    """
    Calculate Euler characteristic for surface of given genus
    
    Args:
        genus (int): Genus of surface
    
    Returns:
        int: Euler characteristic χ = 2 - 2g
    """
    return 2 - 2 * genus


def calculate_distance_bound(n_data, n_check, genus):
    """
    Calculate theoretical distance bounds for QLDPC code on given surface
    
    Args:
        n_data (int): Number of data qubits
        n_check (int): Number of check qubits
        genus (int): Genus of embedding surface
    
    Returns:
        dict: Various distance bounds
    """
    k = max(0, n_data - n_check)  # Number of logical qubits (simplified)
    
    # Quantum Singleton bound
    singleton_bound = n_data - k + 1 if k > 0 else 1
    
    # Topological bound for surface codes
    # d ≤ √(n * (g + 1)) for codes on genus g surfaces
    topological_bound = int(np.sqrt(n_data * (genus + 1)))
    
    # LDPC-specific bounds (constant degree assumption)
    ldpc_bound = int(np.sqrt(n_data / 6))  # Rough estimate for degree-6 LDPC
    
    # Hyperbolic bound (for hyperbolic surface codes)
    hyperbolic_bound = int(np.log(n_data)) if n_data > 1 else 1
    
    return {
        'singleton': singleton_bound,
        'topological': topological_bound,
        'ldpc': ldpc_bound,
        'hyperbolic': hyperbolic_bound,
        'practical': min(singleton_bound, topological_bound, ldpc_bound)
    }


def embed_graph_on_surface(graph, surface_type='torus', genus=1, curvature=-0.5):
    """
    Embed graph vertices on a surface with topological constraints
    
    Args:
        graph (networkx.Graph): Input graph to embed
        surface_type (str): Type of surface ('sphere', 'torus', 'hyperbolic')
        genus (int): Genus of surface
        curvature (float): Curvature parameter
    
    Returns:
        dict: Vertex positions {vertex: (x, y, z)}
    """
    vertices = list(graph.nodes())
    n_vertices = len(vertices)
    
    positions = {}
    
    if surface_type == 'sphere':
        # Distribute on sphere
        for i, vertex in enumerate(vertices):
            theta = 2 * np.pi * i / n_vertices
            phi = np.pi * np.sin(theta)  # Non-uniform distribution
            
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            
            positions[vertex] = (x, y, z)
            
    elif surface_type == 'torus':
        # Distribute on torus
        major_radius = 2.0
        minor_radius = 1.0
        
        for i, vertex in enumerate(vertices):
            u = 2 * np.pi * i / n_vertices
            v = np.pi * np.sin(u * genus)  # Genus-dependent distribution
            
            x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
            y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
            z = minor_radius * np.sin(v)
            
            positions[vertex] = (x, y, z)
            
    elif surface_type == 'hyperbolic':
        # Distribute in hyperbolic space
        for i, vertex in enumerate(vertices):
            theta = 2 * np.pi * i / n_vertices
            phi = 2.0 * i / n_vertices  # Radial coordinate
            
            x, y, z = hyperbolic_to_euclidean(theta, phi, curvature)
            
            # Scale for better visualization
            scale = 2.0 + 0.5 * genus
            positions[vertex] = (scale * x, scale * y, scale * z)
    
    return positions


def optimize_graph_embedding(graph, positions, iterations=100):
    """
    Optimize graph embedding using force-directed layout in 3D
    
    Args:
        graph (networkx.Graph): Graph to optimize
        positions (dict): Initial vertex positions
        iterations (int): Number of optimization iterations
    
    Returns:
        dict: Optimized vertex positions
    """
    vertices = list(graph.nodes())
    n_vertices = len(vertices)
    
    # Convert to array format
    pos_array = np.array([positions[v] for v in vertices])
    
    def energy_function(pos_flat):
        """Energy function for force-directed layout"""
        pos_3d = pos_flat.reshape((n_vertices, 3))
        energy = 0.0
        
        # Repulsive forces between all vertices
        for i in range(n_vertices):
            for j in range(i + 1, n_vertices):
                dist = np.linalg.norm(pos_3d[i] - pos_3d[j])
                if dist > 0:
                    energy += 1.0 / dist
        
        # Attractive forces between connected vertices
        for i, j in graph.edges():
            idx_i = vertices.index(i)
            idx_j = vertices.index(j)
            dist = np.linalg.norm(pos_3d[idx_i] - pos_3d[idx_j])
            energy += dist**2
        
        return energy
    
    # Optimize using scipy
    result = minimize(energy_function, pos_array.flatten(), 
                     method='L-BFGS-B', 
                     options={'maxiter': iterations})
    
    # Convert back to dictionary format
    optimized_positions = result.x.reshape((n_vertices, 3))
    return {vertices[i]: tuple(optimized_positions[i]) for i in range(n_vertices)}


def calculate_graph_topology_metrics(graph):
    """
    Calculate topological metrics for a graph
    
    Args:
        graph (networkx.Graph): Input graph
    
    Returns:
        dict: Topological metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['n_vertices'] = graph.number_of_nodes()
    metrics['n_edges'] = graph.number_of_edges()
    metrics['density'] = nx.density(graph)
    
    # Connectivity
    if nx.is_connected(graph):
        metrics['diameter'] = nx.diameter(graph)
        metrics['radius'] = nx.radius(graph)
        metrics['average_clustering'] = nx.average_clustering(graph)
    else:
        metrics['diameter'] = float('inf')
        metrics['radius'] = float('inf')
        metrics['average_clustering'] = 0.0
    
    # Degree statistics
    degrees = [graph.degree(v) for v in graph.nodes()]
    metrics['min_degree'] = min(degrees) if degrees else 0
    metrics['max_degree'] = max(degrees) if degrees else 0
    metrics['avg_degree'] = np.mean(degrees) if degrees else 0
    metrics['degree_std'] = np.std(degrees) if degrees else 0
    
    # Spectral properties (for small graphs)
    if graph.number_of_nodes() < 100:
        try:
            eigenvalues = nx.laplacian_spectrum(graph)
            metrics['algebraic_connectivity'] = sorted(eigenvalues)[1] if len(eigenvalues) > 1 else 0
            metrics['spectral_gap'] = sorted(eigenvalues)[1] - sorted(eigenvalues)[0] if len(eigenvalues) > 1 else 0
        except:
            metrics['algebraic_connectivity'] = 0
            metrics['spectral_gap'] = 0
    else:
        metrics['algebraic_connectivity'] = 0
        metrics['spectral_gap'] = 0
    
    return metrics


def generate_ldpc_tanner_graph(n_data, n_check, data_degree=3, check_degree=6):
    """
    Generate a random LDPC Tanner graph with specified degree constraints
    
    Args:
        n_data (int): Number of data vertices
        n_check (int): Number of check vertices
        data_degree (int): Target degree for data vertices
        check_degree (int): Target degree for check vertices
    
    Returns:
        networkx.Graph: LDPC Tanner graph
    """
    # Check degree constraints
    if n_data * data_degree != n_check * check_degree:
        # Adjust to satisfy constraint
        total_edges = n_data * data_degree
        check_degree = total_edges // n_check
    
    graph = nx.Graph()
    
    # Add vertices
    data_vertices = [f"d{i}" for i in range(n_data)]
    check_vertices = [f"c{i}" for i in range(n_check)]
    
    for v in data_vertices:
        graph.add_node(v, type='data')
    for v in check_vertices:
        graph.add_node(v, type='check')
    
    # Generate edges using configuration model approach
    data_stubs = []
    for i, vertex in enumerate(data_vertices):
        data_stubs.extend([vertex] * data_degree)
    
    check_stubs = []
    for i, vertex in enumerate(check_vertices):
        check_stubs.extend([vertex] * check_degree)
    
    # Randomly pair stubs to create edges
    np.random.shuffle(data_stubs)
    np.random.shuffle(check_stubs)
    
    # Ensure we have equal number of stubs
    min_stubs = min(len(data_stubs), len(check_stubs))
    data_stubs = data_stubs[:min_stubs]
    check_stubs = check_stubs[:min_stubs]
    
    # Create edges
    for data_stub, check_stub in zip(data_stubs, check_stubs):
        if not graph.has_edge(data_stub, check_stub):  # Avoid multi-edges
            graph.add_edge(data_stub, check_stub)
    
    return graph


def surface_embedding_energy(positions, graph, surface_type='torus', genus=1):
    """
    Calculate embedding energy for graph on surface
    
    Args:
        positions (dict): Vertex positions
        graph (networkx.Graph): Graph to embed
        surface_type (str): Type of surface
        genus (int): Genus of surface
    
    Returns:
        float: Embedding energy
    """
    energy = 0.0
    
    # Edge length energy
    for u, v in graph.edges():
        pos_u = np.array(positions[u])
        pos_v = np.array(positions[v])
        edge_length = np.linalg.norm(pos_u - pos_v)
        energy += edge_length**2
    
    # Surface constraint energy
    for vertex, pos in positions.items():
        x, y, z = pos
        
        if surface_type == 'torus':
            # Distance from torus surface
            R = 2.0  # Major radius
            r = 1.0  # Minor radius
            
            rho = np.sqrt(x**2 + y**2)
            surface_dist = (np.sqrt((rho - R)**2 + z**2) - r)**2
            energy += 10.0 * surface_dist  # Penalty for being off surface
            
        elif surface_type == 'sphere':
            # Distance from sphere surface
            radius = 2.0
            surface_dist = (np.linalg.norm(pos) - radius)**2
            energy += 10.0 * surface_dist
    
    return energy


# Example usage and testing functions
def test_topological_utilities():
    """Test the topological utility functions"""
    print("Testing topological utilities...")
    
    # Test surface generation
    X, Y, Z = generate_genus_g_surface(1)
    print(f"Generated torus surface: {X.shape}")
    
    # Test Euler characteristic
    chi = calculate_euler_characteristic(2)
    print(f"Euler characteristic for genus 2: {chi}")
    
    # Test distance bounds
    bounds = calculate_distance_bound(21, 12, 1)
    print(f"Distance bounds: {bounds}")
    
    # Test graph generation
    graph = generate_ldpc_tanner_graph(12, 8, 3, 4)
    print(f"Generated LDPC graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Test graph embedding
    positions = embed_graph_on_surface(graph, 'torus', 1)
    print(f"Embedded graph with {len(positions)} vertices")
    
    # Test topological metrics
    metrics = calculate_graph_topology_metrics(graph)
    print(f"Graph metrics: {metrics}")
    
    print("All tests completed successfully!")


if __name__ == "__main__":
    test_topological_utilities()