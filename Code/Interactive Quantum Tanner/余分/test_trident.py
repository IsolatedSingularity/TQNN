"""
Test script for Trident 3D Tanner Graph Visualization

This script tests the import functionality and basic operations of the 
Trident visualization components. Run this first to ensure everything
is working correctly before running the main visualization.
"""

import sys
import os
import traceback

def test_basic_imports():
    """Test basic Python imports"""
    print("Testing basic imports...")
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        print("✓ Matplotlib with 3D support imported successfully")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✓ Seaborn imported successfully")
    except ImportError as e:
        print(f"✗ Seaborn import failed: {e}")
        return False
    
    try:
        import networkx as nx
        print("✓ NetworkX imported successfully")
    except ImportError as e:
        print(f"✗ NetworkX import failed: {e}")
        return False
    
    try:
        from scipy.spatial.distance import pdist, squareform
        from scipy.optimize import minimize
        print("✓ SciPy imported successfully")
    except ImportError as e:
        print(f"✗ SciPy import failed: {e}")
        return False
    
    return True


def test_topological_utilities():
    """Test topological utilities module"""
    print("\nTesting topological utilities...")
    
    try:
        from topological_utilities import (
            generate_genus_g_surface,
            calculate_euler_characteristic,
            calculate_distance_bound,
            embed_graph_on_surface,
            generate_ldpc_tanner_graph
        )
        print("✓ Topological utilities imported successfully")
    except ImportError as e:
        print(f"✗ Topological utilities import failed: {e}")
        return False
    
    try:
        # Test surface generation
        X, Y, Z = generate_genus_g_surface(1)
        print(f"✓ Surface generation works (shape: {X.shape})")
    except Exception as e:
        print(f"✗ Surface generation failed: {e}")
        return False
    
    try:
        # Test Euler characteristic
        chi = calculate_euler_characteristic(2)
        expected = 2 - 2*2  # Should be -2
        assert chi == expected, f"Expected {expected}, got {chi}"
        print("✓ Euler characteristic calculation works")
    except Exception as e:
        print(f"✗ Euler characteristic calculation failed: {e}")
        return False
    
    try:
        # Test distance bounds
        bounds = calculate_distance_bound(21, 12, 1)
        assert isinstance(bounds, dict), "Expected dictionary"
        assert 'practical' in bounds, "Missing 'practical' key"
        print("✓ Distance bound calculation works")
    except Exception as e:
        print(f"✗ Distance bound calculation failed: {e}")
        return False
    
    try:
        # Test graph generation
        graph = generate_ldpc_tanner_graph(12, 8, 3, 4)
        assert graph.number_of_nodes() == 20, f"Expected 20 nodes, got {graph.number_of_nodes()}"
        print("✓ LDPC graph generation works")
    except Exception as e:
        print(f"✗ LDPC graph generation failed: {e}")
        return False
    
    try:
        # Test graph embedding
        positions = embed_graph_on_surface(graph, 'torus', 1)
        assert len(positions) == graph.number_of_nodes(), "Position count mismatch"
        print("✓ Graph embedding works")
    except Exception as e:
        print(f"✗ Graph embedding failed: {e}")
        return False
    
    return True


def test_main_visualization():
    """Test main visualization components"""
    print("\nTesting main visualization...")
    
    try:
        # Import main classes (without running)
        import sys
        sys.path.append(os.path.dirname(__file__))
        
        # Test import without running GUI
        print("Testing TopologicalTannerGraph class...")
        from interactive_3d_tanner_graph import TopologicalTannerGraph
        
        # Create instance
        tanner = TopologicalTannerGraph(n_data=12, n_check=8, genus=1)
        print(f"✓ TopologicalTannerGraph created ({tanner.n_data} data, {tanner.n_check} check)")
        
        # Test basic operations
        tanner.inject_error('d0', 1)
        metrics = tanner.get_performance_metrics()
        assert isinstance(metrics, dict), "Expected metrics dictionary"
        print("✓ Error injection and metrics calculation work")
        
        # Test topology changes
        original_genus = tanner.genus
        tanner.set_genus(2)
        assert tanner.genus == 2, f"Genus change failed: {tanner.genus}"
        print("✓ Topology changes work")
        
        return True
        
    except ImportError as e:
        print(f"✗ Main visualization import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Main visualization test failed: {e}")
        return False


def test_examples():
    """Test example scripts"""
    print("\nTesting examples...")
    
    try:
        from tanner_graph_examples import (
            generate_genus_g_surface,
            calculate_distance_bound,
            generate_ldpc_tanner_graph
        )
        print("✓ Examples module imports work")
    except ImportError as e:
        print(f"✗ Examples import failed: {e}")
        return False
    
    try:
        # Test a simple example function
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for testing
        
        # This should work without showing plots
        graph = generate_ldpc_tanner_graph(12, 8, 3, 4)
        print("✓ Example graph generation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Examples test failed: {e}")
        return False


def test_color_palettes():
    """Test color palette setup"""
    print("\nTesting color palettes...")
    
    try:
        import seaborn as sns
        
        # Test the specific palettes used in the project
        seqCmap = sns.color_palette("mako", as_cmap=True)
        divCmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
        altCmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
        
        # Test that they can be called
        test_color = seqCmap(0.5)
        assert len(test_color) >= 3, "Color should have RGB components"
        
        print("✓ Color palettes configured correctly")
        return True
        
    except Exception as e:
        print(f"✗ Color palette test failed: {e}")
        return False


def create_test_plot():
    """Create a simple test plot to verify plotting works"""
    print("\nCreating test plot...")
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        
        # Create simple 3D plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Generate test data
        x = np.random.randn(20)
        y = np.random.randn(20)
        z = np.random.randn(20)
        
        ax.scatter(x, y, z, c='blue', s=50)
        ax.set_title('Trident Test Plot')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Save test plot
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/test_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ Test plot created successfully (saved as plots/test_plot.png)")
        return True
        
    except Exception as e:
        print(f"✗ Test plot creation failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("Trident 3D Tanner Graph Visualization - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Topological Utilities", test_topological_utilities),
        ("Main Visualization", test_main_visualization),
        ("Examples", test_examples),
        ("Color Palettes", test_color_palettes),
        ("Test Plot Creation", create_test_plot)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:<25} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! The Trident visualization should work correctly.")
        print("You can now run:")
        print("  - python interactive_3d_tanner_graph.py (main visualization)")
        print("  - python tanner_graph_examples.py (examples)")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Please check the error messages above.")
        print("You may need to install missing dependencies:")
        print("  pip install matplotlib numpy seaborn networkx scipy")
    
    return passed == len(results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)