"""
Test script to verify TQNN imports work correctly
"""
import sys
import os

# Add the Code directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Code'))

try:
    from tqnn_helpers import TQNNPerceptron, add_topological_defect, create_spin_network_from_pattern
    print("✓ Successfully imported TQNN helpers")
    
    # Test basic functionality
    tqnn = TQNNPerceptron()
    print("✓ Created TQNNPerceptron instance")
    
    # Test pattern creation
    import numpy as np
    test_pattern = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    noisy_pattern = add_topological_defect(test_pattern, 0.1)
    print("✓ Created noisy pattern")
    
    spin_network = create_spin_network_from_pattern(test_pattern)
    print("✓ Created spin network")
    
    print("\nAll imports and basic functionality working correctly!")
    print("The interactive GUI should work properly.")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure you're running from the Code+ directory")
    
except Exception as e:
    print(f"✗ Error: {e}")
