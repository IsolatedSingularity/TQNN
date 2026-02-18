# Interactive TQNN (Topological Quantum Neural Network) GUI

This interactive visualization demonstrates key concepts from the TQNN research project, providing a hands-on exploration of topological quantum neural networks.

## Features

### Core Visualizations
- **Input Pattern + Noise**: Interactive pattern editor with noise injection
- **Anyonic Braiding**: Real-time visualization of topological braiding operations
- **Charge Flow Network**: Topological charge conservation in neural networks
- **Classification Confidence**: Real-time TQNN classification results
- **Spin Network Representation**: Hexagonal lattice spin network visualization
- **Topological Robustness**: Confidence tracking under noise

### Interactive Controls
- **Pattern Selection**: Choose from pre-defined geometric patterns (Vertical, Horizontal, Cross, Circle)
- **Noise Level Slider**: Adjust topological defect density (0.0 to 0.5)
- **Manual Classification**: Single-step classification with current pattern
- **Auto Mode**: Continuous classification with dynamic noise variation
- **Display Options**: Toggle individual visualization components
- **Pattern Editing**: Click pixels to manually edit patterns

## How to Run

1. **Prerequisites**: Make sure you have the required dependencies:
   ```bash
   pip install matplotlib numpy seaborn networkx
   ```

2. **Run the GUI**:
   ```bash
   cd Code+
   python interactive_tqnn_gui.py
   ```

3. **Test imports** (optional):
   ```bash
   python test_imports.py
   ```

## Usage Instructions

### Getting Started
1. **Select a Pattern**: Use the radio buttons to choose a training pattern
2. **Adjust Noise**: Use the slider to add topological defects (noise) to the pattern
3. **Classify**: Click "Classify" to see how the TQNN responds to the noisy pattern
4. **Observe**: Watch the confidence changes in real-time

### Interactive Features
- **Pattern Editing**: Click on pixels in the "Input Pattern + Noise" display to toggle them
- **Auto Mode**: Enable automatic classification with slowly varying noise levels
- **Display Options**: Use checkboxes to show/hide different visualization components
- **Real-time Updates**: All visualizations update dynamically as you interact

### Understanding the Visualizations

#### Anyonic Braiding
- Shows world-lines of anyonic particles
- Demonstrates topological braiding operations
- Red dashed lines indicate active braiding

#### Charge Flow Network
- Visualizes conservation of topological charge
- Nodes show charge values at different network layers
- Animated edges show information flow

#### Spin Network
- Hexagonal lattice representation of the input pattern
- Each hexagon represents a spin state
- ↑ symbols indicate spin-up states

#### Classification Confidence
- Bar chart showing log probabilities for each class
- Green bar indicates the correct class
- Predicted class is highlighted in darker color

#### Topological Robustness
- Time series of classification confidence
- Shows how robust the TQNN is to topological defects
- Demonstrates the key advantage of topological approaches

## Technical Details

### TQNN Implementation
- Based on the semi-classical limit theory from the research papers
- Uses prototype-based learning (no gradient descent required)
- Implements topological charge conservation
- Demonstrates robustness to local perturbations

### Visualization Architecture
- Real-time animation using matplotlib
- Modular design with separate drawing functions
- Interactive controls using matplotlib widgets
- Efficient update system for smooth performance

## Troubleshooting

### Import Errors
If you get import errors for `tqnn_helpers`, make sure:
1. You're running from the `Code+` directory
2. The `Code` directory exists and contains `tqnn_helpers.py`
3. The path resolution in the script is working correctly

### Performance Issues
- Close other applications if the GUI is slow
- Disable some visualizations using checkboxes if needed
- The animation runs at 150ms intervals by default

### Display Issues
- Make sure your screen resolution can accommodate the 18x12 inch figure
- Use the display option checkboxes to hide components if they overlap
- The layout is designed to avoid overlapping elements

## Educational Value

This GUI demonstrates several key concepts from the TQNN research:

1. **Topological Robustness**: How TQNNs maintain performance under noise
2. **Charge Conservation**: Fundamental principle of topological quantum systems
3. **Semi-classical Limit**: How classical neural networks emerge from quantum systems
4. **Anyonic Braiding**: Fundamental topological operations
5. **Spin Networks**: Quantum representation of classical data

## Based on Research

This visualization is based on the research papers:
- "Deep Neural Networks as the Semi-classical Limit of Topological Quantum Neural Networks"
- "Exact Evaluation of Hexagonal Spin-networks and Topological Quantum Neural Networks"
- "Sequential measurements, TQFTs, and TQNNs"

The implementation uses the TQNN sandbox code from the main project, providing an interactive way to explore the theoretical concepts.
