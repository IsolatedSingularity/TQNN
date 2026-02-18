# Code Directory

This directory contains the Python source code for the TQNN project.

## TQNN Sandbox

The primary component is the TQNN sandbox, designed to provide a hands-on demonstration of TQNN principles, especially robustness to noise.

### Files
- `tqnn_sandbox.py`: The main executable script for the sandbox. It defines simple geometric patterns, trains a TQNN classifier, introduces noise, and generates visualizations of the results.
- `tqnn_helpers.py`: A module containing the core logic for the TQNN model. It includes the `TQNNPerceptron` class, which is based on the semi-classical model described in the project's reference papers.

### How to Run
To run the sandbox experiment, execute the main script from the project's root directory:
```bash
python Code/tqnn_sandbox.py
```

### Output
Running the script will produce two files in the `Plots/` directory:
1.  `tqnn_robustness_sandbox.png`: A static line plot showing the degradation of the TQNN's classification confidence as noise increases.
2.  `tqnn_robustness_animation.gif`: A dynamic animation visualizing the noise injection and the classifier's real-time confidence.

## Examples

The `Examples/` subdirectory contains standalone scripts that demonstrate various concepts and visualization techniques, adhering to the project's code quality standards.
- `exampleStatic.py`: Demonstrates the creation of static plots with `matplotlib`.
- `exampleAnimation.py`: Demonstrates the creation of GIF animations.
- `exampleREADME.md`: Provides a template and standard for high-quality documentation. 