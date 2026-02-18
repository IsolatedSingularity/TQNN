"""
Interactive TQNN Tensor Network Simulator

This application provides a real-time interactive visualization of Topological Quantum Neural
Networks based on the Marcianò-Zappalà framework. Users can:

- Draw quantum states on a canvas that are encoded as hexagonal spin-networks
- Watch real-time TQFT transition amplitude computation
- Observe the semi-classical limit where TQNNs recover classical perceptrons
- Visualize 6j-symbol recoupling theory at network vertices
- Compare input spin-networks against class prototypes via physical scalar products
- Track quantum dimensions Δ_j = 2j+1 across the network

This represents TQNN processing of visual input through the lens of topological quantum
field theory - computing transition amplitudes A = Σ Δ_j exp(-(j-j̄)²/2σ²) in real-time.

Theoretical Foundation:
- Spin-network encoding: j_i = N + floor(x_i) for input pixel x_i
- TQFT functor: Z(M): H_in → H_out via cobordism evolution
- Semi-classical limit: Classical DNN weights emerge as j_i/N when N→∞

GUI Framework: tkinter with custom dark theme
Architecture: Event-driven with TQFT computation backend
Author: TQNN Research Team
"""

import tkinter as tk
from tkinter import ttk, messagebox, Canvas
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from collections import deque
import time
from scipy.special import factorial

# Scientific computing
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
import seaborn as sns

# Set up color palettes for consistency with project standards
sns.set_style("darkgrid")
seqCmap = sns.color_palette("mako", as_cmap=True)
divCmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
lightCmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)


class SpinNetworkMode(Enum):
    """Types of spin-network encoding available"""
    HEXAGONAL = "Hexagonal Lattice (Lulli et al.)"
    STAR = "Star Graph (Perceptron)"
    GRID = "Desingularized Grid"


class TutorialWindow:
    """
    Interactive tutorial window for the TQNN Spin-Network Simulator.
    
    Provides a multi-page walkthrough of the simulator's features,
    theoretical background, and usage instructions with colored text.
    """
    
    # Tutorial content with colored text tags
    TUTORIAL_PAGES = [
        {
            "title": "Welcome to the TQNN Spin-Network Simulator",
            "content": [
                ("Welcome!", "header"),
                ("", "normal"),
                ("This simulator implements ", "normal"),
                ("Topological Quantum Neural Networks (TQNNs)", "highlight"),
                (" based on the ", "normal"),
                ("Marcianò-Zappalà framework", "accent"),
                (".", "normal"),
                ("", "normal"),
                ("", "normal"),
                ("TQNNs represent a fundamentally new approach to neural networks where:", "normal"),
                ("", "normal"),
                ("  • ", "normal"),
                ("Input data", "highlight"),
                (" is encoded as ", "normal"),
                ("spin-networks", "accent"),
                (" (quantum graphs)", "normal"),
                ("", "normal"),
                ("  • ", "normal"),
                ("Processing", "highlight"),
                (" occurs via ", "normal"),
                ("TQFT transition amplitudes", "accent"),
                ("", "normal"),
                ("  • ", "normal"),
                ("Classification", "highlight"),
                (" uses ", "normal"),
                ("physical scalar products", "accent"),
                (" between states", "normal"),
                ("", "normal"),
                ("", "normal"),
                ("The key insight: in the ", "normal"),
                ("semi-classical limit (N→∞)", "formula"),
                (", TQNNs", "normal"),
                ("", "normal"),
                ("reduce exactly to classical deep neural networks!", "highlight"),
            ]
        },
        {
            "title": "The Spin-Network Encoding",
            "content": [
                ("Spin-Network Encoding", "header"),
                ("", "normal"),
                ("When you draw on the canvas, your pattern is encoded as a ", "normal"),
                ("spin-network", "highlight"),
                (":", "normal"),
                ("", "normal"),
                ("", "normal"),
                ("The encoding formula:", "normal"),
                ("", "normal"),
                ("    j_i = N + ⌊x_i⌋", "formula"),
                ("", "normal"),
                ("", "normal"),
                ("Where:", "normal"),
                ("  • ", "normal"),
                ("j_i", "accent"),
                (" = spin label on edge i (SU(2) irrep)", "normal"),
                ("", "normal"),
                ("  • ", "normal"),
                ("N", "accent"),
                (" = large spin parameter (adjustable via slider)", "normal"),
                ("", "normal"),
                ("  • ", "normal"),
                ("x_i", "accent"),
                (" = input pixel value from your drawing", "normal"),
                ("", "normal"),
                ("", "normal"),
                ("Each edge also has a ", "normal"),
                ("quantum dimension", "highlight"),
                (":", "normal"),
                ("", "normal"),
                ("    Δ_j = 2j + 1", "formula"),
                ("", "normal"),
                ("", "normal"),
                ("The hexagonal lattice visualization shows your encoded spin-network", "normal"),
                ("with spin values displayed at each node.", "normal"),
            ]
        },
        {
            "title": "TQFT Transition Amplitudes",
            "content": [
                ("TQFT Transition Amplitudes", "header"),
                ("", "normal"),
                ("Classification in TQNNs uses ", "normal"),
                ("transition amplitudes", "highlight"),
                (" from", "normal"),
                ("", "normal"),
                ("Topological Quantum Field Theory:", "accent"),
                ("", "normal"),
                ("", "normal"),
                ("The amplitude formula:", "normal"),
                ("", "normal"),
                ("    A_c = ∏ Δ_j · exp(-(j - j̄)² / 2σ²)", "formula"),
                ("", "normal"),
                ("", "normal"),
                ("Where:", "normal"),
                ("  • ", "normal"),
                ("A_c", "accent"),
                (" = amplitude for class c", "normal"),
                ("", "normal"),
                ("  • ", "normal"),
                ("Δ_j = 2j+1", "accent"),
                (" = quantum dimension factor", "normal"),
                ("", "normal"),
                ("  • ", "normal"),
                ("j̄, σ", "accent"),
                (" = mean and std of class prototype spins", "normal"),
                ("", "normal"),
                ("", "normal"),
                ("The ", "normal"),
                ("bar chart", "highlight"),
                (" shows |A_c|² for each class - the highest", "normal"),
                ("", "normal"),
                ("amplitude wins! A ", "normal"),
                ("★ star", "accent"),
                (" marks the predicted class.", "normal"),
            ]
        },
        {
            "title": "6j-Symbols (Racah-Wigner Coefficients)",
            "content": [
                ("6j-Symbol Recoupling Matrix", "header"),
                ("", "normal"),
                ("At each vertex of a spin-network, angular momenta must ", "normal"),
                ("recouple", "highlight"),
                (".", "normal"),
                ("This is governed by ", "normal"),
                ("6j-symbols", "accent"),
                (":", "normal"),
                ("", "normal"),
                ("", "normal"),
                ("    {j₁  j₂  j₃}", "formula"),
                ("    {j₄  j₅  j₆}", "formula"),
                ("", "normal"),
                ("", "normal"),
                ("These ", "normal"),
                ("Racah-Wigner coefficients", "highlight"),
                (" appear in:", "normal"),
                ("", "normal"),
                ("  • Angular momentum coupling in quantum mechanics", "normal"),
                ("", "normal"),
                ("  • Evaluation of spin-foam amplitudes", "normal"),
                ("", "normal"),
                ("  • TQFT state-sum models (Ponzano-Regge, Turaev-Viro)", "normal"),
                ("", "normal"),
                ("", "normal"),
                ("The ", "normal"),
                ("heatmap", "highlight"),
                (" shows the 6j-symbol matrix computed at", "normal"),
                ("", "normal"),
                ("vertices of your spin-network. These encode ", "normal"),
                ("topological", "accent"),
                ("", "normal"),
                ("invariance", "accent"),
                (" of the TQNN.", "normal"),
            ]
        },
        {
            "title": "The Semi-Classical Limit",
            "content": [
                ("Semi-Classical Limit: N → ∞", "header"),
                ("", "normal"),
                ("The ", "normal"),
                ("key theorem", "highlight"),
                (" of Marcianò-Zappalà:", "normal"),
                ("", "normal"),
                ("", "normal"),
                ("As the large spin parameter N → ∞, the TQNN reduces", "accent"),
                ("", "normal"),
                ("to a classical deep neural network!", "accent"),
                ("", "normal"),
                ("", "normal"),
                ("The emergent DNN weights are:", "normal"),
                ("", "normal"),
                ("    w_i = j_i / N", "formula"),
                ("", "normal"),
                ("", "normal"),
                ("Use the ", "normal"),
                ("N_large slider", "highlight"),
                (" to explore this limit:", "normal"),
                ("", "normal"),
                ("  • ", "normal"),
                ("Small N", "accent"),
                (" (100-500): Strong quantum corrections O(1/√N)", "normal"),
                ("", "normal"),
                ("  • ", "normal"),
                ("Large N", "accent"),
                (" (2000+): Classical DNN behavior emerges", "normal"),
                ("", "normal"),
                ("", "normal"),
                ("The bottom-right plot shows the ", "normal"),
                ("classical weights", "highlight"),
                (" that", "normal"),
                ("", "normal"),
                ("emerge from your quantum spin-network encoding.", "normal"),
            ]
        },
        {
            "title": "How to Use This Simulator",
            "content": [
                ("Quick Start Guide", "header"),
                ("", "normal"),
                ("1. ", "highlight"),
                ("DRAW", "accent"),
                (" a pattern on the canvas (left side)", "normal"),
                ("", "normal"),
                ("   Your drawing is instantly encoded as a spin-network", "normal"),
                ("", "normal"),
                ("", "normal"),
                ("2. ", "highlight"),
                ("OBSERVE", "accent"),
                (" the four visualization panels:", "normal"),
                ("", "normal"),
                ("   • Top-left: Hexagonal spin-network with j labels", "normal"),
                ("", "normal"),
                ("   • Top-right: TQFT transition amplitudes per class", "normal"),
                ("", "normal"),
                ("   • Bottom-left: 6j-symbol recoupling matrix", "normal"),
                ("", "normal"),
                ("   • Bottom-right: Semi-classical weight emergence", "normal"),
                ("", "normal"),
                ("", "normal"),
                ("3. ", "highlight"),
                ("ADJUST", "accent"),
                (" the N_large slider to explore semi-classical limit", "normal"),
                ("", "normal"),
                ("", "normal"),
                ("4. ", "highlight"),
                ("EXPERIMENT", "accent"),
                (" with different patterns:", "normal"),
                ("", "normal"),
                ("   • ", "normal"),
                ("Clear", "highlight"),
                (" - reset the canvas", "normal"),
                ("", "normal"),
                ("   • ", "normal"),
                ("Random", "highlight"),
                (" - generate a random spin-network state", "normal"),
                ("", "normal"),
                ("   • ", "normal"),
                ("Compute TQFT", "highlight"),
                (" - manually trigger computation", "normal"),
            ]
        },
    ]
    
    def __init__(self, parent, show_on_startup_var: tk.BooleanVar):
        """
        Initialize the tutorial window.
        
        Args:
            parent: Parent tkinter window
            show_on_startup_var: BooleanVar controlling show-on-startup behavior
        """
        self.parent = parent
        self.show_on_startup_var = show_on_startup_var
        self.current_page = 0
        self.window = None
        self.text_widget = None
        
    def show(self):
        """Display the tutorial window."""
        if self.window is not None and self.window.winfo_exists():
            self.window.lift()
            return
            
        self.window = tk.Toplevel(self.parent)
        self.window.title("TQNN Simulator Tutorial")
        self.window.geometry("750x650")
        self.window.configure(bg='#1a1a1a')
        self.window.transient(self.parent)
        self.window.minsize(700, 600)
        
        # Make it modal-ish but not blocking
        self.window.grab_set()
        
        self._setup_tutorial_ui()
        self._display_page(0)
        
        # Center the window
        self.window.update_idletasks()
        x = self.parent.winfo_x() + (self.parent.winfo_width() - self.window.winfo_width()) // 2
        y = self.parent.winfo_y() + (self.parent.winfo_height() - self.window.winfo_height()) // 2
        self.window.geometry(f"+{x}+{y}")
        
    def _setup_tutorial_ui(self):
        """Set up the tutorial window UI components."""
        # Main container
        main_frame = tk.Frame(self.window, bg='#1a1a1a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Title bar with page indicator
        title_frame = tk.Frame(main_frame, bg='#1a1a1a')
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.title_label = tk.Label(title_frame, text="", 
                                   font=('Arial', 14, 'bold'),
                                   bg='#1a1a1a', fg='#00ff88')
        self.title_label.pack(side=tk.LEFT)
        
        self.page_label = tk.Label(title_frame, text="", 
                                  font=('Arial', 10),
                                  bg='#1a1a1a', fg='#888888')
        self.page_label.pack(side=tk.RIGHT)
        
        # Content area with scrollbar
        content_frame = tk.Frame(main_frame, bg='#0a0a0a', bd=2, relief='sunken')
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        scrollbar = tk.Scrollbar(content_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.text_widget = tk.Text(content_frame, 
                                  wrap=tk.WORD,
                                  font=('Consolas', 11),
                                  bg='#0a0a0a',
                                  fg='#ffffff',
                                  insertbackground='#00ff88',
                                  selectbackground='#2d2d2d',
                                  padx=15, pady=15,
                                  yscrollcommand=scrollbar.set,
                                  state=tk.DISABLED,
                                  cursor='arrow')
        self.text_widget.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.text_widget.yview)
        
        # Configure text tags for colored text
        self.text_widget.tag_configure('header', 
                                       font=('Arial', 16, 'bold'),
                                       foreground='#00ff88',
                                       spacing1=5, spacing3=10)
        self.text_widget.tag_configure('normal', 
                                       font=('Consolas', 11),
                                       foreground='#ffffff')
        self.text_widget.tag_configure('highlight', 
                                       font=('Consolas', 11, 'bold'),
                                       foreground='#ffff00')
        self.text_widget.tag_configure('accent', 
                                       font=('Consolas', 11, 'bold'),
                                       foreground='#00ffff')
        self.text_widget.tag_configure('formula', 
                                       font=('Courier New', 12, 'bold'),
                                       foreground='#ff88ff',
                                       spacing1=5, spacing3=5)
        
        # Page dots indicator - separate row at top of nav area
        dots_row = tk.Frame(main_frame, bg='#1a1a1a')
        dots_row.pack(fill=tk.X, pady=(0, 5))
        
        self.dots_frame = tk.Frame(dots_row, bg='#1a1a1a')
        self.dots_frame.pack()
        
        self.dot_labels = []
        for i in range(len(self.TUTORIAL_PAGES)):
            dot = tk.Label(self.dots_frame, text="●", 
                          font=('Arial', 14),
                          bg='#1a1a1a', fg='#444444',
                          cursor='hand2')
            dot.pack(side=tk.LEFT, padx=5)
            dot.bind('<Button-1>', lambda e, idx=i: self._display_page(idx))
            self.dot_labels.append(dot)
        
        # Navigation buttons - separate row below dots
        nav_frame = tk.Frame(main_frame, bg='#1a1a1a')
        nav_frame.pack(fill=tk.X, pady=(5, 10))
        
        self.btn_prev = tk.Button(nav_frame, text="  << PREVIOUS  ",
                                 font=('Arial', 11, 'bold'),
                                 bg='#2d2d2d', fg='#ffffff',
                                 activebackground='#3d3d3d',
                                 activeforeground='#00ff88',
                                 relief='raised', bd=2,
                                 command=self._prev_page)
        self.btn_prev.pack(side=tk.LEFT, padx=10)
        
        self.btn_next = tk.Button(nav_frame, text="    NEXT >>    ",
                                 font=('Arial', 11, 'bold'),
                                 bg='#00ff88', fg='#000000',
                                 activebackground='#00cc66',
                                 activeforeground='#000000',
                                 relief='raised', bd=2,
                                 command=self._next_page)
        self.btn_next.pack(side=tk.RIGHT, padx=10)
        
        # Bottom options frame
        options_frame = tk.Frame(main_frame, bg='#1a1a1a')
        options_frame.pack(fill=tk.X)
        
        # Show on startup checkbox
        self.startup_check = tk.Checkbutton(options_frame, 
                                           text="Show tutorial on startup",
                                           variable=self.show_on_startup_var,
                                           font=('Arial', 9),
                                           bg='#1a1a1a', fg='#888888',
                                           activebackground='#1a1a1a',
                                           activeforeground='#ffffff',
                                           selectcolor='#2d2d2d')
        self.startup_check.pack(side=tk.LEFT)
        
        # Close button
        self.btn_close = tk.Button(options_frame, text="Close Tutorial",
                                  font=('Arial', 10, 'bold'),
                                  bg='#00ff88', fg='#000000',
                                  activebackground='#00cc66',
                                  activeforeground='#000000',
                                  bd=0, padx=20, pady=8,
                                  command=self._close)
        self.btn_close.pack(side=tk.RIGHT)
        
    def _display_page(self, page_idx: int):
        """Display a specific tutorial page."""
        self.current_page = max(0, min(len(self.TUTORIAL_PAGES) - 1, page_idx))
        page = self.TUTORIAL_PAGES[self.current_page]
        
        # Update title
        self.title_label.config(text=page["title"])
        self.page_label.config(text=f"Page {self.current_page + 1} of {len(self.TUTORIAL_PAGES)}")
        
        # Update content
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.delete(1.0, tk.END)
        
        for text, tag in page["content"]:
            if text == "":
                self.text_widget.insert(tk.END, "\n")
            else:
                self.text_widget.insert(tk.END, text, tag)
        
        self.text_widget.config(state=tk.DISABLED)
        
        # Update navigation buttons
        self.btn_prev.config(state=tk.NORMAL if self.current_page > 0 else tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL if self.current_page < len(self.TUTORIAL_PAGES) - 1 else tk.DISABLED)
        
        # Update dots
        for i, dot in enumerate(self.dot_labels):
            dot.config(fg='#00ff88' if i == self.current_page else '#444444')
            
    def _next_page(self):
        """Go to next page."""
        self._display_page(self.current_page + 1)
        
    def _prev_page(self):
        """Go to previous page."""
        self._display_page(self.current_page - 1)
        
    def _close(self):
        """Close the tutorial window."""
        if self.window:
            self.window.grab_release()
            self.window.destroy()
            self.window = None


@dataclass
class SpinNetworkState:
    """
    Represents a spin-network state for TQNN computation
    
    Following Marcianò et al.: spin-networks are graphs Γ with edges labeled by
    SU(2) irreps j_e and vertices by intertwiners ι_v.
    
    Attributes:
        n_edges: Number of edges in the spin-network
        spin_labels: Array of spin values j_i on each edge (half-integers)
        quantum_dimensions: Δ_j = 2j + 1 for each edge
        intertwiner_labels: Labels at vertices (simplified)
        N_large: Large spin parameter for semi-classical limit
        transition_amplitude: Complex amplitude A from TQFT evaluation
        log_amplitude: log|A|² for numerical stability
    """
    n_edges: int
    spin_labels: np.ndarray
    quantum_dimensions: np.ndarray
    intertwiner_labels: np.ndarray = None
    N_large: int = 1000
    transition_amplitude: complex = 1.0
    log_amplitude: float = 0.0
    
    def __post_init__(self):
        if self.intertwiner_labels is None:
            self.intertwiner_labels = np.zeros(self.n_edges // 3 + 1)
        # Compute quantum dimensions Δ_j = 2j + 1
        self.quantum_dimensions = 2 * self.spin_labels + 1


@dataclass 
class ClassPrototype:
    """
    Prototype for a class in TQNN classification
    
    Following the "training-free" approach: prototypes are defined by
    mean and std of spin colors from training examples.
    """
    label: str
    mean_spins: np.ndarray
    std_spins: np.ndarray
    n_samples: int = 0


class TQNNProcessor:
    """
    Handles TQFT-based computation for Topological Quantum Neural Networks
    
    This class implements the Marcianò-Zappalà framework:
    - Spin-network encoding of input patterns
    - TQFT transition amplitude computation
    - Semi-classical limit extraction
    - 6j-symbol (Racah-Wigner) coefficient calculation
    - Physical scalar product evaluation
    """
    
    def __init__(self, grid_size: int = 16, N_large: int = 1000):
        """Initialize the TQNN processor"""
        self.grid_size = grid_size
        self.N_large = N_large  # Large spin for semi-classical limit
        self.current_state = None
        self.network_mode = SpinNetworkMode.HEXAGONAL
        
        # Spin-network components
        self.spin_labels = np.array([])
        self.quantum_dimensions = np.array([])
        self.vertex_positions = []
        self.edge_connections = []
        
        # Class prototypes for classification
        self.prototypes: Dict[str, ClassPrototype] = {}
        self._initialize_default_prototypes()
        
        # TQFT computation results
        self.transition_amplitudes = {}  # Per-class amplitudes
        self.log_probabilities = {}
        self.six_j_symbols = np.array([])  # Racah-Wigner coefficients
        
        # Semi-classical analysis
        self.semiclassical_weights = np.array([])
        self.classical_activation = 0.0
        
        # History tracking
        self.amplitude_history = deque(maxlen=100)
        self.weight_convergence_history = deque(maxlen=50)
        
        # Hexagonal lattice for visualization
        self.hex_positions = []
        self.hex_spins = []
        self._generate_hexagonal_lattice()
        
    def _initialize_default_prototypes(self):
        """Initialize default class prototypes for demonstration"""
        # Create simple prototype classes
        n_edges = (self.grid_size * self.grid_size) // 4  # Approximate
        
        # Prototype A: Concentrated pattern (e.g., vertical line)
        mean_a = np.ones(n_edges) * self.N_large
        mean_a[n_edges//3:2*n_edges//3] += 10  # Higher spins in middle
        std_a = np.ones(n_edges) * 2.0
        self.prototypes['Class A'] = ClassPrototype('Class A', mean_a, std_a, 10)
        
        # Prototype B: Spread pattern (e.g., horizontal line)  
        mean_b = np.ones(n_edges) * self.N_large
        mean_b[::3] += 10  # Higher spins distributed
        std_b = np.ones(n_edges) * 2.0
        self.prototypes['Class B'] = ClassPrototype('Class B', mean_b, std_b, 10)
        
        # Prototype C: Diagonal pattern
        mean_c = np.ones(n_edges) * self.N_large
        for i in range(n_edges):
            if i % 4 == 0:
                mean_c[i] += 10
        std_c = np.ones(n_edges) * 2.0
        self.prototypes['Class C'] = ClassPrototype('Class C', mean_c, std_c, 10)
    
    def _generate_hexagonal_lattice(self):
        """Generate hexagonal lattice positions for visualization"""
        self.hex_positions = []
        hex_size = 4  # Number of hexagons per side
        
        for q in range(-hex_size, hex_size + 1):
            for r in range(-hex_size, hex_size + 1):
                if abs(q + r) <= hex_size:
                    # Axial to pixel coordinates
                    x = 1.5 * q
                    y = np.sqrt(3) * (r + q/2)
                    self.hex_positions.append((x, y, q, r))
        
        self.hex_spins = np.zeros(len(self.hex_positions))
    
    def pattern_to_spin_network(self, pattern: np.ndarray) -> SpinNetworkState:
        """
        Convert a drawn pattern into a spin-network state
        
        Following Marcianò et al.: j_i = N + floor(x_i) where x_i is pixel intensity
        This encoding places the system in the semi-classical regime.
        
        Args:
            pattern: 2D array representing the drawn pattern (values 0-1)
            
        Returns:
            SpinNetworkState object with spin labels and quantum dimensions
        """
        # Flatten and subsample pattern to get edge values
        flat_pattern = pattern.flatten()
        
        # Map to spin values: j_i = N_large + floor(x_i * scale)
        # Scale factor determines the "resolution" of spin encoding
        scale = 10.0
        spin_labels = self.N_large + np.floor(flat_pattern * scale)
        
        # Ensure half-integer spins (multiply by 0.5 for true SU(2))
        # For computational simplicity, we use integer approximation
        spin_labels = spin_labels.astype(float)
        
        # Compute quantum dimensions Δ_j = 2j + 1
        quantum_dims = 2 * spin_labels + 1
        
        # Update hexagonal lattice spins for visualization
        n_hex = len(self.hex_positions)
        if len(spin_labels) >= n_hex:
            self.hex_spins = spin_labels[:n_hex] - self.N_large  # Relative spins
        else:
            self.hex_spins = np.pad(spin_labels, (0, n_hex - len(spin_labels))) - self.N_large
        
        # Create state object
        state = SpinNetworkState(
            n_edges=len(spin_labels),
            spin_labels=spin_labels,
            quantum_dimensions=quantum_dims,
            N_large=self.N_large
        )
        
        self.current_state = state
        self.spin_labels = spin_labels
        self.quantum_dimensions = quantum_dims
        
        return state
    
    def compute_transition_amplitude(self, input_spins: np.ndarray, 
                                    proto_mean: np.ndarray, 
                                    proto_std: np.ndarray) -> Tuple[complex, float]:
        """
        Compute TQFT transition amplitude using the semi-classical formula
        
        From Marcianò et al., the amplitude in the large-j limit is:
        A = ∏_i Δ_{j_i} * exp(-(j_i - j̄_i)²/(2σ_i²)) * exp(-iξ_i j_i)
        
        Args:
            input_spins: Spin labels of input pattern
            proto_mean: Mean spin values for prototype
            proto_std: Std dev of spin values for prototype
            
        Returns:
            Tuple of (complex amplitude, log probability)
        """
        # Ensure arrays are same length
        min_len = min(len(input_spins), len(proto_mean), len(proto_std))
        j = input_spins[:min_len]
        j_bar = proto_mean[:min_len]
        sigma = proto_std[:min_len]
        
        # Avoid division by zero
        sigma = np.maximum(sigma, 1e-6)
        
        # Quantum dimension contribution: log(Δ_j) = log(2j + 1)
        log_quantum_dim = np.sum(np.log(2 * j + 1))
        
        # Gaussian suppression term: -(j - j̄)²/(2σ²)
        gaussian_term = -np.sum((j - j_bar)**2 / (2 * sigma**2))
        
        # Phase term (simplified): ξ_i ~ position-dependent phase
        xi = np.linspace(0, 0.1, min_len)  # Small phase variation
        phase_term = -np.sum(xi * j)
        
        # Log amplitude (for numerical stability)
        log_amplitude = log_quantum_dim + gaussian_term
        
        # Complex amplitude with phase
        amplitude = np.exp(log_amplitude / min_len) * np.exp(1j * phase_term / min_len)
        
        return amplitude, log_amplitude
    
    def compute_all_class_amplitudes(self) -> Dict[str, float]:
        """
        Compute transition amplitudes for all class prototypes
        
        Returns:
            Dictionary mapping class labels to log probabilities
        """
        if self.current_state is None:
            return {}
        
        self.transition_amplitudes = {}
        self.log_probabilities = {}
        
        for label, proto in self.prototypes.items():
            amplitude, log_prob = self.compute_transition_amplitude(
                self.current_state.spin_labels,
                proto.mean_spins,
                proto.std_spins
            )
            self.transition_amplitudes[label] = amplitude
            self.log_probabilities[label] = log_prob
        
        # Track amplitude history
        if self.log_probabilities:
            max_prob = max(self.log_probabilities.values())
            self.amplitude_history.append(max_prob)
        
        return self.log_probabilities
    
    def compute_six_j_symbol(self, j1: float, j2: float, j3: float,
                            j4: float, j5: float, j6: float) -> float:
        """
        Compute 6j-symbol (Racah-Wigner coefficient)
        
        The 6j-symbol {j1 j2 j3; j4 j5 j6} represents recoupling of three
        angular momenta and is fundamental to spin-network evaluation.
        
        Uses the Racah formula (simplified for visualization).
        """
        # Check triangle inequalities (admissibility)
        def triangle_ok(a, b, c):
            return abs(a - b) <= c <= a + b
        
        if not (triangle_ok(j1, j2, j3) and triangle_ok(j1, j5, j6) and
                triangle_ok(j4, j2, j6) and triangle_ok(j4, j5, j3)):
            return 0.0
        
        # Simplified computation using asymptotic formula for large j
        # Full Racah formula involves factorials and is expensive
        try:
            # Ponzano-Regge asymptotic: 6j ~ cos(S)/sqrt(12πV) for large j
            # where S is Regge action and V is tetrahedron volume
            avg_j = (j1 + j2 + j3 + j4 + j5 + j6) / 6
            if avg_j < 0.5:
                return 1.0
            
            # Simplified asymptotic
            phase = np.pi * (j1 + j2 + j3 + j4 + j5 + j6) / 4
            amplitude = 1.0 / np.sqrt(12 * np.pi * avg_j**3 + 1)
            
            return amplitude * np.cos(phase)
        except:
            return 0.0
    
    def compute_recoupling_matrix(self) -> np.ndarray:
        """
        Compute matrix of 6j-symbols for current spin configuration
        
        Returns:
            Matrix where entry (i,j) is the 6j-symbol connecting sites i and j
        """
        if self.current_state is None:
            return np.array([[1.0]])
        
        # Sample spins for 6j computation
        n_sample = min(8, len(self.spin_labels))
        sampled_spins = self.spin_labels[:n_sample] - self.N_large + 1  # Relative to N_large
        sampled_spins = np.maximum(sampled_spins, 0.5)  # Ensure positive
        
        # Build recoupling matrix
        matrix = np.zeros((n_sample, n_sample))
        
        for i in range(n_sample):
            for k in range(n_sample):
                if i != k:
                    # Use neighboring spins for the 6j computation
                    j1 = sampled_spins[i]
                    j2 = sampled_spins[k]
                    j3 = (sampled_spins[i] + sampled_spins[k]) / 2  # Coupled spin
                    j4 = sampled_spins[(i + 1) % n_sample]
                    j5 = sampled_spins[(k + 1) % n_sample]
                    j6 = (j4 + j5) / 2
                    
                    matrix[i, k] = self.compute_six_j_symbol(j1, j2, j3, j4, j5, j6)
                else:
                    matrix[i, k] = 1.0
        
        self.six_j_symbols = matrix
        return matrix
    
    def compute_semiclassical_weights(self) -> np.ndarray:
        """
        Extract semi-classical DNN weights from spin-network
        
        In the limit N_large → ∞, the TQNN reduces to a classical perceptron
        with weights w_i = j_i / N_large
        
        Returns:
            Array of classical weights
        """
        if self.current_state is None:
            return np.array([])
        
        # w_i = j_i / N_large (normalized to [0, 1] range)
        weights = (self.spin_labels - self.N_large) / 10.0  # Scale back
        weights = np.clip(weights, 0, 1)
        
        self.semiclassical_weights = weights
        
        # Track convergence to classical limit
        # As N_large increases, quantum corrections vanish
        quantum_correction = 1.0 / np.sqrt(self.N_large)
        self.weight_convergence_history.append(quantum_correction)
        
        return weights
    
    def compute_classical_activation(self) -> float:
        """
        Compute classical perceptron activation in semi-classical limit
        
        Returns:
            Activation value σ(w·x + b)
        """
        if len(self.semiclassical_weights) == 0:
            return 0.5
        
        # Simple dot product with uniform "input" (the pattern itself acts as both)
        z = np.mean(self.semiclassical_weights)
        
        # Sigmoid activation
        self.classical_activation = 1.0 / (1.0 + np.exp(-10 * (z - 0.5)))
        
        return self.classical_activation
    
    def get_predicted_class(self) -> Tuple[str, float]:
        """
        Get predicted class based on maximum transition amplitude
        
        Returns:
            Tuple of (predicted class label, confidence)
        """
        if not self.log_probabilities:
            return "Unknown", 0.0
        
        # Softmax over log probabilities
        log_probs = np.array(list(self.log_probabilities.values()))
        labels = list(self.log_probabilities.keys())
        
        # Numerical stability
        log_probs = log_probs - np.max(log_probs)
        probs = np.exp(log_probs / 100)  # Temperature scaling
        probs = probs / np.sum(probs)
        
        best_idx = np.argmax(probs)
        return labels[best_idx], probs[best_idx]


class TQNNVisualizerGUI:
    """
    Main GUI application for TQNN Spin-Network Visualization
    
    Provides real-time interactive visualization of TQFT computations
    including spin-network encoding, transition amplitudes, 6j-symbols,
    and semi-classical limit analysis.
    """
    
    def __init__(self):
        """Initialize the GUI application"""
        self.root = self._setup_gui()
        self.processor = TQNNProcessor(grid_size=16, N_large=1000)
        
        # Drawing canvas state
        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.pattern_array = np.zeros((16, 16))
        
        # Animation state
        self.animation_running = False
        self.animation_frame = 0
        self.update_interval = 100  # ms
        
        # Display options
        self.show_spin_network = True
        self.show_amplitudes = True
        self.show_six_j = True
        self.show_semiclassical = True
        
        # Tutorial settings
        self.show_tutorial_on_startup = tk.BooleanVar(value=True)
        self.tutorial = TutorialWindow(self.root, self.show_tutorial_on_startup)
        
        self._setup_ui()
        self._bind_events()
        self._start_animation()
        
        # Show tutorial on startup if enabled
        if self.show_tutorial_on_startup.get():
            self.root.after(500, self.tutorial.show)  # Slight delay for window to fully load
    
    def _setup_gui(self) -> tk.Tk:
        """Set up the main GUI window with dark theme"""
        root = tk.Tk()
        root.title("TQNN Spin-Network Simulator - Marcianò-Zappalà Framework")
        root.geometry("1700x1000")
        root.configure(bg='#1a1a1a')
        
        # Configure dark theme style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Dark theme colors
        style.configure('Dark.TFrame', background='#1a1a1a')
        style.configure('Dark.TLabel', background='#1a1a1a', foreground='#ffffff')
        style.configure('Dark.TLabelframe', background='#1a1a1a', foreground='#ffffff', 
                       borderwidth=2, relief='solid')
        style.configure('Dark.TLabelframe.Label', background='#1a1a1a', foreground='#00ff88',
                       font=('TkDefaultFont', 10, 'bold'))
        style.configure('Dark.TButton', background='#2d2d2d', foreground='#ffffff',
                       borderwidth=1, relief='raised')
        style.map('Dark.TButton',
                 background=[('active', '#3d3d3d')],
                 foreground=[('active', '#00ff88')])
        style.configure('Dark.TCheckbutton', background='#1a1a1a', foreground='#ffffff')
        style.configure('Dark.TScale', background='#1a1a1a', foreground='#ffffff')
        
        return root
    
    def _setup_ui(self):
        """Set up the user interface components"""
        # Main container
        main_container = ttk.Frame(self.root, style='Dark.TFrame')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Drawing canvas and controls
        left_panel = ttk.Frame(main_container, style='Dark.TFrame')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        
        # Right panel - Visualizations
        right_panel = ttk.Frame(main_container, style='Dark.TFrame')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self._setup_drawing_area(left_panel)
        self._setup_control_panel(left_panel)
        self._setup_visualization_panel(right_panel)
        self._setup_status_panel(left_panel)
    
    def _setup_drawing_area(self, parent):
        """Setup the drawing canvas area"""
        drawing_frame = ttk.LabelFrame(parent, text="Spin-Network Input (Draw Pattern)", 
                                      style='Dark.TLabelframe', padding=10)
        drawing_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 10))
        
        # Canvas for drawing
        self.drawing_canvas = Canvas(drawing_frame, width=400, height=400, 
                                    bg='#0a0a0a', highlightthickness=2,
                                    highlightbackground='#00ff88')
        self.drawing_canvas.pack()
        
        # Draw grid
        self._draw_grid()
        
        # Instructions
        instr_label = ttk.Label(drawing_frame, 
                              text="Draw patterns → encoded as spin-network j_i = N + ⌊x_i⌋\nReal-time TQFT amplitude computation",
                              style='Dark.TLabel', font=('TkDefaultFont', 9, 'italic'))
        instr_label.pack(pady=(5, 0))
    
    def _draw_grid(self):
        """Draw grid on canvas"""
        cell_size = 400 // 16
        for i in range(17):
            x = i * cell_size
            self.drawing_canvas.create_line(x, 0, x, 400, fill='#2a2a2a', width=1)
            self.drawing_canvas.create_line(0, x, 400, x, fill='#2a2a2a', width=1)
    
    def _setup_control_panel(self, parent):
        """Setup control buttons and options"""
        control_frame = ttk.LabelFrame(parent, text="TQNN Parameters", 
                                      style='Dark.TLabelframe', padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Action buttons
        btn_frame = ttk.Frame(control_frame, style='Dark.TFrame')
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.btn_clear = ttk.Button(btn_frame, text="Clear",
                                   command=self.clear_canvas, style='Dark.TButton')
        self.btn_clear.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        self.btn_compute = ttk.Button(btn_frame, text="Compute TQFT",
                                     command=self.manual_compute, style='Dark.TButton')
        self.btn_compute.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        self.btn_random = ttk.Button(btn_frame, text="Random",
                                    command=self.generate_random_state, style='Dark.TButton')
        self.btn_random.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        # N_large slider for semi-classical limit
        n_frame = ttk.Frame(control_frame, style='Dark.TFrame')
        n_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(n_frame, text="N_large (semi-classical):", 
                 style='Dark.TLabel').pack(anchor=tk.W)
        
        self.n_large_var = tk.IntVar(value=1000)
        self.n_large_scale = tk.Scale(n_frame, from_=100, to=5000, 
                                     orient=tk.HORIZONTAL, variable=self.n_large_var,
                                     bg='#2d2d2d', fg='#00ff88', 
                                     highlightthickness=0, troughcolor='#1a1a1a',
                                     command=self._on_n_large_change)
        self.n_large_scale.pack(fill=tk.X)
        
        # Display options
        option_frame = ttk.Frame(control_frame, style='Dark.TFrame')
        option_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.var_auto = tk.BooleanVar(value=True)
        self.check_auto = ttk.Checkbutton(option_frame, text="Auto Compute",
                                         variable=self.var_auto, style='Dark.TCheckbutton')
        self.check_auto.pack(anchor=tk.W)
        
        # Tutorial button and checkbox
        tutorial_frame = ttk.Frame(control_frame, style='Dark.TFrame')
        tutorial_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.btn_tutorial = tk.Button(tutorial_frame, text="📖 Show Tutorial",
                                     font=('Arial', 9, 'bold'),
                                     bg='#00ff88', fg='#000000',
                                     activebackground='#00cc66',
                                     activeforeground='#000000',
                                     bd=0, padx=10, pady=5,
                                     command=self.tutorial.show)
        self.btn_tutorial.pack(fill=tk.X, pady=(0, 5))
        
        self.check_tutorial_startup = ttk.Checkbutton(tutorial_frame, 
                                                     text="Show tutorial on startup",
                                                     variable=self.show_tutorial_on_startup, 
                                                     style='Dark.TCheckbutton')
        self.check_tutorial_startup.pack(anchor=tk.W)
    
    def _on_n_large_change(self, value):
        """Handle N_large slider change"""
        self.processor.N_large = int(value)
        if np.sum(self.pattern_array) > 0:
            self.process_pattern()
    
    def _setup_status_panel(self, parent):
        """Setup status display panel"""
        status_frame = ttk.LabelFrame(parent, text="TQFT Computation Results", 
                                     style='Dark.TLabelframe', padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status text
        self.status_text = tk.Text(status_frame, height=12, bg='#0a0a0a', fg='#00ff88',
                                  insertbackground='#00ff88', selectbackground='#2d2d2d',
                                  font=('Courier', 9), wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(status_frame, command=self.status_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.config(yscrollcommand=scrollbar.set)
        
        self._update_status("TQNN Spin-Network Simulator initialized.\nDraw patterns to encode as spin-networks.")
    
    def _setup_visualization_panel(self, parent):
        """Setup matplotlib visualization panels with TQNN-specific plots"""
        # Create figure with subplots for TQNN visualizations
        self.fig = Figure(figsize=(12, 10), facecolor='#1a1a1a')
        
        # Create 2x2 grid of TQNN-specific visualizations
        gs = self.fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3, 
                                  top=0.95, bottom=0.08, left=0.08, right=0.95)
        
        # 1. Hexagonal Spin-Network Visualization
        self.ax_spinnet = self.fig.add_subplot(gs[0, 0])
        self.ax_spinnet.set_title("Hexagonal Spin-Network\n$j_i = N + \\lfloor x_i \\rfloor$", 
                                 color='white', fontsize=11, fontweight='bold')
        self.ax_spinnet.set_facecolor('#0a0a0a')
        
        # 2. Transition Amplitude / Class Probabilities
        self.ax_amplitude = self.fig.add_subplot(gs[0, 1])
        self.ax_amplitude.set_title("TQFT Transition Amplitudes\n$A_c = \\prod \\Delta_j e^{-(j-\\bar{j})^2/2\\sigma^2}$", 
                                   color='white', fontsize=11, fontweight='bold')
        self.ax_amplitude.set_facecolor('#0a0a0a')
        
        # 3. 6j-Symbol Recoupling Matrix
        self.ax_sixj = self.fig.add_subplot(gs[1, 0])
        self.ax_sixj.set_title("6j-Symbol Recoupling Matrix\n(Racah-Wigner Coefficients)", 
                              color='white', fontsize=11, fontweight='bold')
        self.ax_sixj.set_facecolor('#0a0a0a')
        
        # 4. Semi-Classical Limit Analysis
        self.ax_semiclass = self.fig.add_subplot(gs[1, 1])
        self.ax_semiclass.set_title("Semi-Classical Limit: $w_i = j_i/N$\n(DNN Weight Emergence)", 
                                   color='white', fontsize=11, fontweight='bold')
        self.ax_semiclass.set_facecolor('#0a0a0a')
        
        # Style all axes
        for ax in [self.ax_spinnet, self.ax_amplitude, self.ax_sixj, self.ax_semiclass]:
            ax.tick_params(colors='white', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#444444')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _bind_events(self):
        """Bind mouse and keyboard events"""
        self.drawing_canvas.bind("<Button-1>", self.start_drawing)
        self.drawing_canvas.bind("<B1-Motion>", self.draw)
        self.drawing_canvas.bind("<ButtonRelease-1>", self.stop_drawing)
        
        self.root.bind("<Escape>", lambda e: self.clear_canvas())
        self.root.bind("<space>", lambda e: self.manual_compute())
    
    def start_drawing(self, event):
        """Start drawing on canvas"""
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
        self.draw(event)
    
    def draw(self, event):
        """Draw on canvas and update pattern array"""
        if not self.drawing:
            return
        
        x, y = event.x, event.y
        
        # Draw on canvas
        if self.last_x and self.last_y:
            self.drawing_canvas.create_line(self.last_x, self.last_y, x, y,
                                          fill='#00ff88', width=3, capstyle=tk.ROUND)
        
        # Update pattern array
        cell_size = 400 // 16
        grid_x = min(15, max(0, x // cell_size))
        grid_y = min(15, max(0, y // cell_size))
        self.pattern_array[grid_y, grid_x] = 1.0
        
        # Also mark nearby cells for smoother patterns
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= nx < 16 and 0 <= ny < 16:
                    self.pattern_array[ny, nx] = max(self.pattern_array[ny, nx], 0.5)
        
        self.last_x = x
        self.last_y = y
    
    def stop_drawing(self, event):
        """Stop drawing and process the pattern"""
        self.drawing = False
        self.last_x = None
        self.last_y = None
        
        if self.var_auto.get():
            self.process_pattern()
    
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.drawing_canvas.delete("all")
        self._draw_grid()
        self.pattern_array = np.zeros((16, 16))
        self._update_status("Canvas cleared.")
    
    def manual_compute(self):
        """Manually trigger TQFT computation"""
        self.process_pattern()
    
    def generate_random_state(self):
        """Generate a random spin-network pattern"""
        # Create interesting random pattern
        self.pattern_array = np.random.random((16, 16))
        self.pattern_array = (self.pattern_array > 0.7).astype(float)
        
        # Clear canvas first
        self.drawing_canvas.delete("all")
        self._draw_grid()
        
        # Draw on canvas
        cell_size = 400 // 16
        for i in range(16):
            for j in range(16):
                if self.pattern_array[i, j] > 0:
                    x1, y1 = j * cell_size, i * cell_size
                    x2, y2 = x1 + cell_size, y1 + cell_size
                    self.drawing_canvas.create_rectangle(x1, y1, x2, y2,
                                                        fill='#00ff88', outline='')
        
        # Process the pattern
        self.process_pattern()
    
    def process_pattern(self):
        """Process the drawn pattern through TQNN computation pipeline"""
        if np.sum(self.pattern_array) == 0:
            return
        
        # 1. Encode pattern as spin-network
        state = self.processor.pattern_to_spin_network(self.pattern_array)
        
        # 2. Compute transition amplitudes for all classes
        log_probs = self.processor.compute_all_class_amplitudes()
        
        # 3. Compute 6j-symbol recoupling matrix
        self.processor.compute_recoupling_matrix()
        
        # 4. Extract semi-classical weights
        weights = self.processor.compute_semiclassical_weights()
        activation = self.processor.compute_classical_activation()
        
        # 5. Get prediction
        pred_class, confidence = self.processor.get_predicted_class()
        
        # Update status with detailed TQNN metrics
        status_msg = f"""
══════════════════════════════════════════════════
         TQFT COMPUTATION RESULTS
══════════════════════════════════════════════════

SPIN-NETWORK ENCODING:
  Edges (n):        {state.n_edges}
  N_large:          {self.processor.N_large}
  Spin range:       j ∈ [{state.spin_labels.min():.0f}, {state.spin_labels.max():.0f}]
  Mean spin ⟨j⟩:    {np.mean(state.spin_labels):.2f}
  Quantum dim Δ_j:  [{state.quantum_dimensions.min():.0f}, {state.quantum_dimensions.max():.0f}]

TRANSITION AMPLITUDES A_c = ∏ Δ_j exp(-(j-j̄)²/2σ²):
"""
        for label, log_p in log_probs.items():
            amp = self.processor.transition_amplitudes[label]
            status_msg += f"  {label}: log|A|² = {log_p:.2f}, |A| = {np.abs(amp):.4f}\n"
        
        status_msg += f"""
CLASSIFICATION (argmax |A_c|²):
  Predicted:        {pred_class}
  Confidence:       {confidence:.2%}

SEMI-CLASSICAL LIMIT (N→∞):
  w_i = j_i/N:      [{weights.min():.3f}, {weights.max():.3f}]
  Classical σ(w·x): {activation:.4f}
  Quantum corr.:    O(1/√N) = {1/np.sqrt(self.processor.N_large):.4f}

6j-SYMBOLS: {self.processor.six_j_symbols.shape[0]}×{self.processor.six_j_symbols.shape[1]} recoupling matrix
══════════════════════════════════════════════════
"""
        self._update_status(status_msg)
        
        # Update visualizations
        self._update_all_plots()
    
    def _update_status(self, message: str):
        """Update status display"""
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, message)
        self.status_text.see(tk.END)
    
    def _update_all_plots(self):
        """Update all visualization plots"""
        self._plot_spin_network()
        self._plot_transition_amplitudes()
        self._plot_six_j_matrix()
        self._plot_semiclassical_limit()
        self.canvas.draw()
    
    def _plot_spin_network(self):
        """Plot hexagonal spin-network with spin labels"""
        self.ax_spinnet.clear()
        
        if self.processor.current_state is None:
            self.ax_spinnet.text(0.5, 0.5, "Draw pattern to\nencode spin-network", 
                               ha='center', va='center', color='gray',
                               transform=self.ax_spinnet.transAxes, fontsize=12)
            self.ax_spinnet.set_title("Hexagonal Spin-Network\n$j_i = N + \\lfloor x_i \\rfloor$", 
                                     color='white', fontsize=11, fontweight='bold')
            return
        
        # Draw hexagonal lattice with spin labels
        hex_positions = self.processor.hex_positions
        hex_spins = self.processor.hex_spins
        
        # Normalize spins for coloring
        if len(hex_spins) > 0 and np.max(np.abs(hex_spins)) > 0:
            norm_spins = (hex_spins - hex_spins.min()) / (hex_spins.max() - hex_spins.min() + 1e-6)
        else:
            norm_spins = np.zeros(len(hex_positions))
        
        patches = []
        colors = []
        
        for i, (x, y, q, r) in enumerate(hex_positions):
            if i < len(norm_spins):
                hex_patch = RegularPolygon((x, y), numVertices=6, radius=0.9,
                                          orientation=np.pi/6, 
                                          edgecolor='#444444', linewidth=1)
                patches.append(hex_patch)
                colors.append(seqCmap(0.2 + 0.7 * norm_spins[i]))
                
                # Add spin label
                j_val = hex_spins[i]
                if abs(j_val) > 0.1:
                    self.ax_spinnet.text(x, y, f'{j_val:.0f}', ha='center', va='center',
                                        fontsize=7, color='white', fontweight='bold')
        
        # Add patches to axes
        for patch, color in zip(patches, colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            self.ax_spinnet.add_patch(patch)
        
        # Draw edges between hexagons (representing spin-network edges)
        for i, (x1, y1, q1, r1) in enumerate(hex_positions):
            for j, (x2, y2, q2, r2) in enumerate(hex_positions):
                if i < j:
                    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if dist < 2.5:  # Adjacent hexagons
                        self.ax_spinnet.plot([x1, x2], [y1, y2], 
                                            color='#00ff88', alpha=0.3, linewidth=0.5)
        
        self.ax_spinnet.set_xlim(-8, 8)
        self.ax_spinnet.set_ylim(-8, 8)
        self.ax_spinnet.set_aspect('equal')
        self.ax_spinnet.axis('off')
        self.ax_spinnet.set_title("Hexagonal Spin-Network\n$j_i = N + \\lfloor x_i \\rfloor$", 
                                 color='white', fontsize=11, fontweight='bold')
        
        # Add colorbar info
        self.ax_spinnet.text(0.02, 0.98, f'N = {self.processor.N_large}', 
                           transform=self.ax_spinnet.transAxes,
                           color='cyan', fontsize=9, va='top',
                           bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.8))
    
    def _plot_transition_amplitudes(self):
        """Plot TQFT transition amplitudes for each class"""
        self.ax_amplitude.clear()
        
        if not self.processor.log_probabilities:
            self.ax_amplitude.text(0.5, 0.5, "Transition amplitudes\nwill appear here", 
                                  ha='center', va='center', color='gray',
                                  transform=self.ax_amplitude.transAxes, fontsize=12)
            self.ax_amplitude.set_title("TQFT Transition Amplitudes\n$A_c = \\prod \\Delta_j e^{-(j-\\bar{j})^2/2\\sigma^2}$", 
                                       color='white', fontsize=11, fontweight='bold')
            return
        
        labels = list(self.processor.log_probabilities.keys())
        log_probs = list(self.processor.log_probabilities.values())
        amplitudes = [np.abs(self.processor.transition_amplitudes[l]) for l in labels]
        
        # Normalize for probability interpretation
        log_probs_arr = np.array(log_probs)
        log_probs_arr = log_probs_arr - np.max(log_probs_arr)  # Numerical stability
        probs = np.exp(log_probs_arr / 100)
        probs = probs / np.sum(probs)
        
        # Create bar plot
        x_pos = np.arange(len(labels))
        colors = [seqCmap(0.3 + 0.5 * p) for p in probs]
        
        bars = self.ax_amplitude.bar(x_pos, probs, color=colors, 
                                    edgecolor='none', linewidth=0, alpha=0.9)
        
        # Add value labels
        for i, (bar, prob, amp) in enumerate(zip(bars, probs, amplitudes)):
            height = bar.get_height()
            self.ax_amplitude.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                                  f'P={prob:.2f}\n|A|={amp:.3f}',
                                  ha='center', va='bottom', color='white', fontsize=8)
        
        self.ax_amplitude.set_xticks(x_pos)
        self.ax_amplitude.set_xticklabels(labels, color='white', fontsize=9)
        self.ax_amplitude.set_ylabel('P(class) = |A|²/Σ|A|²', color='white', fontsize=9)
        self.ax_amplitude.set_ylim(0, max(probs) * 1.4)
        # Hide x-axis tick marks (the thin vertical lines) but keep labels
        self.ax_amplitude.tick_params(axis='x', colors='white', labelsize=8, length=0)
        self.ax_amplitude.tick_params(axis='y', colors='white', labelsize=8)
        self.ax_amplitude.grid(True, alpha=0.2, color='gray', axis='y')
        # Hide bottom spine which can look like a line through bars
        self.ax_amplitude.spines['bottom'].set_visible(False)
        
        # Highlight predicted class with marker
        pred_class, _ = self.processor.get_predicted_class()
        if pred_class in labels:
            idx = labels.index(pred_class)
            # Add star marker above predicted class bar
            self.ax_amplitude.scatter([idx], [probs[idx] + 0.12], marker='*', 
                                     s=200, color='#00ff88', zorder=5)
        
        self.ax_amplitude.set_title("TQFT Transition Amplitudes\n$A_c = \\prod \\Delta_j e^{-(j-\\bar{j})^2/2\\sigma^2}$", 
                                   color='white', fontsize=11, fontweight='bold')
    
    def _plot_six_j_matrix(self):
        """Plot 6j-symbol recoupling matrix"""
        # Remove existing colorbar if present
        if hasattr(self, '_sixj_colorbar') and self._sixj_colorbar is not None:
            try:
                self._sixj_colorbar.remove()
            except:
                pass
            self._sixj_colorbar = None
        
        self.ax_sixj.clear()
        
        matrix = self.processor.six_j_symbols
        
        if matrix.size == 0 or matrix.shape[0] < 2:
            self.ax_sixj.text(0.5, 0.5, "6j-symbol matrix\nwill appear here", 
                            ha='center', va='center', color='gray',
                            transform=self.ax_sixj.transAxes, fontsize=12)
            self.ax_sixj.set_title("6j-Symbol Recoupling Matrix\n(Racah-Wigner Coefficients)", 
                                  color='white', fontsize=11, fontweight='bold')
            return
        
        # Plot heatmap of 6j-symbols
        im = self.ax_sixj.imshow(matrix, cmap=divCmap, aspect='auto',
                                interpolation='nearest', vmin=-0.5, vmax=0.5)
        
        # Add colorbar and store reference
        self._sixj_colorbar = self.fig.colorbar(im, ax=self.ax_sixj, fraction=0.046, pad=0.04)
        self._sixj_colorbar.ax.tick_params(colors='white', labelsize=7)
        self._sixj_colorbar.set_label('{6j}', color='white', fontsize=9)
        
        # Add grid
        for i in range(matrix.shape[0] + 1):
            self.ax_sixj.axhline(i - 0.5, color='#444444', linewidth=0.5)
            self.ax_sixj.axvline(i - 0.5, color='#444444', linewidth=0.5)
        
        # Labels
        self.ax_sixj.set_xlabel('Edge k', color='white', fontsize=9)
        self.ax_sixj.set_ylabel('Edge i', color='white', fontsize=9)
        self.ax_sixj.tick_params(colors='white', labelsize=8)
        
        # Add annotation for formula
        self.ax_sixj.text(0.02, 0.98, 
                         r'$\{j_1\ j_2\ j_3; j_4\ j_5\ j_6\}$',
                         transform=self.ax_sixj.transAxes,
                         color='cyan', fontsize=8, va='top',
                         bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.8))
        
        self.ax_sixj.set_title("6j-Symbol Recoupling Matrix\n(Racah-Wigner Coefficients)", 
                              color='white', fontsize=11, fontweight='bold')
    
    def _plot_semiclassical_limit(self):
        """Plot semi-classical limit analysis showing DNN weight emergence"""
        self.ax_semiclass.clear()
        
        weights = self.processor.semiclassical_weights
        
        if len(weights) == 0:
            self.ax_semiclass.text(0.5, 0.5, "Semi-classical weights\nwill appear here", 
                                  ha='center', va='center', color='gray',
                                  transform=self.ax_semiclass.transAxes, fontsize=12)
            self.ax_semiclass.set_title("Semi-Classical Limit: $w_i = j_i/N$\n(DNN Weight Emergence)", 
                                       color='white', fontsize=11, fontweight='bold')
            return
        
        # Subsample weights for visualization
        n_show = min(50, len(weights))
        indices = np.linspace(0, len(weights)-1, n_show).astype(int)
        w_sample = weights[indices]
        
        # Create dual plot: weights distribution and convergence
        
        # Left: Weight values
        colors = [seqCmap(0.2 + 0.6 * w) for w in w_sample]
        bars = self.ax_semiclass.bar(range(n_show), w_sample, color=colors,
                                    edgecolor='none', alpha=0.8, width=0.8)
        
        # Add classical activation line
        activation = self.processor.classical_activation
        self.ax_semiclass.axhline(y=activation, color='#ff6b6b', 
                                 linestyle='--', linewidth=2, label=f'σ(w·x) = {activation:.3f}')
        
        # Add quantum correction annotation
        quantum_corr = 1.0 / np.sqrt(self.processor.N_large)
        self.ax_semiclass.axhspan(activation - quantum_corr, activation + quantum_corr,
                                 alpha=0.2, color='#ff6b6b', 
                                 label=f'Quantum corr. ±{quantum_corr:.4f}')
        
        self.ax_semiclass.set_xlabel('Edge index i', color='white', fontsize=9)
        self.ax_semiclass.set_ylabel('Weight $w_i = (j_i - N)/scale$', color='white', fontsize=9)
        self.ax_semiclass.tick_params(colors='white', labelsize=8)
        self.ax_semiclass.grid(True, alpha=0.2, color='gray', axis='y')
        self.ax_semiclass.legend(loc='upper right', fontsize=7, 
                                facecolor='#1a1a1a', edgecolor='#444444',
                                labelcolor='white')
        
        self.ax_semiclass.set_ylim(-0.1, 1.1)
        
        # Add N_large annotation
        self.ax_semiclass.text(0.02, 0.98, 
                              f'N = {self.processor.N_large}\n'
                              f'ℏ_eff ~ 1/N = {1/self.processor.N_large:.4f}',
                              transform=self.ax_semiclass.transAxes,
                              color='cyan', fontsize=8, va='top',
                              bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.8))
        
        self.ax_semiclass.set_title("Semi-Classical Limit: $w_i = j_i/N$\n(DNN Weight Emergence)", 
                                   color='white', fontsize=11, fontweight='bold')
    
    def _start_animation(self):
        """Start the animation loop"""
        self.animation_running = True
        self._animation_step()
    
    def _animation_step(self):
        """Single animation step"""
        if not self.animation_running:
            return
        
        self.animation_frame += 1
        
        # Schedule next frame
        self.root.after(self.update_interval, self._animation_step)
    
    def run(self):
        """Start the GUI main loop"""
        self._update_status("══════════════════════════════════════════════════\n"
                           "    TQNN SPIN-NETWORK SIMULATOR READY\n"
                           "══════════════════════════════════════════════════\n\n"
                           "This simulator implements the Marcianò-Zappalà\n"
                           "framework for Topological Quantum Neural Networks.\n\n"
                           "DRAW patterns to encode as spin-networks.\n"
                           "OBSERVE real-time TQFT amplitude computation.\n"
                           "ADJUST N_large to explore semi-classical limit.\n\n"
                           "Key formulas:\n"
                           "• Spin encoding: j_i = N + ⌊x_i⌋\n"
                           "• Amplitude: A = ∏ Δ_j exp(-(j-j̄)²/2σ²)\n"
                           "• Quantum dim: Δ_j = 2j + 1\n"
                           "• Semi-classical: w_i = j_i/N as N→∞\n"
                           "══════════════════════════════════════════════════")
        self.root.mainloop()


def main():
    """Main entry point for the application"""
    print("═" * 60)
    print("  TQNN SPIN-NETWORK SIMULATOR")
    print("  Marcianò-Zappalà Framework Implementation")
    print("═" * 60)
    print("\nThis simulator implements Topological Quantum Neural Networks")
    print("based on the TQFT formalism from the reference papers.\n")
    print("Key Features:")
    print("• Spin-network encoding: j_i = N + ⌊x_i⌋")
    print("• TQFT transition amplitude: A = ∏ Δ_j exp(-(j-j̄)²/2σ²)")
    print("• 6j-symbol (Racah-Wigner) recoupling computation")
    print("• Semi-classical limit: w_i = j_i/N → classical DNN weights")
    print("• Real-time classification via physical scalar products")
    print("\nStarting GUI...")
    
    try:
        app = TQNNVisualizerGUI()
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
