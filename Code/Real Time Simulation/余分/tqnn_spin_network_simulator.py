"""
Real-Time TQNN Spin-Network Evolution Simulator

This module implements an interactive real-time visualization of Topological Quantum
Neural Network (TQNN) spin-network evolution through cobordism layers. Users can:

- Construct hexagonal spin-networks interactively
- Manipulate SU(2) spin labels in real-time
- Watch TQFT functor evolution through cobordism layers
- Observe transition amplitude calculations with 6j-symbols
- Visualize semi-classical limit convergence
- Explore topological charge conservation
- See recoupling theory in action

Based on the theoretical framework of Marcianò & Zappalà's TQNN papers,
this simulator brings the mathematical formalism to life through interactive
3D visualization.

GUI Framework: tkinter with custom dark theme
Rendering: Custom isometric 3D hexagonal lattice
Backend: Real-time quantum recoupling calculations
Architecture: Event-driven with TQFT computational core

Author: TQNN Research Project
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from collections import deque
import time

# Scientific computing
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns

# Set up color palettes for consistency with project standards
sns.set_style("darkgrid")
seqCmap = sns.color_palette("mako", as_cmap=True)
divCmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
lightCmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)


class SpinValue(Enum):
    """SU(2) spin values for quantum labels"""
    SPIN_0 = 0
    SPIN_HALF = 0.5
    SPIN_1 = 1
    SPIN_3_HALF = 1.5
    SPIN_2 = 2
    SPIN_5_HALF = 2.5
    SPIN_3 = 3
    
    def quantum_dimension(self):
        """Calculate quantum dimension Δ_j = 2j + 1"""
        return 2 * self.value + 1
    
    def __str__(self):
        if self.value == int(self.value):
            return f"j={int(self.value)}"
        else:
            # Format as fraction
            numerator = int(self.value * 2)
            return f"j={numerator}/2"


@dataclass
class SpinNode:
    """
    Represents a node in the hexagonal spin-network.
    
    Attributes:
        position: (x, y, z) coordinates in hexagonal lattice
        spin: SU(2) spin label
        layer: Cobordism layer index
        amplitude: Complex transition amplitude
        color: RGB color for rendering
        neighbors: Connected node IDs
        is_intertwiner: Whether this is an intertwiner vertex
    """
    position: Tuple[float, float, float]
    spin: SpinValue = SpinValue.SPIN_HALF
    layer: int = 0
    amplitude: complex = 1.0 + 0j
    color: Tuple[float, float, float] = (0.5, 0.7, 0.9)
    neighbors: List[int] = field(default_factory=list)
    is_intertwiner: bool = False
    belief_value: float = 0.5  # For semi-classical limit


@dataclass
class CobordismLayer:
    """
    Represents a cobordism layer in the TQFT evolution.
    
    Each layer is a 2-complex with spin-networks on boundaries.
    """
    layer_id: int
    nodes: List[SpinNode]
    transition_amplitude: complex = 1.0 + 0j
    entropy: float = 0.0
    topological_charge: float = 0.0
    recoupling_coefficient: float = 1.0


class HexagonalLatticeRenderer:
    """
    Handles rendering of hexagonal spin-network lattice on tkinter canvas.
    
    Implements isometric projection for 3D hexagonal grid visualization
    with proper depth ordering and visual clarity.
    """
    
    def __init__(self, canvas: tk.Canvas, scale: float = 25.0):
        """
        Initialize the hexagonal lattice renderer.
        
        Args:
            canvas: tkinter Canvas widget for rendering
            scale: Scaling factor for hexagonal cells
        """
        self.canvas = canvas
        self.scale = scale
        self.offset_x = 450
        self.offset_y = 250
        
        # Hexagonal geometry
        self.hex_angle = math.pi / 3  # 60 degrees
        self.hex_radius = scale
        
        # Isometric angles
        self.iso_angle_x = math.radians(30)
        self.iso_angle_y = math.radians(30)
        
    def hex_to_pixel(self, q: int, r: int, layer: int = 0) -> Tuple[float, float]:
        """
        Convert hexagonal axial coordinates to pixel position.
        
        Args:
            q: Axial coordinate q
            r: Axial coordinate r
            layer: Vertical layer (z-coordinate)
            
        Returns:
            Tuple of (x, y) pixel coordinates
        """
        # Pointy-top hexagon orientation
        x = self.hex_radius * (math.sqrt(3) * q + math.sqrt(3)/2 * r)
        y = self.hex_radius * (3/2 * r)
        z = layer * self.scale * 2
        
        # Apply isometric projection
        iso_x = x * math.cos(self.iso_angle_x) - y * math.cos(self.iso_angle_y)
        iso_y = x * math.sin(self.iso_angle_x) + y * math.sin(self.iso_angle_y) - z
        
        return iso_x + self.offset_x, iso_y + self.offset_y
    
    def draw_hexagon(self, q: int, r: int, layer: int, color: Tuple[float, float, float],
                    outline: str = "#333333", width: int = 2) -> List[int]:
        """
        Draw a hexagonal cell.
        
        Args:
            q, r: Hexagonal coordinates
            layer: Vertical layer
            color: RGB color tuple (0-1 range)
            outline: Outline color
            width: Line width
            
        Returns:
            List of canvas item IDs
        """
        center_x, center_y = self.hex_to_pixel(q, r, layer)
        
        # Calculate hexagon vertices
        vertices = []
        for i in range(6):
            angle = self.hex_angle * i
            x = center_x + self.hex_radius * math.cos(angle)
            y = center_y + self.hex_radius * math.sin(angle)
            vertices.extend([x, y])
        
        # Convert color to hex
        hex_color = self._rgb_to_hex(color)
        
        # Draw hexagon
        item_id = self.canvas.create_polygon(vertices, fill=hex_color, 
                                            outline=outline, width=width,
                                            tags="hexagon")
        
        return [item_id]
    
    def draw_spin_label(self, q: int, r: int, layer: int, spin: SpinValue,
                       amplitude: complex = 1.0 + 0j) -> int:
        """
        Draw spin label and amplitude on hexagon.
        
        Args:
            q, r: Hexagonal coordinates
            layer: Vertical layer
            spin: Spin value
            amplitude: Complex amplitude
            
        Returns:
            Canvas text item ID
        """
        center_x, center_y = self.hex_to_pixel(q, r, layer)
        
        # Format spin label
        spin_text = str(spin)
        
        # Add amplitude information
        amp_magnitude = abs(amplitude)
        amp_text = f"|A|={amp_magnitude:.2f}"
        
        # Draw spin value
        text_id = self.canvas.create_text(center_x, center_y - 8, 
                                         text=spin_text,
                                         fill="#ffffff", 
                                         font=("Arial", 10, "bold"),
                                         tags="label")
        
        # Draw amplitude
        amp_id = self.canvas.create_text(center_x, center_y + 8,
                                        text=amp_text,
                                        fill="#ffff00",
                                        font=("Arial", 8),
                                        tags="label")
        
        return text_id
    
    def draw_edge(self, q1: int, r1: int, layer1: int,
                 q2: int, r2: int, layer2: int,
                 color: str = "#00ffaa", width: int = 2) -> int:
        """
        Draw edge between two hexagons.
        
        Args:
            q1, r1, layer1: First hexagon coordinates
            q2, r2, layer2: Second hexagon coordinates
            color: Edge color
            width: Line width
            
        Returns:
            Canvas line item ID
        """
        x1, y1 = self.hex_to_pixel(q1, r1, layer1)
        x2, y2 = self.hex_to_pixel(q2, r2, layer2)
        
        line_id = self.canvas.create_line(x1, y1, x2, y2,
                                         fill=color, width=width,
                                         tags="edge")
        return line_id
    
    def _rgb_to_hex(self, color: Tuple[float, float, float]) -> str:
        """Convert RGB tuple (0-1 range) to hex color string."""
        r, g, b = [int(c * 255) for c in color]
        return f"#{r:02x}{g:02x}{b:02x}"


class TQFTComputationEngine:
    """
    Computational backend for TQFT calculations.
    
    Implements:
    - Transition amplitude calculations
    - 6j-symbol recoupling
    - Semi-classical limit
    - Topological charge conservation
    """
    
    def __init__(self, n_large: int = 1000):
        """
        Initialize TQFT computation engine.
        
        Args:
            n_large: Large N parameter for semi-classical limit
        """
        self.n_large = n_large
        self.cache_6j = {}  # Cache for 6j-symbol calculations
        
    def calculate_transition_amplitude(self, layer_input: CobordismLayer,
                                       layer_output: CobordismLayer) -> complex:
        """
        Calculate transition amplitude between two cobordism layers.
        
        Based on the formula:
        A = Σ Δ_j * exp(-(j-j̄)²/(2σ²)) * exp(-iξj)
        
        Args:
            layer_input: Input cobordism layer
            layer_output: Output cobordism layer
            
        Returns:
            Complex transition amplitude
        """
        amplitude = 1.0 + 0j
        
        # Pair up corresponding nodes
        min_nodes = min(len(layer_input.nodes), len(layer_output.nodes))
        
        for i in range(min_nodes):
            node_in = layer_input.nodes[i]
            node_out = layer_output.nodes[i]
            
            # Quantum dimension
            j_in = node_in.spin.value
            j_out = node_out.spin.value
            delta_j = node_in.spin.quantum_dimension()
            
            # Gaussian term (simplified)
            sigma = 0.5
            gaussian = np.exp(-((j_in - j_out)**2) / (2 * sigma**2))
            
            # Phase term (simplified)
            xi = 0.1
            phase = np.exp(-1j * xi * j_in)
            
            # Combine
            node_amplitude = delta_j * gaussian * phase
            amplitude *= node_amplitude
        
        return amplitude
    
    def calculate_6j_symbol(self, j1: float, j2: float, j3: float,
                           j4: float, j5: float, j6: float) -> float:
        """
        Calculate Racah-Wigner 6j-symbol (simplified).
        
        The full calculation is complex; this is a simplified version
        for visualization purposes.
        
        Args:
            j1-j6: Six spin values
            
        Returns:
            6j-symbol value
        """
        # Check cache
        key = (j1, j2, j3, j4, j5, j6)
        if key in self.cache_6j:
            return self.cache_6j[key]
        
        # Simplified calculation (not physically accurate, for demo)
        # Real implementation would use Racah formula
        if self._is_admissible_triple(j1, j2, j3) and \
           self._is_admissible_triple(j4, j5, j6):
            # Simplified formula
            value = np.exp(-0.1 * (j1 + j2 + j3 + j4 + j5 + j6))
            value *= (-1) ** int(j1 + j2 + j3 + j4 + j5 + j6)
        else:
            value = 0.0
        
        self.cache_6j[key] = value
        return value
    
    def _is_admissible_triple(self, j1: float, j2: float, j3: float) -> bool:
        """Check if (j1, j2, j3) form an admissible triple (triangle inequality)."""
        return (abs(j1 - j2) <= j3 <= j1 + j2) and \
               ((j1 + j2 + j3) == int(j1 + j2 + j3))
    
    def calculate_semiclassical_weights(self, layer: CobordismLayer) -> np.ndarray:
        """
        Calculate semi-classical weights from spin-network.
        
        In the large-N limit, spins become classical weights:
        w_i = <j_i> / N_large
        
        Args:
            layer: Cobordism layer
            
        Returns:
            Array of classical weights
        """
        weights = []
        for node in layer.nodes:
            weight = (self.n_large + node.spin.value) / self.n_large
            weights.append(weight)
        
        return np.array(weights)
    
    def calculate_topological_charge(self, layer: CobordismLayer) -> float:
        """
        Calculate topological charge for a layer.
        
        Q = Σ_i q_i where q_i depends on spin configuration
        
        Args:
            layer: Cobordism layer
            
        Returns:
            Total topological charge
        """
        charge = 0.0
        for node in layer.nodes:
            # Charge depends on spin value and neighbors
            node_charge = node.spin.value * (1 if node.is_intertwiner else -1)
            charge += node_charge
        
        return charge
    
    def calculate_recoupling_coefficient(self, spins: List[float]) -> float:
        """
        Calculate recoupling coefficient using 6j-symbols.
        
        Args:
            spins: List of spin values
            
        Returns:
            Recoupling coefficient
        """
        if len(spins) < 6:
            return 1.0
        
        # Use first 6 spins for 6j-symbol
        sixj = self.calculate_6j_symbol(*spins[:6])
        
        # Coefficient includes quantum dimensions
        coeff = sixj
        for s in spins[:6]:
            coeff *= (2 * s + 1)
        
        return abs(coeff)


class TQNNSpinNetworkSimulator:
    """
    Main application class for TQNN Spin-Network Evolution Simulator.
    
    Manages GUI, user interactions, and coordinates between rendering
    and computational backend.
    """
    
    def __init__(self):
        """Initialize the TQNN simulator."""
        self.root = self._setup_gui()
        
        # Simulation state
        self.layers: List[CobordismLayer] = []
        self.current_layer = 0
        self.selected_node = None
        self.node_counter = 0
        
        # Hexagonal grid parameters
        self.grid_size = 5  # 5x5 hexagonal grid
        self.num_layers = 3
        
        # Computation engine
        self.tqft_engine = TQFTComputationEngine()
        
        # Rendering
        self.renderer = None
        
        # Animation state
        self.animation_running = False
        self.animation_speed = 500  # ms
        self.evolution_step = 0
        
        # Performance tracking
        self.frame_times = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Metrics history
        self.amplitude_history = deque(maxlen=100)
        self.charge_history = deque(maxlen=100)
        
        # Setup UI and initialize
        self._setup_ui()
        self._initialize_spin_network()
        self._bind_events()
        
    def _setup_gui(self) -> tk.Tk:
        """Set up the main GUI window with dark theme."""
        root = tk.Tk()
        root.title("Real-Time TQNN Spin-Network Evolution Simulator")
        root.geometry("1600x1000")
        root.configure(bg='#1e1e1e')
        
        # Configure dark theme style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Dark theme colors
        bg_dark = '#1e1e1e'
        bg_medium = '#2b2b2b'
        bg_light = '#404040'
        fg_color = '#ffffff'
        accent_color = '#00ff88'
        
        style.configure('Dark.TFrame', background=bg_dark)
        style.configure('Dark.TLabel', background=bg_dark, foreground=fg_color)
        style.configure('Dark.TLabelframe', background=bg_dark, foreground=fg_color,
                       borderwidth=2, relief='solid')
        style.configure('Dark.TLabelframe.Label', background=bg_dark, foreground=accent_color,
                       font=('TkDefaultFont', 10, 'bold'))
        style.configure('Dark.TButton', background=bg_light, foreground=fg_color)
        style.configure('Dark.Horizontal.TScale', background=bg_dark)
        
        return root
    
    def _setup_ui(self):
        """Set up the user interface components."""
        # Main container
        main_frame = ttk.Frame(self.root, style='Dark.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Hexagonal lattice canvas
        left_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Canvas for hexagonal lattice
        self.canvas = tk.Canvas(left_frame, bg='#0a0a0a', highlightthickness=2,
                               highlightbackground='#404040')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Initialize renderer
        self.renderer = HexagonalLatticeRenderer(self.canvas)
        
        # Right panel - Controls and metrics
        right_frame = ttk.Frame(main_frame, style='Dark.TFrame', width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)
        
        self._setup_control_panel(right_frame)
        self._setup_metrics_panel(right_frame)
        self._setup_status_panel(right_frame)
        
    def _setup_control_panel(self, parent):
        """Setup control panel with sliders and buttons."""
        control_frame = ttk.LabelFrame(parent, text="Evolution Controls",
                                      style='Dark.TLabelframe', padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Layer selector
        layer_frame = ttk.Frame(control_frame, style='Dark.TFrame')
        layer_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(layer_frame, text="Current Layer:", style='Dark.TLabel').pack(side=tk.LEFT)
        self.layer_var = tk.IntVar(value=0)
        self.layer_scale = ttk.Scale(layer_frame, from_=0, to=self.num_layers-1,
                                    variable=self.layer_var, orient=tk.HORIZONTAL,
                                    command=self.on_layer_change, style='Dark.Horizontal.TScale')
        self.layer_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.layer_label = ttk.Label(layer_frame, text="0", style='Dark.TLabel', width=3)
        self.layer_label.pack(side=tk.LEFT)
        
        # N_large parameter (semi-classical limit)
        n_frame = ttk.Frame(control_frame, style='Dark.TFrame')
        n_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(n_frame, text="N_large:", style='Dark.TLabel').pack(side=tk.LEFT)
        self.n_var = tk.IntVar(value=1000)
        self.n_scale = ttk.Scale(n_frame, from_=100, to=5000,
                                variable=self.n_var, orient=tk.HORIZONTAL,
                                command=self.on_n_change, style='Dark.Horizontal.TScale')
        self.n_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.n_label = ttk.Label(n_frame, text="1000", style='Dark.TLabel', width=5)
        self.n_label.pack(side=tk.LEFT)
        
        # Animation speed
        speed_frame = ttk.Frame(control_frame, style='Dark.TFrame')
        speed_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(speed_frame, text="Speed (ms):", style='Dark.TLabel').pack(side=tk.LEFT)
        self.speed_var = tk.IntVar(value=500)
        self.speed_scale = ttk.Scale(speed_frame, from_=100, to=2000,
                                    variable=self.speed_var, orient=tk.HORIZONTAL,
                                    command=self.on_speed_change, style='Dark.Horizontal.TScale')
        self.speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.speed_label = ttk.Label(speed_frame, text="500", style='Dark.TLabel', width=5)
        self.speed_label.pack(side=tk.LEFT)
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Action buttons
        button_frame = ttk.Frame(control_frame, style='Dark.TFrame')
        button_frame.pack(fill=tk.X)
        
        self.btn_evolve = ttk.Button(button_frame, text="Evolve Layer",
                                     command=self.evolve_layer, style='Dark.TButton')
        self.btn_evolve.pack(fill=tk.X, pady=2)
        
        self.btn_animate = ttk.Button(button_frame, text="Start Animation",
                                      command=self.toggle_animation, style='Dark.TButton')
        self.btn_animate.pack(fill=tk.X, pady=2)
        
        self.btn_recalculate = ttk.Button(button_frame, text="Recalculate Amplitudes",
                                         command=self.recalculate_amplitudes, style='Dark.TButton')
        self.btn_recalculate.pack(fill=tk.X, pady=2)
        
        self.btn_reset = ttk.Button(button_frame, text="Reset Network",
                                    command=self.reset_network, style='Dark.TButton')
        self.btn_reset.pack(fill=tk.X, pady=2)
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Spin selector
        spin_frame = ttk.LabelFrame(control_frame, text="Spin Selection",
                                   style='Dark.TLabelframe', padding=5)
        spin_frame.pack(fill=tk.X, pady=5)
        
        self.spin_var = tk.StringVar(value="j=1/2")
        for spin in SpinValue:
            rb = ttk.Radiobutton(spin_frame, text=str(spin), variable=self.spin_var,
                                value=str(spin), style='Dark.TRadiobutton')
            rb.pack(anchor=tk.W)
        
    def _setup_metrics_panel(self, parent):
        """Setup metrics display panel."""
        metrics_frame = ttk.LabelFrame(parent, text="TQFT Metrics",
                                      style='Dark.TLabelframe', padding=10)
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create matplotlib figure for metrics
        self.metrics_fig = Figure(figsize=(4, 6), facecolor='#1e1e1e')
        self.metrics_fig.subplots_adjust(hspace=0.4)
        
        # Amplitude plot
        self.ax_amplitude = self.metrics_fig.add_subplot(311)
        self.ax_amplitude.set_title("Transition Amplitude", color='white', fontsize=10)
        self.ax_amplitude.set_facecolor('#0a0a0a')
        self.ax_amplitude.tick_params(colors='white', labelsize=8)
        self.ax_amplitude.spines['bottom'].set_color('white')
        self.ax_amplitude.spines['left'].set_color('white')
        
        # Topological charge plot
        self.ax_charge = self.metrics_fig.add_subplot(312)
        self.ax_charge.set_title("Topological Charge", color='white', fontsize=10)
        self.ax_charge.set_facecolor('#0a0a0a')
        self.ax_charge.tick_params(colors='white', labelsize=8)
        self.ax_charge.spines['bottom'].set_color('white')
        self.ax_charge.spines['left'].set_color('white')
        
        # Semi-classical weights
        self.ax_weights = self.metrics_fig.add_subplot(313)
        self.ax_weights.set_title("Semi-Classical Weights", color='white', fontsize=10)
        self.ax_weights.set_facecolor('#0a0a0a')
        self.ax_weights.tick_params(colors='white', labelsize=8)
        self.ax_weights.spines['bottom'].set_color('white')
        self.ax_weights.spines['left'].set_color('white')
        
        # Embed in tkinter
        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_fig, metrics_frame)
        self.metrics_canvas.draw()
        self.metrics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _setup_status_panel(self, parent):
        """Setup status display panel."""
        status_frame = ttk.LabelFrame(parent, text="System Status",
                                     style='Dark.TLabelframe', padding=10)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Status text
        self.status_text = tk.Text(status_frame, height=8, bg='#0a0a0a', fg='#00ff88',
                                  insertbackground='#00ff88', font=('Courier', 9))
        self.status_text.pack(fill=tk.X)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(status_frame, command=self.status_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.config(yscrollcommand=scrollbar.set)
        
        self._log_status("TQNN Spin-Network Simulator initialized")
        self._log_status("Based on Marcianò & Zappalà framework")
        self._log_status("Ready for interactive spin-network construction")
        
    def _initialize_spin_network(self):
        """Initialize the spin-network with default configuration."""
        # Create cobordism layers
        for layer_id in range(self.num_layers):
            nodes = []
            
            # Create hexagonal grid of nodes
            for q in range(-self.grid_size//2, self.grid_size//2 + 1):
                for r in range(-self.grid_size//2, self.grid_size//2 + 1):
                    if abs(q + r) <= self.grid_size//2:
                        # Create node
                        x, y = self.renderer.hex_to_pixel(q, r, layer_id)
                        node = SpinNode(
                            position=(q, r, layer_id),
                            spin=SpinValue.SPIN_HALF,
                            layer=layer_id,
                            amplitude=1.0 + 0j,
                            color=self._get_spin_color(SpinValue.SPIN_HALF)
                        )
                        nodes.append(node)
            
            # Create layer
            layer = CobordismLayer(
                layer_id=layer_id,
                nodes=nodes,
                transition_amplitude=1.0 + 0j,
                entropy=0.0,
                topological_charge=0.0,
                recoupling_coefficient=1.0
            )
            
            self.layers.append(layer)
        
        # Calculate initial amplitudes
        self.recalculate_amplitudes()
        
        # Draw initial state
        self._draw_network()
        
        self._log_status(f"Initialized {self.num_layers} cobordism layers")
        self._log_status(f"Total nodes: {sum(len(l.nodes) for l in self.layers)}")
        
    def _bind_events(self):
        """Bind mouse and keyboard events."""
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)
        self.root.bind("<space>", lambda e: self.toggle_animation())
        self.root.bind("<Return>", lambda e: self.evolve_layer())
        
    def _draw_network(self):
        """Draw the entire spin-network."""
        # Clear canvas
        self.canvas.delete("all")
        
        # Draw all layers
        current_layer_obj = self.layers[self.current_layer]
        
        # Draw nodes
        for node in current_layer_obj.nodes:
            q, r, layer = node.position
            
            # Draw hexagon
            self.renderer.draw_hexagon(q, r, layer, node.color)
            
            # Draw spin label
            self.renderer.draw_spin_label(q, r, layer, node.spin, node.amplitude)
        
        # Draw edges (connections between adjacent hexagons)
        for i, node in enumerate(current_layer_obj.nodes):
            q1, r1, layer1 = node.position
            
            # Check adjacent hexagons
            adjacent = [
                (q1+1, r1, layer1), (q1-1, r1, layer1),
                (q1, r1+1, layer1), (q1, r1-1, layer1),
                (q1+1, r1-1, layer1), (q1-1, r1+1, layer1)
            ]
            
            for q2, r2, layer2 in adjacent:
                # Find matching node
                for j, other_node in enumerate(current_layer_obj.nodes):
                    if j > i and other_node.position == (q2, r2, layer2):
                        self.renderer.draw_edge(q1, r1, layer1, q2, r2, layer2)
                        break
        
        # Draw layer information
        self._draw_layer_info()
        
    def _draw_layer_info(self):
        """Draw layer information overlay."""
        layer = self.layers[self.current_layer]
        
        info_text = (f"Layer {layer.layer_id + 1}/{self.num_layers}\n"
                    f"|A| = {abs(layer.transition_amplitude):.4f}\n"
                    f"Q = {layer.topological_charge:.2f}\n"
                    f"Nodes: {len(layer.nodes)}")
        
        self.canvas.create_text(50, 30, text=info_text, fill="#00ff88",
                               font=("Courier", 10, "bold"), anchor=tk.NW,
                               tags="info")
        
        # Evolution step
        self.canvas.create_text(self.canvas.winfo_reqwidth() - 50, 30,
                               text=f"Step: {self.evolution_step}",
                               fill="#ffaa00", font=("Arial", 10, "bold"),
                               anchor=tk.NE, tags="info")
        
    def _get_spin_color(self, spin: SpinValue) -> Tuple[float, float, float]:
        """Get color for spin value."""
        # Map spin values to colors
        spin_colors = {
            SpinValue.SPIN_0: (0.3, 0.3, 0.5),
            SpinValue.SPIN_HALF: (0.5, 0.7, 0.9),
            SpinValue.SPIN_1: (0.3, 0.8, 0.6),
            SpinValue.SPIN_3_HALF: (0.9, 0.7, 0.3),
            SpinValue.SPIN_2: (0.9, 0.4, 0.4),
            SpinValue.SPIN_5_HALF: (0.8, 0.3, 0.8),
            SpinValue.SPIN_3: (0.6, 0.3, 0.9)
        }
        return spin_colors.get(spin, (0.5, 0.5, 0.5))
    
    def on_canvas_click(self, event):
        """Handle left click on canvas - select/modify node spin."""
        # Find clicked node
        clicked_node = self._find_node_at_position(event.x, event.y)
        
        if clicked_node:
            # Get selected spin from radio buttons
            selected_spin_str = self.spin_var.get()
            
            # Find matching spin value
            for spin in SpinValue:
                if str(spin) == selected_spin_str:
                    clicked_node.spin = spin
                    clicked_node.color = self._get_spin_color(spin)
                    break
            
            self._log_status(f"Set node at {clicked_node.position[:2]} to {clicked_node.spin}")
            
            # Recalculate and redraw
            self.recalculate_amplitudes()
            self._draw_network()
            self._update_metrics()
    
    def on_canvas_right_click(self, event):
        """Handle right click - cycle through spin values."""
        clicked_node = self._find_node_at_position(event.x, event.y)
        
        if clicked_node:
            # Cycle to next spin value
            spins = list(SpinValue)
            current_idx = spins.index(clicked_node.spin)
            next_idx = (current_idx + 1) % len(spins)
            
            clicked_node.spin = spins[next_idx]
            clicked_node.color = self._get_spin_color(spins[next_idx])
            
            self._log_status(f"Cycled node to {clicked_node.spin}")
            
            # Recalculate and redraw
            self.recalculate_amplitudes()
            self._draw_network()
            self._update_metrics()
    
    def _find_node_at_position(self, x: int, y: int) -> Optional[SpinNode]:
        """Find node at screen position."""
        layer = self.layers[self.current_layer]
        
        for node in layer.nodes:
            q, r, layer_id = node.position
            node_x, node_y = self.renderer.hex_to_pixel(q, r, layer_id)
            
            # Check if click is within hexagon radius
            distance = math.sqrt((x - node_x)**2 + (y - node_y)**2)
            if distance < self.renderer.hex_radius:
                return node
        
        return None
    
    def on_layer_change(self, value):
        """Handle layer slider change."""
        self.current_layer = int(float(value))
        self.layer_label.config(text=str(self.current_layer))
        self._draw_network()
        self._update_metrics()
    
    def on_n_change(self, value):
        """Handle N_large parameter change."""
        n_value = int(float(value))
        self.n_label.config(text=str(n_value))
        self.tqft_engine.n_large = n_value
        self._log_status(f"N_large set to {n_value} (semi-classical limit)")
        self._update_metrics()
    
    def on_speed_change(self, value):
        """Handle animation speed change."""
        speed_value = int(float(value))
        self.speed_label.config(text=str(speed_value))
        self.animation_speed = speed_value
    
    def evolve_layer(self):
        """Evolve to next layer."""
        if self.current_layer < self.num_layers - 1:
            self.current_layer += 1
            self.layer_var.set(self.current_layer)
            self.layer_label.config(text=str(self.current_layer))
            self.evolution_step += 1
            
            self._log_status(f"Evolved to layer {self.current_layer}")
            self._draw_network()
            self._update_metrics()
        else:
            self._log_status("Already at final layer")
    
    def toggle_animation(self):
        """Toggle animation on/off."""
        self.animation_running = not self.animation_running
        
        if self.animation_running:
            self.btn_animate.config(text="Stop Animation")
            self._log_status("Animation started")
            self._animate_step()
        else:
            self.btn_animate.config(text="Start Animation")
            self._log_status("Animation stopped")
    
    def _animate_step(self):
        """Perform one animation step."""
        if not self.animation_running:
            return
        
        # Evolve to next layer
        self.current_layer = (self.current_layer + 1) % self.num_layers
        self.layer_var.set(self.current_layer)
        self.layer_label.config(text=str(self.current_layer))
        self.evolution_step += 1
        
        # Update display
        self._draw_network()
        self._update_metrics()
        
        # Schedule next step
        self.root.after(self.animation_speed, self._animate_step)
    
    def recalculate_amplitudes(self):
        """Recalculate all transition amplitudes."""
        self._log_status("Recalculating transition amplitudes...")
        
        # Calculate amplitudes between consecutive layers
        for i in range(len(self.layers) - 1):
            layer_in = self.layers[i]
            layer_out = self.layers[i + 1]
            
            # Calculate transition amplitude
            amplitude = self.tqft_engine.calculate_transition_amplitude(layer_in, layer_out)
            layer_out.transition_amplitude = amplitude
            
            # Update node amplitudes
            for node in layer_out.nodes:
                node.amplitude = amplitude
            
            # Calculate topological charge
            layer_out.topological_charge = self.tqft_engine.calculate_topological_charge(layer_out)
            
            # Calculate recoupling coefficient
            spins = [node.spin.value for node in layer_out.nodes]
            layer_out.recoupling_coefficient = self.tqft_engine.calculate_recoupling_coefficient(spins)
        
        # Update first layer
        self.layers[0].topological_charge = self.tqft_engine.calculate_topological_charge(self.layers[0])
        
        self._log_status("Amplitudes recalculated")
        self._update_metrics()
    
    def reset_network(self):
        """Reset spin-network to default state."""
        self._log_status("Resetting spin-network...")
        
        self.layers.clear()
        self.current_layer = 0
        self.evolution_step = 0
        self.animation_running = False
        self.btn_animate.config(text="Start Animation")
        
        # Clear history
        self.amplitude_history.clear()
        self.charge_history.clear()
        
        # Reinitialize
        self._initialize_spin_network()
        self._update_metrics()
        
        self._log_status("Network reset complete")
    
    def _update_metrics(self):
        """Update metrics plots."""
        current_layer = self.layers[self.current_layer]
        
        # Update history
        self.amplitude_history.append(abs(current_layer.transition_amplitude))
        self.charge_history.append(current_layer.topological_charge)
        
        # Clear axes
        self.ax_amplitude.clear()
        self.ax_charge.clear()
        self.ax_weights.clear()
        
        # Plot amplitude history
        if len(self.amplitude_history) > 1:
            self.ax_amplitude.plot(list(self.amplitude_history), color='#00ff88', linewidth=2)
            self.ax_amplitude.fill_between(range(len(self.amplitude_history)),
                                          list(self.amplitude_history),
                                          alpha=0.3, color='#00ff88')
        self.ax_amplitude.set_title("Transition Amplitude", color='white', fontsize=10)
        self.ax_amplitude.set_facecolor('#0a0a0a')
        self.ax_amplitude.tick_params(colors='white', labelsize=8)
        self.ax_amplitude.grid(True, alpha=0.2, color='white')
        
        # Plot charge history
        if len(self.charge_history) > 1:
            self.ax_charge.plot(list(self.charge_history), color='#ffaa00', linewidth=2)
        self.ax_charge.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        self.ax_charge.set_title("Topological Charge", color='white', fontsize=10)
        self.ax_charge.set_facecolor('#0a0a0a')
        self.ax_charge.tick_params(colors='white', labelsize=8)
        self.ax_charge.grid(True, alpha=0.2, color='white')
        
        # Plot semi-classical weights
        weights = self.tqft_engine.calculate_semiclassical_weights(current_layer)
        self.ax_weights.bar(range(len(weights)), weights, color='#aa00ff', alpha=0.7)
        self.ax_weights.set_title("Semi-Classical Weights", color='white', fontsize=10)
        self.ax_weights.set_facecolor('#0a0a0a')
        self.ax_weights.tick_params(colors='white', labelsize=8)
        self.ax_weights.set_xlabel("Node Index", color='white', fontsize=8)
        self.ax_weights.set_ylabel("Weight", color='white', fontsize=8)
        self.ax_weights.grid(True, alpha=0.2, color='white')
        
        # Redraw
        self.metrics_canvas.draw()
    
    def _log_status(self, message: str):
        """Log message to status panel."""
        timestamp = time.strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()
    
    def run(self):
        """Start the application main loop."""
        self._log_status("Starting TQNN Spin-Network Simulator")
        self._log_status("Click nodes to change spin values")
        self._log_status("Use controls to evolve through cobordism layers")
        self._log_status("Press SPACE to start/stop animation")
        self._log_status("Press ENTER to evolve one layer")
        
        self.root.mainloop()


def main():
    """Main entry point for the application."""
    try:
        app = TQNNSpinNetworkSimulator()
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
