# Open Problems: Topological-Quantum-Neural-Networks (TQNN)

This document catalogs open problems, algorithmic corrections, and architectural refactoring targets for the **Topological-Quantum-Neural-Networks** (TQNN) spin-network simulation and classification codebase (`tqnn/simulation/gui.py`, `tqnn/helpers.py`, `tqnn/cobordism/gui.py`).

---

## 1. Algorithmic & Implementation Problems

- **Exact Wigner 6j-Symbol Evaluation (`C1`)**
  - **Problem**: `compute_six_j_symbol()` in `tqnn/simulation/gui.py` computes an unphysical cosine approximation (`amplitude * cos(pi * sum(j) / 4)`) instead of evaluating true Wigner 6j-symbols or Ponzano-Regge asymptotics.
  - **Context**: Tracked in `AUDIT-2026-04-14.md`. Requires replacing with `sympy.physics.wigner.wigner_6j` for small spins or calculating tetrahedral Cayley-Menger volumes $V$ and Regge actions $S_R = \sum_i (j_i + 1/2)\theta_i$ for the asymptotic limit $\frac{1}{\sqrt{12\pi V}} \cos(S_R + \pi/4)$.
- **Clebsch-Gordan SU(2) Recoupling Matrix Construction (`H1`)**
  - **Problem**: `compute_recoupling_matrix()` in `tqnn/simulation/gui.py` computes intermediate spins by averaging ($j_3 = (j_i + j_k)/2$) instead of enumerating valid total angular momenta from the Clebsch-Gordan series ($|j_1 - j_2| \leq j_3 \leq j_1 + j_2$) and constructing a proper unitary Wigner 6j transformation matrix.
- **Quantum Dimension Term in Topological Log-Probability (`H2`)**
  - **Problem**: `_calculate_log_probability()` in `tqnn/helpers.py` omits the quantum dimension weighting term $\sum_i \log(2j_i + 1)$, causing numerical divergence from the cobordism evaluator (`tqnn/cobordism/gui.py`).

---

## 2. Bugs & Unresolved Issues

- **Broad Exception Catching in Simulation Loop (`H3`)**
  - **Problem**: Bare `except:` clauses in `tqnn/simulation/gui.py` capture and swallow `KeyboardInterrupt` and `SystemExit`, preventing clean process termination.
- **Documentation Overclaims (`C2`)**
  - **Problem**: `README.md` claims Matrix Product State (MPS) decompositions and exact TQFT amplitudes in the feature table despite neither being implemented in the codebase.

---

## 3. Theoretical & Scientific Problems

- **Ponzano-Regge Asymptotic Error Bounds at Small Half-Integer Spins**
  - **Problem**: Establishing the exact spin threshold $j_{\text{crit}}$ where the Ponzano-Regge asymptotic formula deviates by more than 1% from exact Racah coefficients, particularly for $j \in \{1/2, 1, 3/2\}$.

---

## 4. Code Maintenance & Refactoring Opportunities

- **Decoupling `TQNNProcessor` from GUI Monolith (`H5, H6`)**
  - **Opportunity**: `tqnn/simulation/gui.py` is a 1,627-line monolith combining Tkinter GUI views with backend spin-network simulation logic. Extracting `TQNNProcessor` into `tqnn/processor.py` will enable automated unit testing (`tests/test_processor.py`).
- **Style Guide Alignment (`H4`)**
  - **Opportunity**: Pervasive snake_case naming throughout `tqnn/` conflicts with project camelCase mandates; requires systematic migration during future backend refactoring.
