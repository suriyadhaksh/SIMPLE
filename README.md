# SIMPLE Algorithm for 2D Lid-Driven Cavity Flow

This repository implements the SIMPLE (Semi-Implicit Method for Pressure-Linked Equations) algorithm to solve the steady-state, incompressible Navier–Stokes equations for a 2D lid-driven cavity flow problem.

## Table of Contents

* [Overview](#overview)
* [Installation](#installation)
* [Configuration](#configuration)
* [Running the Solver](#running-the-solver)
* [Output and Usage](#output-and-usage)
* [Postprocessing](#postprocessing)
* [Contact](#Contact)

## Overview

The `SIMPLE.py` script sets up a staggered grid, applies under-relaxed Gauss–Seidel iterations for the momentum predictors, solves the pressure correction via BiCGStab (or direct solver fallback), and updates velocity and pressure until convergence. Results are saved to `results.npz`.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/suriyadhaksh/SIMPLE.git
cd SIMPLE
```

### 2. Install Python Dependencies

Use `pip` to install the required libraries:

```bash
pip install numpy scipy matplotlib
```

## Configuration

All key parameters are defined at the top of `SIMPLE.py`. Edit these to customize your simulation:

```python
# Grid parameters
nx_cells = 100       # Number of cells in x-direction
ny_cells = 100       # Number of cells in y-direction

# Problem parameters
Re = 100.0           # Reynolds number

# Under-relaxation factors
alpha_u = 0.8        # u-velocity
alpha_v = 0.8        # v-velocity
alpha_p = 0.4        # pressure

# Iterative solver settings
momentum_gs_iter_max = 12   # Gauss–Seidel iterations for momentum
bicgstab_iter_max   = 1000  # Max BiCGStab iterations
bicgstab_tol        = 1e-8  # BiCGStab tolerance
SIMPLE_iter_max     = 10000 # Max SIMPLE loops
SIMPLE_tol          = 1e-8  # SIMPLE convergence tolerance
```

Modify these values to refine grid resolution, change flow regime, or adjust solver behavior.

## Running the Solver

Execute the main script:

```bash
python SIMPLE.py
```

This prints convergence information and saves the output as `results.npz`.

## Output and Usage

The solver produces `results.npz` containing:

* `X`, `Y`: 2D meshgrid of cell-center coordinates
* `p`: Pressure field
* `u`: u-velocity field
* `v`: v-velocity field

### Example: Loading and Inspecting Results

```python
import numpy as np

data = np.load('results.npz')
print(data.files)
# ['X', 'Y', 'p', 'u', 'v']
shapes = {key: data[key].shape for key in data.files}
print(shapes)
# e.g. {'X': (100, 100), 'Y': (100, 100), 'p': (100, 100), 'u': (100, 100), 'v': (100, 100)}
```

## Postprocessing

Example scripts for plotting pressure contours and velocity streamlines are available in the `postprocessing` directory:

```
postprocessing/
├── contourplot/contour_gen.py
├── midsection/midsection_gen.py
└── streamfunc/streamline_gen.py
```

## Contact
For questions or further information, please contact:

**Suriya Dhakshinamoorthy**  
_snarayan@iastate.edu_
