'''
Created by Suriya Dhakshinamoorthy (2025).
https://github.com/suriyadhaksh/SIMPLE.git
This code implements the SIMPLE algorithm for solving the
steady-state, incompressible Navier-Stokes equations for 
a 2D lid-driven cavity flow problem

Config parameters:
    - nx_cells: Number of cells in the x-direction
    - ny_cells: Number of cells in the y-direction
    - Re: Reynolds number
    - alpha_u: Under-relaxation factor for u-velocity
    - alpha_v: Under-relaxation factor for v-velocity
    - alpha_p: Under-relaxation factor for pressure
    - momentum_gs_iter_max: Maximum iterations for Gauss-Seidel
    - bicgstab_iter_max: Maximum iterations for BiCGStab
    - bicgstab_tol: Tolerance for BiCGStab
    - SIMPLE_iter_max: Maximum iterations for SIMPLE
    - SIMPLE_tol: Tolerance for SIMPLE convergence

Output:
    - Results are saved in a .npz file containing the following grid and velocity fields
    - X, Y: Meshgrid for the cell center grid 
    - p: Pressure field at cell centers
    - u: u-velocity at cell centers
    - v: v-velocity at cell centers
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from bicgstab_solver import bicgstab

################ ---------------- CONFIGURATION --------- #######################

# Grid parameters
nx_cells = 100
ny_cells = 100

# Problem parameters
Re = 100.0

# Under-relaxation factors
alpha_u = 0.8
alpha_v = 0.8
alpha_p = 0.4

# Iterative solve parameters
momentum_gs_iter_max = 12
bicgstab_iter_max = 1000
bicgstab_tol = 1e-8
SIMPLE_iter_max = 10000
SIMPLE_tol = 1e-8

############### ---------------- END OF CONFIGURATION --------- #######################

print("[STAT] Starting SIMPLE solver")
print(f"[STAT] Grid size: {nx_cells} x {ny_cells}")
print(f"[STAT] Reynolds number: {Re}")

Lx = 1.0
Ly = 1.0
nx = nx_cells - 1
ny = ny_cells - 1
dx = Lx / nx
dy = Ly / ny

# Initialize staggered grid fields
u = np.zeros((ny+2, nx+1))
v = np.zeros((ny+1, nx+2))
p = np.ones((ny+2, nx+2))

# Initialize guess grid fields
v_star = np.zeros_like(v)
u_star = np.zeros_like(u)
pc = np.zeros_like(p)

print("[STAT] Initializing solution fields")

# Initialize coefficient arrays

# Cell center coefficients
aPu = np.ones_like(u)
aPv = np.ones_like(v)
aPp = np.zeros_like(p)

# Neighbor coefficients
aw_ = np.zeros_like(p)
ae_ = np.zeros_like(p)
an_ = np.zeros_like(p)
as_ = np.zeros_like(p)

# Balance coefficients
b  = np.zeros_like(p)

print("[STAT] Initializing coefficient arrays")

# Set boundary conditions
u[-1, :] = 1.0              # Lid velocity (north wall)
u_star[-1, :] = 1.0

print("[STAT] Setting boundary conditions")

SIMPLE_conv_error = np.inf
SIMPLE_iter_count = 0

while SIMPLE_conv_error > SIMPLE_tol and SIMPLE_iter_count < SIMPLE_iter_max:
    SIMPLE_iter_count += 1

    # X-momentum predictor
    for i in range(1, ny+1):
        for j in range(1, nx):
            # Convection terms
            Fw = 0.5 * dy * (u[i, j] + u[i, j-1])
            Fe = 0.5 * dy * (u[i, j+1] + u[i, j])
            Fs = 0.5 * dx * (v[i, j] + v[i, j+1])
            Fn = 0.5 * dx * (v[i-1, j] + v[i-1, j+1])

            # Diffusion terms
            Dw = dy / (dx * Re)
            De = dy / (dx * Re)
            Ds = dx / (dy * Re)
            Dn = dx / (dy * Re)

            # Peclet numbers
            Pe_w = Fw / Dw 
            Pe_e = Fe / De 
            Pe_s = Fs / Ds
            Pe_n = Fn / Dn 

            # Patankar power-law weighting
            weight_w = max(0.0, (1.0 - 0.1*abs(Pe_w))**5)
            weight_e = max(0.0, (1.0 - 0.1*abs(Pe_e))**5)
            weight_n = max(0.0, (1.0 - 0.1*abs(Pe_n))**5)
            weight_s = max(0.0, (1.0 - 0.1*abs(Pe_s))**5)

            # Face coefficients
            ae_[i,j] =  De*weight_e + max(0.0, -Fe)
            aw_[i,j] =  Dw*weight_w + max(0.0,  Fw)
            an_[i,j] =  Dn*weight_n + max(0.0, -Fn)
            as_[i,j] =  Ds*weight_s + max(0.0,  Fs)

            aPu[i, j] = (ae_[i, j] + aw_[i, j] + an_[i, j] + as_[i, j] + (Fe - Fw) + (Fn - Fs)) / alpha_u

    # Under-relaxed Gauss-Seidel for u_star
    for _ in range(momentum_gs_iter_max):
        for i in range(1, ny+1):
            for j in range(1, nx):
                u_star[i, j] = ((1 - alpha_u) * u[i, j] +
                                (1.0 / aPu[i, j]) *
                                (ae_[i, j] * u_star[i, j+1] +
                                 aw_[i, j] * u_star[i, j-1] +
                                 an_[i, j] * u_star[i-1, j] +
                                 as_[i, j] * u_star[i+1, j] +
                                 dy * (p[i, j+1] - p[i, j])))
                

    # Y-momentum predictor
    for i in range(1, ny):
        for j in range(1, nx+1):
            # Convection terms
            Fw = 0.5 * dx * (u[i, j-1] + u[i+1, j-1])
            Fe = 0.5 * dx * (u[i, j]   + u[i+1, j])
            Fs = 0.5 * dx * (v[i, j]   + v[i+1, j])
            Fn = 0.5 * dx * (v[i-1, j] + v[i, j])

            # Diffusion terms
            Dw = dx / (dy * Re)
            De = dx / (dy * Re)
            Ds = dy / (dx * Re)
            Dn = dy / (dx * Re)

            # Peclet numbers
            Pe_w = Fw / Dw
            Pe_e = Fe / De
            Pe_s = Fs / Ds
            Pe_n = Fn / Dn

            # Patankar power-law weighting
            weight_w = max(0.0, (1.0 - 0.1*abs(Pe_w))**5)
            weight_e = max(0.0, (1.0 - 0.1*abs(Pe_e))**5)
            weight_n = max(0.0, (1.0 - 0.1*abs(Pe_n))**5)
            weight_s = max(0.0, (1.0 - 0.1*abs(Pe_s))**5)

            aw_[i, j] = Dw * weight_w + max(0.0,  Fw)
            ae_[i, j] = De * weight_e + max(0.0, -Fe)
            an_[i, j] = Dn * weight_n + max(0.0, -Fn)
            as_[i, j] = Ds * weight_s + max(0.0,  Fs)
            
            aPv[i, j] = (ae_[i, j] + aw_[i, j] + an_[i, j] + as_[i, j] + (Fe - Fw) + (Fn - Fs)) / alpha_v

    # Under-relaxed Gauss-Seidel for v_star
    for _ in range(momentum_gs_iter_max):
        for i in range(1, ny):
            for j in range(1, nx+1):
                v_star[i, j] = ((1 - alpha_v) * v[i, j] +
                                (1.0 / aPv[i, j]) *
                                (ae_[i, j] * v_star[i, j+1] +
                                 aw_[i, j] * v_star[i, j-1] +
                                 an_[i, j] * v_star[i-1, j] +
                                 as_[i, j] * v_star[i+1, j] +
                                 dx * (p[i, j] - p[i+1, j])))

    # Continuity residual
    error = 0.0
    for i in range(1, ny+1):
        for j in range(1, nx+1):
            b[i, j] = (u_star[i, j] - u_star[i, j-1]) * dy + (v_star[i-1, j] - v_star[i, j]) * dx
            error += b[i, j]**2
    SIMPLE_conv_error = np.sqrt(error)
    print(f"Iteration {SIMPLE_iter_count}: continuity error = {SIMPLE_conv_error:.3e}")           
    
    # Pressure correction equation
    for i in range(1, ny+1):
        for j in range(1, nx+1):
            ae_[i, j] = (dx * dy) / aPu[i, j]
            aw_[i, j] = (dx * dy) / aPu[i, j-1]
            an_[i, j] = (dy * dx) / aPv[i-1, j]
            as_[i, j] = (dy * dx) / aPv[i, j]
            aPp[i, j] = ae_[i, j] + aw_[i, j] + an_[i, j] + as_[i, j]

    # Assemble A matrix and b vector for pressure correction
    Np = nx * ny
    A = lil_matrix((Np, Np))
    b_vec = np.zeros(Np)
    for i in range(1, ny+1):
        for j in range(1, nx+1):
            idx = (i-1) * nx + (j-1)
            A[idx, idx] = aPp[i, j]
            if j < nx: A[idx, idx+1] = -ae_[i, j]
            if j > 1:  A[idx, idx-1] = -aw_[i, j]
            if i > 1:  A[idx, idx-nx] = -an_[i, j]
            if i < ny: A[idx, idx+nx] = -as_[i, j]
            b_vec[idx] = b[i, j]
    A = A.tocsr()

    # Solve pressure correction equation
    pc_internal, info, _, _ = bicgstab(A, b_vec, tol=bicgstab_tol, max_iter=bicgstab_iter_max)
    if info != 0:
        print("[WARNING] bicgstab did not converge. Retrying with tighter tol…")
        pc_internal, info, _, _ = bicgstab(A, b_vec, tol=bicgstab_tol*10, max_iter=5*bicgstab_iter_max)
        if info != 0:
            print("[WARNING] retry also failed. Falling back to direct solver…")
            pc_internal = spsolve(A, b_vec)

    pc[1:ny+1, 1:nx+1] = pc_internal.reshape((ny, nx))
    
    # Update pressure and velocity fields
    p[1:ny+1, 1:nx+1] += alpha_p * pc[1:ny+1, 1:nx+1]
    u_star[1:ny+1, 1:nx] += (dy / aPu[1:ny+1, 1:nx]) * (pc[1:ny+1, 2:nx+1] - pc[1:ny+1, 1:nx])
    v_star[1:ny,   1:nx+1] += (dx / aPv[1:ny,   1:nx+1]) * (pc[1:ny,   1:nx+1] - pc[2:ny+1, 1:nx+1])
    u[:, :] = u_star
    v[:, :] = v_star    

print("[STAT] Solving process completed")

# Interpolate velocity to cell centers
u_grid = 0.5 * (u[:-1, :] + u[1:, :])
v_grid = -0.5 * (v[:, :-1] + v[:, 1:])
p_grid = 0.25 * (p[:-1, :-1] + p[:-1, 1:] + p[1:, :-1] + p[1:, 1:])

# Generate meshgrid for outputting
x = np.linspace(0, Lx, nx_cells)
y = np.linspace(0, Ly, ny_cells)
X, Y = np.meshgrid(x, y)

np.savez("results.npz", X=X, Y=Y, p=p_grid, u=u_grid, v=v_grid)
print("[STAT] Results saved to results.npz")








