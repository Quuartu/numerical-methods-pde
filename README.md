# Numerical Methods for Differential Equations — Homework

Three assignments from the course *Numerical Methods for Differential Equations* (M.Sc. Mathematical Engineering, University of Padova, 2025/26), covering time integration of ODEs, finite-difference schemes for PDEs and the finite element method.
Each folder contains the written report and, where applicable, the MATLAB implementation.

**Author:** Davide Quartucci

---

## Contents

| # | Topic | Methods | Code |
|---|-------|---------|------|
| 1 | ODE time integration | RK4 convergence study; derivation of BDF formulas; stiff linear system from the 2D Laplacian (Crank–Nicolson vs. `ode45`, error vs. CPU time); Lotka–Volterra with implicit CN + Newton (analytic Jacobian); **adaptive-step Crank–Nicolson** with error control | MATLAB |
| 2 | Finite differences for PDEs | Helmholtz equation with the 5-point Laplacian; Crank–Nicolson for the heat equation; von Neumann stability analysis; schemes for the advection equation | — (analytical) |
| 3 | Finite element method | P1 elements for the Laplace/Poisson problem with mixed Dirichlet–Neumann conditions; sparse assembly of the stiffness matrix; L² error over 5 refinement levels; PCG with **Jacobi vs. incomplete Cholesky** preconditioning | MATLAB |

---

## Highlights

- **Convergence verified empirically:** log-log error plots compared against the theoretical order of each scheme.
- **Stiff problems:** explicit vs. implicit integrators compared on accuracy *and* computational cost.
- **Adaptive time stepping:** step size controlled by a local error estimate (tolerance 10⁻⁸).
- **FEM from scratch:** element-by-element assembly into a sparse global matrix, boundary-condition enforcement and preconditioned iterative solvers.

---

## Repository structure

```
HW1_RungeKutta/
├── report.pdf
└── matlab_script/        # RK4, ODE systems, Lotka–Volterra, adaptive CN
HW2_FiniteDifferences/
└── report.pdf
HW3_FEM/
├── report.pdf
└── matlab_script/
    ├── main.m                          # driver: loops over mesh levels 0–4
    ├── assemble_system.m               # stiffness matrix assembly
    ├── apply_boundary_conditions.m     # Dirichlet / Neumann enforcement
    └── Inputfiles/                     # meshes (coord, topol, bound, trace) + reference solution
```

## How to run

Open MATLAB in the relevant `matlab_script/` folder and run the scripts. For HW3, run `main.m` and unzip `Inputfiles/solRef.dat.zip` first.

---

## Relevance to quantitative finance

The same toolkit underlies PDE-based derivative pricing: the Black–Scholes equation is a parabolic PDE solved with Crank–Nicolson finite-difference schemes, and stability, convergence and the efficiency of linear solvers determine the accuracy and speed of a pricer.
