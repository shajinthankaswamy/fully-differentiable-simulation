# Benchmarking Gradient-Based and Black-Box Optimizers on a Differentiable Physics Simulator

**Question.** A finite-element (FE) solver in an optimization loop is expensive, because every evaluated design costs one FE
simulation. Gradient-based optimizers need the derivative of the output with respect to the geometry, which common FE solvers do
not provide; gradient-free optimizers need only the output, but many more evaluations. For a design problem with only a few
variables, does a gradient-based optimizer on a differentiable FE solver need fewer simulations than gradient-free methods?

**Approach.** An end-to-end differentiable FE solver for magnetostatics is written in JAX, its adjoint gradient is verified, and five
optimizers — two gradient-based, three gradient-free — are compared on the same constrained inverse-design problem under a shared
objective, start design and stopping rule. The test problem is the smallest gapped PQ 40/40 ferrite core (TDK N87) with a one-turn
inductance of 250 nH: 6 geometric design variables, 10 physical constraints.

## Highlights

- **FE solver.** Axisymmetric magnetostatic FE solver in JAX (Q1 elements), solved by conjugate gradients with a geometric multigrid
  V-cycle preconditioner. On the 0.1 mm conforming mesh (206,400 cells, 205,457 unknowns) one solve takes 0.62 s on CPU, with 18–28 CG iterations at every mesh size from 0.5 to 0.05 mm.
- **Adjoint sensitivities.** The adjoint method is implemented as a `jax.custom_vjp`: one forward and one adjoint solve give the
  gradient with respect to all design variables, independent of their number.
- **Inverse design framework.** Inductor design is formulated as a constrained inverse problem (6 design variables, 10 conditions on
  flux density, winding space, clearance, gap and envelope) on the differentiable solver.
- **Gradient verification.** Second-order Taylor test (rate 2.000 down to a round-off plateau of about 5 × 10⁻¹³), independent
  automatic differentiation (agreement to 4.3 × 10⁻¹²) and central finite differences.
- **Gradient-based vs. black-box optimization.** L-BFGS, Adam, CMA-ES, a genetic algorithm and GP-based Bayesian optimization.
  L-BFGS was 6.7× more sample-efficient than the genetic algorithm; CMA-ES found the best design (20 % volume reduction).
- **Testing.** pytest suite of 161 tests: Taylor, second-order, symmetry, mesh-convergence and end-to-end checks.

## Results

Cost is counted in FE simulations until the best feasible design first reaches 0.886 of the datasheet volume — the largest of the
five final volumes, so a task every optimizer completed. All five final designs, re-simulated, lie within 0.29 nH of the target and
meet all ten conditions.

| Optimizer             | Family       | L (nH) | V / V_e   | FE simulations | Time (s) |
| --------------------- | ------------ | -----: | --------: | -------------: | -------: |
| L-BFGS                | gradient     | 249.99 |     0.850 |        **122** |  **384** |
| Bayesian optimization | surrogate    | 249.85 |     0.810 |            145 |      775 |
| Adam                  | gradient     | 249.99 |     0.886 |            236 |      645 |
| CMA-ES                | evolutionary | 250.04 | **0.798** |            516 |     1617 |
| Genetic algorithm     | evolutionary | 249.71 |     0.887 |            822 |     2106 |

- **The gradient buys sample efficiency.** L-BFGS needed the fewest simulations, 1.2–6.7× fewer than the other four.
- **Global search buys a smaller core.** CMA-ES and Bayesian optimization also shrank the post and the gap together, reaching designs
  where both the yoke and the post operate near the 0.30 T limit — about 20 % below the datasheet volume.
- **Bayesian optimization comes close on both.** 19 % more simulations than L-BFGS, and within 1.5 % of the smallest core.

## Figures

### 1. Forward model

The datasheet core (N87, 1.0 mm gap) at 1 A through one turn on the 0.1 mm conforming mesh (206,400 cells).

<p align="center">
  <img src="output/lineax_forward/geometry.png" width="32%">
  <img src="output/lineax_forward/psi.png" width="32%">
  <img src="output/lineax_forward/bfield.png" width="32%">
</p>
<p align="center"><em>Geometry with core, coil and gap (left); flux function ψ, whose level sets are the flux lines (centre); flux-density
magnitude with flux lines (right).</em></p>

### 2. Choice of the solver

<p align="center">
  <img src="output/solver_comparison/mesh_vs_inductance.png" width="45%">
  <img src="output/solver_comparison/tolerance_choice.png" width="52%">
</p>
<p align="center"><em>Mesh study over ten cell sizes from 0.5 to 0.05 mm: the 0.1 mm working mesh lies 0.41 nH (0.15 %) below the converged
inductance (left). CG tolerance sweep: the inductance error reaches its round-off plateau at 10⁻⁶, which is used as the working
tolerance (right).</em></p>

<p align="center">
  <img src="output/solver_inductance_comparison/gap_vs_total_time.png" width="48%">
  <img src="output/solver_inductance_comparison/gap_vs_backward_time.png" width="48%">
</p>
<p align="center"><em>Time of one gradient (forward + adjoint solve, left) and of the adjoint solve alone (right) over ten air gaps, median of
ten calls. Both CG libraries return the same inductance; lineax is 2.1–7.0 % faster at the working tolerance.</em></p>

### 3. Verification of the adjoint gradient

<p align="center">
  <img src="output/gradient_taylor_test/taylor_test.png" width="60%">
</p>
<p align="center"><em>Second-order Taylor test in three random directions. Without the gradient, the remainder r₀ falls as ε; with it, r₁
falls as ε² — which holds only for an exact derivative. Rate 2.000 for both CG libraries.</em></p>

<p align="center">
  <img src="output/gradient_taylor_test/taylor_test_libraries.png" width="48%">
  <img src="output/gradient_taylor_test/taylor_test_tolerance.png" width="48%">
</p>
<p align="center"><em>Taylor test down to the round-off plateau of about 5 × 10⁻¹³ for both libraries (left), and at CG tolerances of 10⁻¹⁰
and the working 10⁻⁶ (right): the working tolerance does not limit the accuracy of the gradient.</em></p>

### 4. Comparison of the optimizers

<p align="center">
  <img src="output/lbfgs/pq_40x40_target250.0nH_N87_mesh0.100mm/lineax/convergence.png" width="32%">
  <img src="output/adam/pq_40x40_target250.0nH_N87_mesh0.100mm/lineax/convergence.png" width="32%">
  <img src="output/bayesian/pq_40x40_target250.0nH_N87_mesh0.100mm/lineax/convergence.png" width="32%">
</p>
<p align="center">
  <img src="output/cmaes/pq_40x40_target250.0nH_N87_mesh0.100mm/lineax/convergence.png" width="32%">
  <img src="output/ga/pq_40x40_target250.0nH_N87_mesh0.100mm/lineax/convergence.png" width="32%">
  <img src="output/comparison/pq_40x40_target250.0nH_N87_mesh0.100mm/optimizer_vs_incumbent.png" width="32%">
</p>
<p align="center"><em>Convergence from the shared start design — L-BFGS, Adam, Bayesian optimization (top); CMA-ES, genetic algorithm, and
the best inductance error reached against the number of FE simulations (bottom). Each run shows the objective (top) and the
inductance of the proposal (bottom). L-BFGS enters the finish band after four iterations; Adam's momentum makes the inductance
oscillate around the target.</em></p>

<p align="center">
  <img src="output/comparison/pq_40x40_target250.0nH_N87_mesh0.100mm/optimizer_vs_simulations.png" width="48%">
  <img src="output/comparison/pq_40x40_target250.0nH_N87_mesh0.100mm/optimizer_vs_time.png" width="48%">
</p>
<p align="center">
  <img src="output/comparison/pq_40x40_target250.0nH_N87_mesh0.100mm/optimizer_vs_volume.png" width="48%">
  <img src="output/comparison/pq_40x40_target250.0nH_N87_mesh0.100mm/optimizer_vs_analytical.png" width="48%">
</p>
<p align="center"><em>FE simulations (top left) and wall time (top right) until the best feasible design first reached V ≤ 0.886 V_e;
ferrite volume of each re-simulated final design (bottom left); FE and reluctance-model inductance of each final design
(bottom right).</em></p>

<p align="center">
  <img src="output/comparison/pq_40x40_target250.0nH_N87_mesh0.100mm/optimizer_vs_parameter_change.png" width="85%">
</p>
<p align="center"><em>Change of the six design variables from the datasheet design: all five optimizers shorten the legs and thin the yoke;
CMA-ES and Bayesian optimization also shrink the post and the gap together.</em></p>

<p align="center">
  <img src="output/lbfgs/pq_40x40_target250.0nH_N87_mesh0.100mm/lineax/bfield.png" width="19%">
  <img src="output/adam/pq_40x40_target250.0nH_N87_mesh0.100mm/lineax/bfield.png" width="19%">
  <img src="output/bayesian/pq_40x40_target250.0nH_N87_mesh0.100mm/lineax/bfield.png" width="19%">
  <img src="output/cmaes/pq_40x40_target250.0nH_N87_mesh0.100mm/lineax/bfield.png" width="19%">
  <img src="output/ga/pq_40x40_target250.0nH_N87_mesh0.100mm/lineax/bfield.png" width="19%">
</p>
<p align="center"><em>The five finished cores with their flux density and flux lines at 1 A through one turn: L-BFGS, Adam, Bayesian
optimization, CMA-ES and genetic algorithm.</em></p>

## Method in short

| Step       | What                                                                                                                                                                                                 | Where                                     |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| Model      | Axisymmetric magnetostatics in the flux function ψ = r·A_φ: −∇·((ν/r)∇ψ) = J_φ. Bilinear (Q1) elements on a fixed 0.1 mm grid (350 × 600 = 210,000 cells, 209,051 unknowns). Cells cut by a material edge are treated as laminates, so the inductance is continuous in the geometry. | `new_fem_engine/problem.py`, `geometry.py` |
| Solver     | CG with a geometric multigrid V-cycle preconditioner (damped Jacobi smoothing, Galerkin coarse operators), compiled with the solver. Two CG libraries: lineax and feax (`jax.scipy` CG).                | `new_fem_engine/solver.py`, `multigrid.py` |
| Gradient   | Adjoint method as a `jax.custom_vjp`: one forward and one adjoint solve give dL/d(all parameters); the chain rule through homogenization and geometry gives dL/dx for the six design variables.        | `new_fem_engine/solver.py`                |
| Objective  | J(x) = V/V_e + w_b (L/L* − 1)² + w_c Σ max(g_k + m, 0)², with 10 conditions g_k (flux density in post, yokes and outer leg ≤ 0.30 T, copper and winding fit, coil clearance, gap ≤ 0.1 window height, envelope, domain, volume). | `optimizers/objective.py`, `settings.py`  |
| Optimizers | Adam, L-BFGS (Optax); CMA-ES, GA (evosax); trust-region Bayesian optimization with expected improvement (GPJax). Same objective, start design and stopping rule, in log coordinates v = ln(x/x_ds).     | `optimizers/*.py`                         |

Solver settings are chosen by measurement: a ten-level mesh study places the 0.1 mm working mesh 0.41 nH (0.15 %) below the
converged inductance; a tolerance sweep fixes CG at 10⁻⁶, where the inductance error reaches round-off; lineax and feax give the same
inductance, and lineax is used as 2.1–7.0 % faster at that tolerance.

## Repository layout

```
├── new_fem_engine/       mesh, Q1 elements, magnetostatic problem, CG + multigrid, adjoint
├── comparison_studies/   solver choice (mesh, tolerance, library) and gradient verification
├── optimizers/           objective, settings, the five optimizers, comparison plots
├── cores/pq_40x40/       test-problem geometry and N87 material data
├── tests/                pytest suite (unit, accuracy and end-to-end checks)
└── output/               results and figures of every study (CSV, JSON, PNG)
```

## How to run

### Installation

Python 3.12. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -m "not slow"             # quick check
```

### Running the studies

| What                                            | Command                                                  |
| ----------------------------------------------- | -------------------------------------------------------- |
| One forward solve of the datasheet core         | `python new_fem_engine/run_forward.py`                    |
| Mesh study                                      | `python comparison_studies/solver_comparison.py`          |
| CG tolerance study                              | `python comparison_studies/tolerance_comparison.py`       |
| CG library timing over the air gap              | `python comparison_studies/solver_inductance_comparison.py` |
| Taylor test of the adjoint gradient             | `python comparison_studies/gradient_taylor_test.py`       |
| Further gradient checks (AD, regions, FD)       | `python comparison_studies/gradient_verification.py`      |
| Five-optimizer comparison                       | `python optimizers/main_run_optimizer.py`                 |
| Redraw comparison figures only                  | `python optimizers/compare_plots.py`                      |
| Tests                                           | `pytest -m "not slow"` (fast) or `pytest` (all)           |

Most study scripts accept `--plot` to redraw their figures from the saved CSV without solving again. Run settings are in
`optimizers/settings.py`.

## Scope and future work

The study covers a linear, axisymmetric magnetostatic model with six design variables, one core and one inductance target; saturation
is enforced through the 0.30 T flux-density conditions. Natural extensions are:

- repeating the stochastic optimizers over several random seeds and inductance targets;
- scaling to more design variables, where the adjoint gradient keeps its cost of two simulations while finite differences and
  population sizes grow;
- a differentiable inverse-design loop in which a global search or a generative model conditioned on the target inductance proposes
  candidate designs, and the differentiable FE model refines each of them with its exact adjoint gradient.

## Author

Shajin Thankaswamy — [shajinthankaswamy.github.io](https://shajinthankaswamy.github.io)

Supervisors: Frank Wendler (Chair of Materials Simulation, FAU Erlangen-Nürnberg) and Georg Kruse (AI-augmented Simulation group,
Fraunhofer IISB).
