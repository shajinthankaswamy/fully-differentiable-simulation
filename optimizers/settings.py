"""
Optimizer Settings
==================
Every setting of the five-optimizer race.  objective.py, the five optimizers,
main_run_optimizer.py and compare_plots.py all read ``config`` below; edit it
to change a run.  Free of jax, so the driver can read it without loading the
engine.
"""

import os
from dataclasses import dataclass, field

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---- SETTINGS ----


# fmt: off
@dataclass
class Settings:
    """Settings of one race; edit ``config`` below to change a run."""

    # --- Run ---
    core_name:          str   = "pq_40x40"   # core YAML under cores/
    material:           str   = "N87"        # ferrite of the YAML's ungapped data
    target_l_nh:        float = 250.0        # target inductance            [nH]
    mesh_size_mm:       float = 0.1          # working mesh                 [mm]
    optimizers:         tuple = ("adam", "lbfgs", "cmaes", "ga", "bayesian")
    backends:           tuple = ("lineax",)  # linear-solver backend
    n_seeds:            int   = 1            # repeats of each stochastic optimizer
    stochastic:         tuple = ("cmaes", "ga", "bayesian")
    isolate_processes:  bool  = True         # one process per seed
    cores_dir:          str   = os.path.join(_repo_root, "cores")
    output_dir:         str   = os.path.join(_repo_root, "output")

    # --- Design variables: the six geometry dimensions of the core YAML ---
    vary_keys:          tuple = ("leg_inner_diameter", "center_post_diameter", "leg_height",
                                 "window_height", "coil_clearance", "gap_size")
    frozen_keys:        tuple = ()
    log_coord_max:      float = 20.0         # cap on |v| in x = x_datasheet * exp(v)
    start_perturbation: float = 0.10         # seed k >= 1 moves each variable by up to this
    start_seed_offset:  int   = 10_000       # random stream of seed k is offset + k

    # --- Finish line ---
    early_stop:         bool  = True         # reaching the finish line ends the run
    converge_tol_rel:   float = 0.003        # band: 0.3 % of the target
    converge_rounds:    int   = 10           # consecutive rounds on the finish line
    protocol_version:   int   = 12           # bumped when stored runs stop being comparable
    pop_chunk:          int   = 1            # designs per vmapped batch; 1 fits 12 GB at 0.1 mm

    # --- Physical conditions ---
    b_op_t:             float = 0.25         # rated flux density in the reference post  [T]
    b_max_t:            float = 0.30         # limit in every ferrite section              [T]
    crest_factor:       float = 2**0.5       # peak / rms current
    j_a_per_m2:         float = 4.0e6        # conductor current density              [A/m^2]
    k_fill:             float = 0.4          # copper fill of the coil rectangle
    gap_to_window_max:  float = 0.1          # gap at most this fraction of the window height
    constraint_margin:  float = 0.02         # hinges close this far inside each limit
    constraint_names:   tuple = ("b_post", "b_yoke", "b_outer_leg", "copper_fit", "winding",
                                 "clearance", "fringing", "envelope", "domain", "volume")

    # --- Reporting ---
    print_design:       bool  = True         # print each finished design
    history_fields:     tuple = ("iteration", "evals", "solves", "loss", "l_nh", "elapsed_s")

    # --- Linear solver: CG 1e-6, the inductance plateau of tolerance_comparison.py ---
    lineax_opts: dict = field(default_factory=lambda: {
        "solver": "cg", "rtol": 1e-6, "atol": 1e-6, "max_steps": 20000,
        "preconditioner": "multigrid", "mg_sweeps": 3, "mg_min_dofs": 64,
    })

    # --- Derived from the values above ---
    settle_rounds:      int   = field(init=False)  # rounds the volume must stop changing over
    band_weight:        float = field(init=False)  # so the design settles inside the band
    constraint_weight:  float = field(init=False)  # one datasheet volume at the tolerance

    def __post_init__(self):
        self.settle_rounds = 2 * self.converge_rounds
        self.band_weight = 3.0 * 3.0 / (2.0 * self.converge_tol_rel / 3.0)
        self.constraint_weight = 1.0 / (self.constraint_margin + self.converge_tol_rel) ** 2
# fmt: on


config = Settings()
