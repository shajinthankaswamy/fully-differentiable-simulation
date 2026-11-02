"""Sparse assembly, linear solves, Newton iteration and the adjoint.

Every linear solve runs on a JAX BCOO tangent: get_a_bcoo keeps only the
flat per-cell Jacobian entries traced, with the sparsity pattern and the
Dirichlet-DOF set baked in as mesh constants, so assembly and solve run
inside jax.jit.  The Newton machinery is derived from jax-fem.
"""

import dataclasses
import time

import equinox as eqx
import feax
import jax
import jax.flatten_util
import jax.numpy as np
import lineax
import numpy as onp
from jax import config
from jax.experimental.sparse import BCOO

from . import logger, multigrid
from .feax_compat import is_iterative, make_iterative_options

config.update("jax_enable_x64", True)


# ---- LOGGING ----


def _timing_record(timing, name, dt):
    timing[name] += dt


def _log_newton_iter_start(iter_num):
    logger.info("  iter %d", iter_num)


def _log_newton_iter_summary(iter_num, local_s, global_s, res_val, rel_res_val, linear_s=None):
    logger.info(
        "           nonlinear residual: L2 norm = %.3g (relative to initial = %.3g)",
        res_val,
        rel_res_val,
    )
    if linear_s is None:
        logger.info(
            "           timing: local assembly %6.3f s, global matrix %6.3f s", local_s, global_s
        )
    else:
        logger.info(
            "           timing: linear solve %6.3f s, local assembly %6.3f s, global matrix %6.3f s",
            linear_s,
            local_s,
            global_s,
        )


def _log_timing_table(n_iters, parts, wall_s):
    rows = (
        ("local_assembly", "local"),
        ("global_matrix", "global"),
        ("linear", "linear"),
    )

    logger.info("Timing summary — %d Newton iter, %.3f s wall", n_iters, wall_s)
    for key, label in rows:
        dt = parts[key]
        pct = 100.0 * dt / wall_s if wall_s > 0 else 0.0
        logger.info("  %-8s %7.3f s  %5.1f%%", label, dt, pct)
    other = wall_s - sum(parts.values())
    if other >= 0.01:
        pct = 100.0 * other / wall_s if wall_s > 0 else 0.0
        logger.info("  %-8s %7.3f s  %5.1f%%", "other", other, pct)


# ---- SPARSE ASSEMBLY ----


class _SparseMatvec(eqx.Module):
    """Matvec with a BCOO matrix as an equinox module.

    lineax stores a closure_convert=False callable as a jit static hashed by
    identity, so a bare closure would retrace the solve on every call.
    """

    bcoo: BCOO

    def __call__(self, v):
        return self.bcoo @ v


def _bcoo_structure(problem):
    """Return the static assembly structure, cached on the problem.

    Q1 assembly emits one entry per (cell, node pair), so an interior
    coupling arrives from up to four cells; ``dup`` maps the raw entries onto
    the deduplicated (row, col) pairs so the matrix carries one entry per
    nonzero.  ``keep`` masks out the rows and columns of constrained DOFs.
    """
    struct = getattr(problem, "_bcoo_struct", None)
    if struct is not None:
        return struct
    n = problem.num_total_dofs_all_vars
    rows = onp.asarray(problem.row_inds)
    cols = onp.asarray(problem.col_inds)
    bc = _bc_dofs(problem)
    keep = ~(onp.isin(rows, bc) | onp.isin(cols, bc))
    row_f = onp.concatenate([rows[keep], bc])
    col_f = onp.concatenate([cols[keep], bc])
    lin = row_f.astype(onp.int64) * n + col_f.astype(onp.int64)
    uniq, dup = onp.unique(lin, return_inverse=True)
    struct = (
        n,
        row_f,
        col_f,
        keep,
        bc.size,
        (uniq // n).astype(onp.int32),
        (uniq % n).astype(onp.int32),
        dup.reshape(-1).astype(onp.int32),
        int(uniq.size),
    )
    problem._bcoo_struct = struct
    return struct


def get_a_bcoo(problem, jac_values=None):
    """Assemble the global tangent as a BCOO matrix with Dirichlet rows and
    columns replaced by the identity.

    Only ``jac_values`` (default ``problem.jac_values``) is traced, so this
    runs inside jax.jit.  Duplicate entries are summed here, not in the matvec.
    """
    n, row_f, col_f, keep, n_bc, row_u, col_u, dup, n_u = _bcoo_structure(problem)
    if jac_values is None:
        jac_values = problem.jac_values
    values_keep = jac_values[keep]
    raw = np.concatenate([values_keep, np.ones(n_bc, dtype=jac_values.dtype)])
    data = np.zeros(n_u, dtype=raw.dtype).at[np.asarray(dup)].add(raw)
    indices = np.stack([np.asarray(row_u), np.asarray(col_u)], axis=1)
    return BCOO((data, indices), shape=(n, n), indices_sorted=True, unique_indices=True)


# ---- MATVEC AND PRECONDITIONERS ----


class _StencilMatvec(eqx.Module):
    """Matvec with the nine-point stencil of a grid matrix; see _SparseMatvec.

    The same product as the BCOO matvec, several times faster on the CPU.
    """

    stencil: jax.Array
    shape: tuple = eqx.field(static=True)

    def __call__(self, v):
        return multigrid.stencil_matvec(self.stencil, v.reshape(self.shape)).reshape(-1)


def _matvec(a_bcoo, options):
    """Return the matvec of ``a_bcoo``: the grid stencil when ``grid_shape`` is set, else BCOO."""
    shape = options.get("grid_shape")
    if shape is None:
        return _SparseMatvec(a_bcoo)
    shape = tuple(shape)
    return _StencilMatvec(multigrid.stencil_from_bcoo(a_bcoo, shape), shape)


def _bcoo_diag(a_bcoo):
    row, col = a_bcoo.indices[:, 0], a_bcoo.indices[:, 1]
    diag_entries = np.where(row == col, a_bcoo.data, np.zeros_like(a_bcoo.data))
    d = np.zeros(a_bcoo.shape[0], dtype=a_bcoo.dtype)
    return d.at[row].add(diag_entries)


class _DiagPrecond(eqx.Module):
    """Jacobi preconditioner as an equinox module; see _SparseMatvec."""

    dinv: jax.Array

    def __call__(self, r):
        return self.dinv * r


def _make_preconditioner(a_bcoo, options):
    """Return a callable approximating A^-1 r, or None, from ``options``.

    ``preconditioner`` is 'jacobi' (default), 'multigrid' or 'none';
    multigrid needs ``grid_shape`` = (n_r + 1, n_z + 1) to coarsen the grid.
    """
    kind = options.get("preconditioner", "jacobi")
    if kind in (None, "none"):
        return None
    if kind == "jacobi":
        return _DiagPrecond(1.0 / _bcoo_diag(a_bcoo))
    if kind == "multigrid":
        shape = options.get("grid_shape")
        if shape is None:
            raise ValueError(
                "preconditioner='multigrid' needs options['grid_shape'] = "
                "(n_r+1, n_z+1); the V-cycle coarsens the structured grid."
            )
        precond, _ = multigrid.make_preconditioner(
            a_bcoo,
            tuple(shape),
            sweeps=options.get("mg_sweeps", 3),
            min_dofs=options.get("mg_min_dofs", 64),
        )
        return precond
    raise ValueError(f"Unknown preconditioner {kind!r}: use 'jacobi', 'multigrid' or 'none'.")


# ---- LINEAR SOLVERS ----


def bcoo_solve_lineax(a_bcoo, b, x0, linear_options, precond=None):
    """Solve A x = b with lineax on the BCOO matrix.

    Options: ``solver`` (cg | bicgstab | lu | cholesky | auto), ``rtol``,
    ``atol``, ``max_steps``, ``throw`` and the preconditioner keys of
    _make_preconditioner.  The tolerance is tested on the true residual.
    """
    solver_name = linear_options.get("solver", "auto")
    rtol = linear_options.get("rtol", 1e-10)
    atol = linear_options.get("atol", 1e-10)
    max_steps = linear_options.get("max_steps", 10000)
    linear_options = dict(linear_options)
    linear_options.setdefault(
        "preconditioner", "jacobi" if linear_options.get("jacobi", True) else "none"
    )
    # throw=False lets a fixed-iteration run (rtol = atol = 0) return its iterate.
    throw = linear_options.get("throw", True)

    if solver_name == "cg":
        solver = lineax.CG(rtol=rtol, atol=atol, max_steps=max_steps)
    elif solver_name == "bicgstab":
        solver = lineax.BiCGStab(rtol=rtol, atol=atol, max_steps=max_steps)
    elif solver_name == "lu":
        solver = lineax.LU()
    elif solver_name == "cholesky":
        solver = lineax.Cholesky()
    else:
        solver = lineax.AutoLinearSolver(well_posed=True)

    x0 = np.zeros_like(b) if x0 is None else x0

    # The tangent is SPD after the Dirichlet elimination; CG and Cholesky
    # need the operator tagged as such.
    if solver_name in ("cg", "cholesky"):
        tags = frozenset({lineax.symmetric_tag, lineax.positive_semidefinite_tag})
    else:
        tags = frozenset()

    if solver_name in ("lu", "cholesky", "auto"):
        # lineax has no sparse direct solver, so these densify.
        operator = lineax.MatrixLinearOperator(a_bcoo.todense(), tags=tags)
        options = {}
    else:
        operator = lineax.FunctionLinearOperator(
            _matvec(a_bcoo, linear_options),
            jax.ShapeDtypeStruct(b.shape, b.dtype),
            tags=tags,
            closure_convert=False,
        )
        options = {}
        if precond is None:
            precond = _make_preconditioner(a_bcoo, linear_options)
        if precond is not None:
            # A V-cycle with equal pre- and post-smoothing is symmetric.
            spd = frozenset({lineax.symmetric_tag, lineax.positive_semidefinite_tag})
            options["preconditioner"] = lineax.TaggedLinearOperator(
                lineax.FunctionLinearOperator(
                    precond,
                    jax.ShapeDtypeStruct(b.shape, b.dtype),
                    tags=spd,
                    closure_convert=False,
                ),
                spd,
            )

    # lineax takes no initial guess, so x0 is folded in as A (x0 + d) = b.
    solution = lineax.linear_solve(
        operator, b - operator.mv(x0), solver=solver, options=options, throw=throw
    )
    return x0 + solution.value


def bcoo_solve_feax(a_bcoo, b, x0, linear_options, precond=None):
    """Solve A x = b with a feax iterative solver on the BCOO matrix.

    ``options`` is a feax iterative-options object (default CG); direct feax
    solvers need a scipy CSR matrix and are refused.  The preconditioner
    keys of _make_preconditioner apply.
    """
    x0 = np.zeros_like(b) if x0 is None else x0

    solver_options = linear_options.get(
        "options",
        make_iterative_options(
            solver="cg", tol=1e-10, atol=1e-10, maxiter=10000, use_jacobi_preconditioner=True
        ),
    )

    if not is_iterative(solver_options):
        raise NotImplementedError(
            f"feax {type(solver_options).__name__} needs a scipy CSR matrix, "
            "which this engine does not build. Use an iterative options "
            "instance, e.g. feax_compat.make_iterative_options(solver='cg')."
        )
    if solver_options.solver == "auto":
        solver_options = feax.resolve_iterative_solver(solver_options, feax.MatrixProperty.SPD)

    if precond is None:
        precond = _make_preconditioner(a_bcoo, linear_options)
    solver_options = dataclasses.replace(
        solver_options, preconditioner=precond, use_jacobi_preconditioner=False
    )

    return feax.create_linear_solve_fn(solver_options)(_matvec(a_bcoo, linear_options), b, x0)


_linear_solver_keys = frozenset({"lineax_solver", "feax_solver", "custom_solver"})


def _backend_options(linear_options):
    """Return the lineax or feax options inside ``linear_options``, or None for a custom solver."""
    for key in ("lineax_solver", "feax_solver"):
        if key in linear_options:
            return linear_options[key]
    if "custom_solver" in linear_options:
        return None
    return {"solver": "cg"}


def linear_solver(a_bcoo, b, x0, linear_options, precond=None):
    """Dispatch A x = b to the backend named in ``linear_options``.

    One of ``lineax_solver``, ``feax_solver`` or ``custom_solver``
    (a callable ``(A, b, x0, options)``); defaults to lineax CG.  A given
    ``precond`` replaces the one lineax or feax would build from ``a_bcoo``.
    """
    options = dict(linear_options)

    if len(options.keys() & _linear_solver_keys) == 0:
        options["lineax_solver"] = {"solver": "cg"}

    if "lineax_solver" in options:
        return bcoo_solve_lineax(a_bcoo, b, x0, options["lineax_solver"], precond)
    if "feax_solver" in options:
        return bcoo_solve_feax(a_bcoo, b, x0, options["feax_solver"], precond)
    if "custom_solver" in options:
        return options["custom_solver"](a_bcoo, b, x0, options)
    raise NotImplementedError("Unknown linear solver.")


# ---- BOUNDARY CONDITIONS ----


def apply_bc_vec(res_vec, dofs, problem, scale=1.0):
    """Replace the residual on Dirichlet DOFs with (dof - prescribed value) * scale."""
    res_list = problem.unflatten_fn_sol_list(res_vec)
    sol_list = problem.unflatten_fn_sol_list(dofs)

    for ind, fe in enumerate(problem.fes):
        res = res_list[ind]
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            res = res.at[fe.node_inds_list[i], fe.vec_inds_list[i]].set(
                sol[fe.node_inds_list[i], fe.vec_inds_list[i]], unique_indices=True
            )
            res = res.at[fe.node_inds_list[i], fe.vec_inds_list[i]].add(-fe.vals_list[i] * scale)

        res_list[ind] = res

    return jax.flatten_util.ravel_pytree(res_list)[0]


def apply_bc(res_fn, problem, scale=1.0):
    """Wrap a residual function so apply_bc_vec is applied to its output."""

    def res_fn_bc(dofs):
        res_vec = res_fn(dofs)
        return apply_bc_vec(res_vec, dofs, problem, scale)

    return res_fn_bc


def assign_bc(dofs, problem):
    """Return ``dofs`` with the Dirichlet DOFs set to their prescribed values."""
    sol_list = problem.unflatten_fn_sol_list(dofs)
    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            sol = sol.at[fe.node_inds_list[i], fe.vec_inds_list[i]].set(fe.vals_list[i])
        sol_list[ind] = sol
    return jax.flatten_util.ravel_pytree(sol_list)[0]


def copy_bc(dofs, problem):
    """Return a zero vector carrying only the Dirichlet DOFs of ``dofs``."""
    new_dofs = np.zeros_like(dofs)
    sol_list = problem.unflatten_fn_sol_list(dofs)
    new_sol_list = problem.unflatten_fn_sol_list(new_dofs)

    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        new_sol = new_sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            new_sol = new_sol.at[fe.node_inds_list[i], fe.vec_inds_list[i]].set(
                sol[fe.node_inds_list[i], fe.vec_inds_list[i]]
            )
        new_sol_list[ind] = new_sol

    return jax.flatten_util.ravel_pytree(new_sol_list)[0]


def get_flatten_fn(fn_sol_list, problem):
    """Wrap a function of solution lists as a function of the flat DOF vector."""

    def fn_dofs(dofs):
        sol_list = problem.unflatten_fn_sol_list(dofs)
        val_list = fn_sol_list(sol_list)
        return jax.flatten_util.ravel_pytree(val_list)[0]

    return fn_dofs


def _bc_dofs(problem):
    return onp.unique(
        onp.concatenate(
            [
                onp.asarray(fe.node_inds_list[i]) * fe.vec + onp.asarray(fe.vec_inds_list[i])
                for fe in problem.fes
                for i in range(len(fe.node_inds_list))
            ]
        )
    )


# ---- NEWTON ----


def _jitted_linear_solver(linear_options):
    """Return the linear solve compiled once per options dict, cached on the dict."""
    fn = linear_options.get("_jitted_solve")
    if fn is None:
        fn = jax.jit(lambda a_bcoo, b, x0: linear_solver(a_bcoo, b, x0, linear_options))
        linear_options["_jitted_solve"] = fn
    return fn


def newton_step(problem, res_vec, a_bcoo, dofs, newton_cfg, timing):
    """Take one Newton step; return (dofs, linear solve seconds)."""
    logger.debug("Solving linear system...")
    b = -res_vec

    x0_1 = assign_bc(np.zeros(problem.num_total_dofs_all_vars), problem)
    x0_2 = copy_bc(dofs, problem)
    x0 = x0_1 - x0_2

    t0 = time.perf_counter()
    inc = _jitted_linear_solver(newton_cfg.get("linear", {}))(a_bcoo, b, x0)
    # Dispatch is asynchronous; without blocking the clock stops at enqueue.
    inc = jax.block_until_ready(inc)
    linear_s = time.perf_counter() - t0
    _timing_record(timing, "linear", linear_s)

    if newton_cfg.get("line_search_flag", False):
        dofs = line_search(problem, dofs, inc)
    else:
        dofs = dofs + inc

    return dofs, linear_s


def line_search(problem, dofs, inc):
    """Backtracking line search on the residual norm along ``inc``."""
    res_fn = problem.compute_residual
    res_fn = get_flatten_fn(res_fn, problem)
    res_fn = apply_bc(res_fn, problem)

    def res_norm_fn(alpha):
        res_vec = res_fn(dofs + alpha * inc)
        return np.linalg.norm(res_vec)

    alpha = 1.0
    res_norm = res_norm_fn(alpha)
    for i in range(3):
        alpha *= 0.5
        res_norm_half = res_norm_fn(alpha)
        logger.debug(f"i = {i}, res_norm = {res_norm}, res_norm_half = {res_norm_half}")
        if res_norm_half > res_norm:
            alpha *= 2.0
            break
        res_norm = res_norm_half

    return dofs + alpha * inc


# ---- SOLVER ENTRY POINTS ----


# solver_options is {'newton': {...}} or a flat dict auto-wrapped as Newton:
#   {'newton': {'tol': 1e-6, 'rel_tol': 1e-8, 'line_search_flag': False,
#               'initial_guess': sol_list,
#               'linear': {'lineax_solver': {'solver': 'cg'}}}}
_method_keys = frozenset({"newton"})
_linear_option_keys = frozenset({"lineax_solver", "feax_solver", "custom_solver"})
_newton_option_keys = frozenset({"tol", "rel_tol", "line_search_flag", "initial_guess"})


def _resolve_solver_options(solver_options):
    opts = solver_options or {}
    methods = [m for m in _method_keys if m in opts]

    if not methods:
        linear = {k: opts[k] for k in _linear_option_keys if k in opts}
        cfg = {k: opts[k] for k in _newton_option_keys if k in opts}
        if linear:
            cfg["linear"] = linear
        return "newton", cfg

    if len(methods) > 1:
        raise ValueError(f"Pick one nonlinear method, got {methods}.")

    method = methods[0]
    if not isinstance(opts[method], dict):
        raise ValueError(f"solver_options['{method}'] must be a dict.")
    return method, opts[method]


def _with_traceable(solver_options):
    import copy

    opts = copy.deepcopy(solver_options) if solver_options else {}
    method, _ = _resolve_solver_options(opts)
    opts.setdefault(method, {})["traceable"] = True
    return opts


def solver(problem, solver_options={}, return_timing=False):
    """Solve the problem by Newton iteration.

    A ``traceable`` config takes the jit-safe path of solver_jax.  A problem
    with ``linear_problem = True`` takes exactly one Newton step.

    Returns:
        The solution list, or (solution list, timing dict) with
        ``return_timing``; the timing keys are local_assembly, global_matrix
        and linear.
    """
    method, cfg = _resolve_solver_options(solver_options)
    assert method == "newton"

    if cfg.get("traceable"):
        sol_list = solver_jax(problem, cfg)
        if return_timing:
            return sol_list, {"local_assembly": 0.0, "global_matrix": 0.0, "linear": 0.0}
        return sol_list

    logger.info("Solving the nonlinear problem...")
    timing = {"local_assembly": 0.0, "global_matrix": 0.0, "linear": 0.0}
    wall_start = time.perf_counter()

    if "initial_guess" in cfg:
        initial_guess = jax.lax.stop_gradient(cfg["initial_guess"])
        dofs = jax.flatten_util.ravel_pytree(initial_guess)[0]
    else:
        dofs = np.zeros(problem.num_total_dofs_all_vars)

    rel_tol = cfg.get("rel_tol", 1e-8)
    tol = cfg.get("tol", 1e-6)

    def newton_update_helper(dofs):
        sol_list = problem.unflatten_fn_sol_list(dofs)
        t0 = time.perf_counter()
        res_list, jac_values = problem.newton_update(sol_list)
        jac_values = jax.block_until_ready(jac_values)
        local_s = time.perf_counter() - t0
        _timing_record(timing, "local_assembly", local_s)
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        res_vec = apply_bc_vec(res_vec, dofs, problem)

        t0 = time.perf_counter()
        a_bcoo = get_a_bcoo(problem, jac_values)
        jax.block_until_ready(a_bcoo.data)
        global_s = time.perf_counter() - t0
        _timing_record(timing, "global_matrix", global_s)
        return res_vec, a_bcoo, local_s, global_s

    _log_newton_iter_start(0)
    res_vec, a_bcoo, local_s, global_s = newton_update_helper(dofs)
    if getattr(problem, "linear_problem", False):
        dofs, linear_s = newton_step(problem, res_vec, a_bcoo, dofs, cfg, timing)
        _log_newton_iter_summary(1, 0.0, 0.0, 0.0, 0.0, linear_s)
        n_iters = 1
        res_val = 0.0
    else:
        res_val = np.linalg.norm(res_vec)
        res_val_initial = res_val
        rel_res_val = res_val / res_val_initial
        _log_newton_iter_summary(0, local_s, global_s, res_val, rel_res_val)
        n_iters = 0
        while (rel_res_val > rel_tol) and (res_val > tol):
            n_iters += 1
            _log_newton_iter_start(n_iters)
            dofs, linear_s = newton_step(problem, res_vec, a_bcoo, dofs, cfg, timing)
            res_vec, a_bcoo, local_s, global_s = newton_update_helper(dofs)
            res_val = np.linalg.norm(res_vec)
            rel_res_val = res_val / res_val_initial
            _log_newton_iter_summary(n_iters, local_s, global_s, res_val, rel_res_val, linear_s)

    assert np.all(np.isfinite(res_val)), "res_val contains NaN, stop the program!"
    assert np.all(np.isfinite(dofs)), "dofs contains NaN, stop the program!"

    sol_list = problem.unflatten_fn_sol_list(dofs)

    _log_timing_table(n_iters, timing, time.perf_counter() - wall_start)

    logger.info(f"max of dofs = {np.max(dofs)}")
    logger.info(f"min of dofs = {np.min(dofs)}")

    if return_timing:
        return sol_list, timing
    return sol_list


def solver_jax(problem, cfg, return_precond=False):
    """One linear Newton step from zero with no Python control flow, safe under jax.jit.

    With ``return_precond`` also returns the preconditioner built for the
    solve, so the adjoint can reuse it.
    """
    dofs = np.zeros(problem.num_total_dofs_all_vars)
    sol_list = problem.unflatten_fn_sol_list(dofs)
    res_list, jac_values = problem.newton_update(sol_list)
    res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
    res_vec = apply_bc_vec(res_vec, dofs, problem)

    a_bcoo = get_a_bcoo(problem, jac_values)
    b = -res_vec
    linear = cfg.get("linear", {})
    backend = _backend_options(linear)
    precond = None if backend is None else _make_preconditioner(a_bcoo, backend)
    inc = linear_solver(a_bcoo, b, np.zeros_like(b), linear, precond)
    sol_list = problem.unflatten_fn_sol_list(dofs + inc)
    return (sol_list, precond) if return_precond else sol_list


# ---- ADJOINT ----


def _constraint_fn(problem, dofs, params):
    problem.set_params(params)
    res_fn = problem.compute_residual
    res_fn = get_flatten_fn(res_fn, problem)
    res_fn = apply_bc(res_fn, problem)
    return res_fn(dofs)


def _vjp_contraint_fn_params(problem, params, sol_list):
    def constraint_fn_sol_to_sol(sol_list, params):
        dofs = jax.flatten_util.ravel_pytree(sol_list)[0]
        con_vec = _constraint_fn(problem, dofs, params)
        return problem.unflatten_fn_sol_list(con_vec)

    def partial_params_c_fn(params):
        return constraint_fn_sol_to_sol(sol_list, params)

    def vjp_linear_fn(v_list):
        primals_output, f_vjp = jax.vjp(partial_params_c_fn, params)
        (val,) = f_vjp(v_list)
        return val

    return vjp_linear_fn


def _solve_adjoint(a_bcoo, v_vec, adjoint_solver_options, problem=None, precond=None):
    """Solve A^T z = v with the seed normalised, so absolute tolerances stay meaningful.

    The seed is zeroed on the Dirichlet DOFs, whose values do not depend on
    the parameters.  ``precond`` is the forward solve's; A is symmetric, so it
    serves A^T too.
    """
    v_vec = jax.flatten_util.ravel_pytree(v_vec)[0]
    if problem is not None:
        v_vec = v_vec.at[_bc_dofs(problem)].set(0.0)
    v_norm = np.linalg.norm(v_vec)
    v_norm_safe = np.where(v_norm > 0.0, v_norm, 1.0)
    adjoint_vec = linear_solver(
        a_bcoo.T, v_vec / v_norm_safe, np.zeros_like(v_vec), adjoint_solver_options, precond
    )
    return adjoint_vec * v_norm_safe


def implicit_vjp(problem, sol_list, params, v_list, adjoint_solver_options, precond=None):
    """Return the VJP of the solve with respect to ``params`` by the adjoint method."""
    problem.set_params(params)
    _, jac_values = problem.newton_update(sol_list)

    a_bcoo = get_a_bcoo(problem, jac_values)
    adjoint_vec = _solve_adjoint(a_bcoo, v_list, adjoint_solver_options, problem, precond)

    vjp_linear_fn = _vjp_contraint_fn_params(problem, params, sol_list)
    vjp_result = vjp_linear_fn(problem.unflatten_fn_sol_list(adjoint_vec))
    return jax.tree_util.tree_map(lambda x: -x, vjp_result)


def ad_wrapper(problem, solver_options={}, adjoint_solver_options={}, traceable=False):
    """Return ``params -> sol_list`` with a custom VJP through the adjoint solve.

    With ``traceable`` the forward solve takes the jit-safe path.
    """
    if traceable:
        solver_options = _with_traceable(solver_options)

    @jax.custom_vjp
    def fwd_pred(params):
        problem.set_params(params)
        return solver(problem, solver_options)

    def f_fwd(params):
        if traceable:
            problem.set_params(params)
            _, cfg = _resolve_solver_options(solver_options)
            sol_list, precond = solver_jax(problem, cfg, return_precond=True)
        else:
            sol_list, precond = fwd_pred(params), None
        return sol_list, (params, sol_list, precond)

    def f_bwd(res, v):
        logger.info("Running backward and solving the adjoint problem...")
        params, sol_list, precond = res
        vjp_result = implicit_vjp(problem, sol_list, params, v, adjoint_solver_options, precond)
        return (vjp_result,)

    fwd_pred.defvjp(f_fwd, f_bwd)
    return fwd_pred
