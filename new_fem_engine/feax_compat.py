import feax


def iterative_options_cls():
    """Return feax's iterative-solver options class under either of its names."""
    return getattr(feax, "IterativeSolverOptions", None) or feax.KrylovSolverOptions


def make_iterative_options(
    solver="bicgstab", tol=1e-10, atol=1e-10, maxiter=10000, use_jacobi_preconditioner=True
):
    """Build a feax iterative-solver options object."""
    return iterative_options_cls()(
        solver=solver,
        tol=tol,
        atol=atol,
        maxiter=maxiter,
        use_jacobi_preconditioner=use_jacobi_preconditioner,
    )


def is_iterative(solver_options):
    """Return True if ``solver_options`` is a feax iterative-solver options object."""
    return isinstance(solver_options, iterative_options_cls())
