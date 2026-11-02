"""Package logger."""

import logging


def setup_logger(name):
    """Return a logger with one stream handler and JAX's own logging silenced."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logging.getLogger("jax").setLevel(logging.ERROR)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(name)s: %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    # One handler and no propagation, or every message prints twice.
    if not logger.handlers:
        logger.addHandler(handler)
        logger.propagate = False

    return logger
