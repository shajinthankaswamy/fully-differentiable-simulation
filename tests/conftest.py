import gc
import sys
from pathlib import Path

import jax
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True, scope="module")
def _release_compiled_programs():
    """Drop every compiled JAX program once a module is done.

    The 0.1 mm modules each hold several GB of executables and buffers;
    kept across the whole session they peaked at 9.9 GB on an 11 GB machine.
    """
    yield
    jax.clear_caches()
    gc.collect()
