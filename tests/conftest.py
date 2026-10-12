"""Test-wide setup.

**Thread limits must be set before numpy is imported**, which is why they are
here at module scope rather than in a fixture: pytest imports the root conftest
before any test module, and OpenBLAS reads these once, when it initialises.

The dev laptop has 7.4 GB of RAM and 12 logical cores. OpenBLAS sizes its
per-thread scratch buffers from the core count, and on a machine already at 98%
commit it cannot get them:

    OpenBLAS error: Memory allocation still failed after 10 retries, giving up.

That kills the interpreter outright - pytest exits non-zero having printed only
dots, with no traceback to explain it. The work here is small-array signal
processing (resampling a few seconds of speech), so one thread is both
sufficient and faster than paying for thread setup. It matters more in M3, where
the eval harness resamples the whole corpus across every condition.
"""

import os

for _var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

# Tests never spend credits, so metering is off unless a test asks for it.
os.environ.setdefault("KHATACALL_USAGE_LOG", "off")
