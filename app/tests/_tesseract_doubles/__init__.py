"""Served-Tesseract doubles for the app tests.

The pipeline drives every solver through a served
:class:`~tesseract_core.Tesseract` composed by
:func:`tesseract_jax.apply_tesseract`. These helpers serve the physics-free
fixture modules under this package as real in-process Tesseracts, so a test can
exercise the production component builders and JAX composition without the Julia
or FEniCS backend. Each helper returns ``(tesseract, module)``: the served
handle to pass to a builder, and the imported fixture module whose recording
lists (``FORWARD_BIASES``, ``FIELDS``, ...) the test asserts against.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from tesseract_core import Tesseract


def _serve(fixture: str) -> tuple[Tesseract, ModuleType]:
    module = importlib.import_module(f"_tesseract_doubles.{fixture}.tesseract_api")
    # Clear any recording left by a prior serve of the same module: the module
    # object is a process singleton, so one test's calls must not leak into the
    # next.
    for name in (
        "FORWARD_BIASES",
        "VJP_BIASES",
        "FORWARD_MESH_PATHS",
        "FIELDS",
        "APPLY_MODE_INDICES",
        "VJP_MODE_INDICES",
    ):
        recorder = getattr(module, name, None)
        if recorder is not None:
            recorder.clear()
    return Tesseract.from_tesseract_api(module), module


def serve_chargetransport() -> tuple[Tesseract, ModuleType]:
    """Serve the linear ChargeTransport double (carriers = 1e18·doping at 0 V)."""
    return _serve("ct_linear")


def serve_gyptis_mean() -> tuple[Tesseract, ModuleType]:
    """Serve the effective-medium gyptis double (``neff_sq = mean(eps)``)."""
    return _serve("gyptis_mean")


def serve_gyptis_sumsq() -> tuple[Tesseract, ModuleType]:
    """Serve the structure-sensitive gyptis double (``neff_sq = sum(eps**2)``)."""
    return _serve("gyptis_sumsq")
