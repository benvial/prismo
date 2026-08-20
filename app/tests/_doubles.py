"""Shared physics-free component doubles for the app tests.

Ticket 04 deleted the implicit no-backend stubs, so tests inject explicit
JAX-native doubles through ``pipeline(..., components=...)`` instead of relying
on a fabricated default. This module holds the one double every test reuses --
identity carriers and an effective-medium (mean) ``neff_sq`` -- so the shape is
defined once rather than re-declared per test file.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
from prismo.pipeline import PipelineComponents


def stub_components(**overrides) -> PipelineComponents:
    """Explicit JAX-native doubles standing in for the absent solver backends.

    Defaults reproduce the retired no-backend behaviour -- identity carriers
    from ChargeTransport and a mean-field ``neff_sq`` from gyptis -- and any
    component can be overridden per test via ``overrides``.
    """

    def ct(doping, bias_voltage, mesh_ref=None):
        return doping, doping

    def gyptis(design_epsilon, core_epsilon=None):
        return jnp.mean(design_epsilon)

    base = PipelineComponents(chargetransport=ct, gyptis=gyptis, gyptis_background=gyptis)
    return replace(base, **overrides)
