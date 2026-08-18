"""Tests for the shared SolveSession seam.

The seam makes the "call the forward solve before the adjoint, with identical
inputs" contract a property of values handed back by forward solves, rather
than a module-global lookup. A registry holds the open sessions keyed by input
identity; the adjoint retrieves the one whose inputs match. An optional scope
bounds how many coexist.
"""

import numpy as np

from prismo_shared.session import (
    SolveSession,
    SolveSessionRegistry,
    array_identity,
)


def test_session_carries_forward_state_to_the_adjoint() -> None:
    payload = {"simu": object(), "k0": 4.05}
    session = SolveSession(state=payload)
    assert session.state is payload


def test_session_state_defaults_to_none() -> None:
    assert SolveSession().state is None


def test_array_identity_is_stable_for_equal_arrays() -> None:
    assert array_identity(np.array([1.0, 2.25, 12.1])) == array_identity(
        np.array([1.0, 2.25, 12.1])
    )


def test_array_identity_distinguishes_content_shape_and_dtype() -> None:
    base = np.array([1.0, 2.25, 12.1])
    assert array_identity(base) != array_identity(np.array([1.0, 2.25, 12.2]))
    assert array_identity(base) != array_identity(np.array([1.0, 2.25]))
    assert array_identity(base) != array_identity(base.astype(np.float32))


def test_registry_starts_empty() -> None:
    registry = SolveSessionRegistry()
    assert not registry.has_any()
    assert registry.match("id") is None


def test_registry_returns_the_session_opened_on_it() -> None:
    registry = SolveSessionRegistry()
    session = registry.open("id", state=42)
    assert registry.match("id") is session
    assert session.state == 42


def test_registry_without_scope_keeps_only_the_most_recent() -> None:
    registry = SolveSessionRegistry()
    registry.open("first")
    registry.open("second")
    assert registry.match("first") is None
    assert registry.match("second") is not None


def test_registry_retains_coexisting_sessions_under_one_scope() -> None:
    registry = SolveSessionRegistry()
    registry.open(("doping", 0.0), scope="doping")
    registry.open(("doping", -5.0), scope="doping")
    assert registry.match(("doping", 0.0)) is not None
    assert registry.match(("doping", -5.0)) is not None


def test_registry_new_scope_evicts_prior_scope_sessions() -> None:
    registry = SolveSessionRegistry()
    registry.open(("dopingA", 0.0), scope="dopingA")
    registry.open(("dopingA", -5.0), scope="dopingA")
    registry.open(("dopingB", 0.0), scope="dopingB")
    assert registry.match(("dopingA", 0.0)) is None
    assert registry.match(("dopingA", -5.0)) is None
    assert registry.match(("dopingB", 0.0)) is not None


def test_registry_clear_forgets_all_sessions() -> None:
    registry = SolveSessionRegistry()
    registry.open(("doping", 0.0), scope="doping")
    registry.open(("doping", -5.0), scope="doping")
    registry.clear()
    assert not registry.has_any()


def test_registry_match_distinguishes_absent_from_mismatch() -> None:
    registry = SolveSessionRegistry()
    assert registry.match("id") is None
    assert not registry.has_any()

    registry.open("opened")
    assert registry.match("other") is None
    assert registry.has_any()


def test_registry_keyed_by_array_identity() -> None:
    registry = SolveSessionRegistry()
    registry.open(array_identity(np.array([1.0, 2.25, 12.1])), state={"k0": 4.05})

    assert registry.match(array_identity(np.array([1.0, 2.25, 12.1]))) is not None
    assert registry.match(array_identity(np.array([1.0, 2.25, 12.2]))) is None
