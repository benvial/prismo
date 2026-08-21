# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""A forward-solve session that gates its own adjoint.

Both differentiable Tesseract components enforce the same contract: the
adjoint (``vector_jacobian_product``) may only run after a forward solve
(``apply``) on identical inputs, so it can reuse that solve's state. The
Tesseract runtime fixes the shapes of both endpoints, so a session value
cannot be threaded across the boundary as a return value — it must sit
*behind* the endpoints. The forward solve therefore stores the session it
produces in a :class:`SolveSessionRegistry`, and the adjoint retrieves the one
whose inputs match and reuses its carried state.

The ordering-and-input-match requirement thus lives on these values rather than
in a module-global set of solve-state keys.

A single autodiff pass can open several forward solves on one component before
any adjoint runs — for instance ChargeTransport solved at two bias voltages on
the same doping. The registry keeps each such session concurrently, keyed by
its input identity. An optional ``scope`` bounds retention: opening a session
under a new scope drops sessions from prior scopes, so a component solved at
several operating points keeps them all within one autodiff pass while stale
sessions from earlier passes are evicted. Omitting the scope makes each session
its own scope, so the registry holds only the most-recent forward.

Some components have no scope narrow enough to turn over between passes: the
gyptis background and perturbed solves share only the constant surroundings, so
scoping by those keeps one scope for the whole optimization. ``max_sessions``
bounds those registries directly, evicting the least recently used session
(counting both ``open`` and ``match``) once the cap is reached.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["SolveSession", "SolveSessionRegistry", "array_identity"]

_UNSET: Any = object()


def array_identity(array: np.ndarray) -> tuple[Any, ...]:
    """Return an exact, hashable fingerprint of ``array``'s shape and content.

    Shared by both components to key a :class:`SolveSessionRegistry` on the
    array a forward solve ran on, so its adjoint retrieves the matching session.
    """
    return (array.shape, array.dtype.str, array.tobytes())


class SolveSession:
    """A single forward solve, carrying the state its adjoint reuses.

    A registry keys these by their input identity; the session is the value the
    adjoint retrieves and reads state from.

    Attributes:
        state: Whatever the adjoint needs from the forward solve (e.g. the
            assembled simulation and wavenumber). May be ``None`` when matching
            the identity alone gates the adjoint and the numerical state lives
            elsewhere — for instance inside a persistent worker keyed by the
            same inputs.
    """

    def __init__(self, state: Any = None) -> None:
        self.state = state


class SolveSessionRegistry:
    """Holds the open forward :class:`SolveSession` values, keyed by identity.

    Each entry is a session value the adjoint reuses, not a bare solve-state
    key. ``scope`` bounds how many coexist: sessions opened under one scope
    coexist; opening a session under a new scope first drops every earlier one.

    Args:
        max_sessions: Cap on how many sessions one scope retains. ``None``
            (the default) keeps every session of the current scope, which is
            right when the scope itself turns over each autodiff pass. A
            registry whose scope spans many passes needs the cap so its state
            -- assembled operators, eigenvectors -- does not accumulate; the
            least recently used session is evicted when the cap is reached.
    """

    def __init__(self, max_sessions: int | None = None) -> None:
        if max_sessions is not None and max_sessions < 1:
            raise ValueError("max_sessions must be at least one")
        self._sessions: dict[Any, SolveSession] = {}
        self._scope: Any = _UNSET
        self._max_sessions = max_sessions

    def open(
        self, identity: Any, state: Any = None, *, scope: Any = _UNSET
    ) -> SolveSession:
        """Record a forward solve and return its session.

        Args:
            identity: Exact fingerprint of the inputs this solve ran on; the
                adjoint retrieves the session by matching it.
            state: State the adjoint reuses (see :class:`SolveSession`).
            scope: Grouping that bounds retention. Sessions sharing a scope
                coexist; a new scope evicts sessions from prior scopes.
                Defaults to ``identity``, so the registry keeps only the
                most-recent forward when no scope is given.
        """
        if scope is _UNSET:
            scope = identity
        if scope != self._scope:
            self._sessions.clear()
            self._scope = scope
        session = SolveSession(state)
        self._sessions.pop(identity, None)
        self._sessions[identity] = session
        if self._max_sessions is not None:
            while len(self._sessions) > self._max_sessions:
                # dicts preserve insertion order, and both open and match move
                # an entry to the end, so the first key is the least recent.
                del self._sessions[next(iter(self._sessions))]
        return session

    def match(self, identity: Any) -> SolveSession | None:
        """Return the retained session with ``identity``, or ``None``.

        A match counts as a use: under ``max_sessions`` it makes the session
        the most recently used, so a session an adjoint is still working
        through is not the one evicted next.
        """
        session = self._sessions.get(identity)
        if session is not None and self._max_sessions is not None:
            del self._sessions[identity]
            self._sessions[identity] = session
        return session

    def has_any(self) -> bool:
        """Return whether any forward session is currently retained."""
        return bool(self._sessions)

    def clear(self) -> None:
        """Forget every retained session (e.g. on lifecycle shutdown)."""
        self._sessions.clear()
        self._scope = _UNSET
