"""Turn an external Tesseract into a JAX-differentiable callable.

A single deep module that concentrates the AD ceremony every component
call site used to re-derive by hand: the ``custom_vjp`` forward/backward
pair, the ``pure_callback`` bridge into numpy solver code, the output
shape-structs, and the short-circuit that keeps the pipeline differentiable
when a component is unavailable.

A component is described by five callables and one predicate:

- ``forward(x_np, *static) -> out_np``: the real (container or local) solve,
  operating on numpy arrays. Its container/local dispatch and serialization
  live here, out of the AD machinery.
- ``vjp(x_np, cotangent, *static) -> in_cotangent_np``: the matching adjoint.
- ``out_struct(x, *static) -> ShapeDtypeStruct pytree``: the forward output
  structure, used to bridge back into JAX.
- ``stub_forward(x, *static) -> out`` / ``stub_vjp(x, g, *static) -> in``:
  JAX-native behaviour when no real backend is available, so the composed
  pipeline still produces a value and a gradient.
- ``available() -> bool``: whether a real backend (container or local) exists.

Ref: pipeline-deepening ticket 01.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, TypeVar

import jax

_T = TypeVar("_T")


def invoke_tesseract(
    container: Any | None,
    local_api: Any | None,
    *,
    container_call: Callable[[Any], _T],
    local_call: Callable[[Any], _T],
) -> _T:
    """Route one real component call to its container or local backend.

    The container backend is preferred when present, then the in-process
    local module. The unavailable (stub) case is handled by
    :class:`DifferentiableComponent` in JAX space and never reaches here, so
    a missing backend is a programming error rather than a stub short-circuit.
    """
    if container is not None:
        return container_call(container)
    if local_api is not None:
        return local_call(local_api)
    raise RuntimeError("no component backend available for this call")


@dataclass(frozen=True)
class DifferentiableComponent:
    """A ``(forward, vjp, out_struct)`` triple made JAX-differentiable.

    Calling the instance evaluates the component with one differentiable
    array input ``x`` followed by any static (non-differentiated) arguments.
    """

    forward: Callable[..., Any]
    vjp: Callable[..., Any]
    out_struct: Callable[..., Any]
    stub_forward: Callable[..., Any]
    stub_vjp: Callable[..., Any]
    available: Callable[[], bool]

    def __call__(self, x: jax.Array, *static: Any) -> Any:
        """Evaluate the component at ``x`` with any static arguments."""
        return self._impl(x, static)

    def _impl(self, x: jax.Array, static: tuple[Any, ...]) -> Any:
        # Built per call so the frozen dataclass stays hashable and the
        # closed-over ``static`` tuple carries this call's non-diff args.
        @partial(jax.custom_vjp, nondiff_argnums=(1,))
        def call(x: jax.Array, static: tuple[Any, ...]) -> Any:
            return self._forward_value(x, static)

        def call_fwd(
            x: jax.Array, static: tuple[Any, ...]
        ) -> tuple[Any, jax.Array]:
            return self._forward_value(x, static), x

        def call_bwd(
            static: tuple[Any, ...], x: jax.Array, g: Any
        ) -> tuple[jax.Array]:
            return (self._vjp_value(x, static, g),)

        call.defvjp(call_fwd, call_bwd)
        return call(x, static)

    def _forward_value(self, x: jax.Array, static: tuple[Any, ...]) -> Any:
        if not self.available():
            return self.stub_forward(x, *static)
        return jax.pure_callback(
            lambda x_np: self.forward(x_np, *static),
            self.out_struct(x, *static),
            x,
        )

    def _vjp_value(self, x: jax.Array, static: tuple[Any, ...], g: Any) -> jax.Array:
        if not self.available():
            return self.stub_vjp(x, g, *static)
        return jax.pure_callback(
            lambda x_np, g_np: self.vjp(x_np, g_np, *static),
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            x,
            g,
        )
