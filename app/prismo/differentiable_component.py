"""Turn an external Tesseract into a JAX-differentiable callable.

A single deep module that concentrates the AD ceremony every component
call site used to re-derive by hand: the ``custom_vjp`` forward/backward
pair, the ``pure_callback`` bridge into numpy solver code, and the output
shape-structs.

A component is described by three callables:

- ``forward(x_np, *static) -> out_np``: the real (container or local) solve,
  operating on numpy arrays. Its container/local dispatch and serialization
  live here, out of the AD machinery.
- ``vjp(x_np, cotangent, *static) -> in_cotangent_np``: the matching adjoint.
- ``out_struct(x, *static) -> ShapeDtypeStruct pytree``: the forward output
  structure, used to bridge back into JAX.

There is no physics-free stub: a component with no live backend raises rather
than fabricating a value or gradient. Tests compose explicit JAX-native
doubles through the pipeline's ``components=`` seam instead.

Ref: pipeline-deepening ticket 01; rethink ticket 04 (delete fake fallbacks).
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
    local module. With neither, a missing backend is a hard error -- there is
    no silent stub short-circuit -- so a physics-free run fails loudly.
    """
    if container is not None:
        return container_call(container)
    if local_api is not None:
        return local_call(local_api)
    raise RuntimeError("no component backend available for this call")


@dataclass(frozen=True)
class DifferentiableComponent:
    """An external component made JAX-differentiable via one adapter.

    Carries the real ``forward``/``vjp`` numpy callables and the forward
    ``out_struct``. Calling the instance evaluates the component with one
    differentiable array input ``x`` followed by any static (non-differentiated)
    arguments. The ``forward``/``vjp`` callables reach a live backend or raise;
    the adapter never substitutes a physics-free value.
    """

    forward: Callable[..., Any]
    vjp: Callable[..., Any]
    out_struct: Callable[..., Any]

    def __call__(self, x: jax.Array, *static: Any) -> Any:
        """Evaluate the component at ``x`` with any static arguments."""
        return self._impl(x, static)

    def _impl(self, x: jax.Array, static: tuple[Any, ...]) -> Any:
        # Built per call so the frozen dataclass stays hashable and the
        # closed-over ``static`` tuple carries this call's non-diff args.
        @partial(jax.custom_vjp, nondiff_argnums=(1,))
        def call(x: jax.Array, static: tuple[Any, ...]) -> Any:
            return self._forward_value(x, static)

        def call_fwd(x: jax.Array, static: tuple[Any, ...]) -> tuple[Any, jax.Array]:
            return self._forward_value(x, static), x

        def call_bwd(static: tuple[Any, ...], x: jax.Array, g: Any) -> tuple[jax.Array]:
            return (self._vjp_value(x, static, g),)

        call.defvjp(call_fwd, call_bwd)
        return call(x, static)

    def _forward_value(self, x: jax.Array, static: tuple[Any, ...]) -> Any:
        return jax.pure_callback(
            lambda x_np: self.forward(x_np, *static),
            self.out_struct(x, *static),
            x,
        )

    def _vjp_value(self, x: jax.Array, static: tuple[Any, ...], g: Any) -> jax.Array:
        return jax.pure_callback(
            lambda x_np, g_np: self.vjp(x_np, g_np, *static),
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            x,
            g,
        )
