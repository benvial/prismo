"""Direct tests for the differentiable-component adapter.

Exercises the adapter through its own interface -- forward value, vjp and
gradient, JIT -- and the hard-error routing when no backend is bound, without
reaching into pipeline module internals.

Ref: pipeline-deepening ticket 01; rethink ticket 04 (delete fake fallbacks).
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from prismo.differentiable_component import (  # noqa: E402
    DifferentiableComponent,
    invoke_tesseract,
)

jax.config.update("jax_enable_x64", True)


def _squaring_component() -> DifferentiableComponent:
    """A component whose real backend squares its input elementwise."""

    def forward(x_np, scale=1.0):
        return np.asarray(scale) * x_np**2

    def vjp(x_np, g_np, scale=1.0):
        return g_np * (2.0 * scale * x_np)

    def out_struct(x, scale=1.0):
        return jax.ShapeDtypeStruct(x.shape, x.dtype)

    return DifferentiableComponent(forward=forward, vjp=vjp, out_struct=out_struct)


class TestBackedComponent:
    def test_forward_value_uses_real_backend(self):
        comp = _squaring_component()
        x = jnp.asarray([1.0, 2.0, 3.0])
        np.testing.assert_allclose(comp(x), [1.0, 4.0, 9.0])

    def test_forward_honours_static_arg(self):
        comp = _squaring_component()
        x = jnp.asarray([1.0, 2.0, 3.0])
        np.testing.assert_allclose(comp(x, 2.0), [2.0, 8.0, 18.0])

    def test_gradient_uses_supplied_vjp(self):
        comp = _squaring_component()
        x = jnp.asarray([1.0, 2.0, 3.0])
        grad = jax.grad(lambda v: jnp.sum(comp(v)))(x)
        np.testing.assert_allclose(grad, 2.0 * np.asarray([1.0, 2.0, 3.0]))

    def test_gradient_vs_finite_difference(self):
        comp = _squaring_component()
        x = jnp.asarray([0.5, 1.5, 2.5])

        def loss(v):
            return jnp.sum(comp(v, 1.3) ** 2)

        grad = jax.grad(loss)(x)
        h = 1e-6
        for i in range(x.shape[0]):
            e = jnp.zeros_like(x).at[i].set(h)
            fd = float((loss(x + e) - loss(x - e)) / (2 * h))
            assert abs(fd - float(grad[i])) < 1e-4

    def test_jit_forward_and_gradient(self):
        comp = _squaring_component()
        x = jnp.asarray([1.0, 2.0, 3.0])
        val = jax.jit(lambda v: comp(v, 2.0))(x)
        np.testing.assert_allclose(val, [2.0, 8.0, 18.0])
        grad = jax.jit(jax.grad(lambda v: jnp.sum(comp(v))))(x)
        np.testing.assert_allclose(grad, [2.0, 4.0, 6.0])


class TestNoBackend:
    """No physics-free fallback: routing a call without a backend is a hard error."""

    def test_invoke_tesseract_without_backend_raises(self):
        with pytest.raises(RuntimeError, match="no component backend available"):
            invoke_tesseract(
                None,
                None,
                container_call=lambda tess: tess,
                local_call=lambda api: api,
            )

    def test_component_backed_by_missing_backend_raises_on_call(self):
        """A component whose forward routes to a missing backend raises, not fabricates."""

        def forward(x_np):
            return invoke_tesseract(
                None,
                None,
                container_call=lambda tess: x_np,
                local_call=lambda api: x_np,
            )

        comp = DifferentiableComponent(
            forward=forward,
            vjp=lambda x_np, g_np: g_np,
            out_struct=lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
        )
        with pytest.raises(Exception, match="backend"):
            comp(jnp.asarray([1.0, 2.0, 3.0]))


class TestTupleOutputComponent:
    """A component returning a tuple output with a tuple cotangent."""

    def _component(self) -> DifferentiableComponent:
        def forward(x_np):
            return x_np**2, x_np**3

        def vjp(x_np, g_np):
            g_sq, g_cube = g_np
            return g_sq * (2.0 * x_np) + g_cube * (3.0 * x_np**2)

        def out_struct(x):
            s = jax.ShapeDtypeStruct(x.shape, x.dtype)
            return (s, s)

        return DifferentiableComponent(forward=forward, vjp=vjp, out_struct=out_struct)

    def test_tuple_forward(self):
        comp = self._component()
        x = jnp.asarray([1.0, 2.0, 3.0])
        a, b = comp(x)
        np.testing.assert_allclose(a, [1.0, 4.0, 9.0])
        np.testing.assert_allclose(b, [1.0, 8.0, 27.0])

    def test_tuple_gradient(self):
        comp = self._component()
        x = jnp.asarray([1.0, 2.0, 3.0])

        def loss(v):
            a, b = comp(v)
            return jnp.sum(a) + jnp.sum(b)

        grad = jax.grad(loss)(x)
        expected = 2.0 * np.asarray([1.0, 2.0, 3.0]) + 3.0 * np.asarray(
            [1.0, 2.0, 3.0]
        ) ** 2
        np.testing.assert_allclose(grad, expected)
