:layout: landing
:description: Differentiable PN-junction photonic phase shifter.

.. container:: title-bar

   .. image:: _static/prismo-name.svg
      :alt: prismo
      :width: 400px


.. rst-class:: lead

    Free-form doping inverse design of a silicon PN-junction phase shifter:
    ChargeTransport.jl (Julia, discrete adjoint) and gyptis/FEniCS
    (eigen-adjoint) composed as Tesseracts into one ``jax.grad``.

.. container:: buttons

    :doc:`Install <install>`
    :doc:`Guide <problem>`
    :doc:`Results <results>`
    `GitHub <https://github.com/benvial/prismo>`_

.. grid:: 1 1 2 3
   :gutter: 2
   :padding: 0
   :class-row: surface

   .. grid-item-card:: :iconify:`mdi:puzzle` Multi-Tesseract pipeline

      Combines several Tesseracts — eigen-adjoint (gyptis) and discrete
      adjoint (ChargeTransport.jl) — into one differentiable pipeline.
      Shared Pydantic schemas define the exchange format across solver
      boundaries.

   .. grid-item-card:: :iconify:`mdi:vector-triangle` Mesh & geometry

      One shared Gmsh mesh of the SOI cross-section for both solvers,
      plus density filtering of the signed design field.

   .. grid-item-card:: :iconify:`mdi:chart-line` Differentiable optimization

      End-to-end adjoint gradients through the PN-junction phase shifter
      model, from the signed doping field to the effective-index figure of
      merit, validated against finite differences.

   .. grid-item-card:: :iconify:`mdi:docker` Containerized solvers

      Each solver runs as a standalone Tesseract container with its own
      automatic differentiation strategy.

   .. grid-item-card:: :iconify:`mdi:console` CLI entry point

      Run the full pipeline from the command line with ``prismo``.

   .. grid-item-card:: :iconify:`tabler:circuit-diode` Soref–Bennett model

      Carrier-induced index change through the Soref–Bennett equations.


.. toctree::
    :caption: Getting started
    :hidden:

    install
    usage

.. toctree::
    :caption: Guide
    :hidden:

    problem
    physics
    adjoint
    design
    architecture
    results
    glossary

.. toctree::
    :caption: API
    :hidden:

    api
