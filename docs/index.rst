:layout: landing
:description: Differentiable PN-junction photonic phase shifter.

.. raw:: html

    <div class="logo-homepage sy-head-brand">
        <img class="light-logo" src="_static/prismo-name.svg" alt="prismo" height="28" loading="lazy">
        <img class="dark-logo" src="_static/prismo-name-dark.svg" alt="prismo" height="28" loading="lazy">
    </div>


.. rst-class:: lead

    Free-form doping inverse design of a silicon PN-junction phase shifter

.. container:: buttons

    :doc:`Install <install>`
    :doc:`Guide <problem>`
    :doc:`Results <results>`
    `GitHub <https://github.com/benvial/prismo>`_

.. raw:: html

    <p class="badges">
      <a href="https://github.com/benvial/prismo/actions/workflows/test.yaml"><img alt="Tests" src="https://img.shields.io/github/actions/workflow/status/benvial/prismo/test.yaml?branch=main&style=for-the-badge&label=tests&logo=githubactions&logoColor=white&labelColor=1c2024"></a>
      <a href="https://github.com/benvial/prismo/actions/workflows/pre_commit.yml"><img alt="Lint" src="https://img.shields.io/github/actions/workflow/status/benvial/prismo/pre_commit.yml?branch=main&style=for-the-badge&label=lint&logo=ruff&logoColor=white&labelColor=1c2024"></a>
      <a href="https://benvial.github.io/prismo/"><img alt="Docs" src="https://img.shields.io/github/actions/workflow/status/benvial/prismo/docs.yaml?branch=main&style=for-the-badge&label=docs&logo=sphinx&logoColor=white&labelColor=1c2024"></a>
      <a href="https://mybinder.org/v2/gh/benvial/prismo/main?urlpath=lab/tree/notebooks/prismo.ipynb"><img alt="Launch on Binder" src="https://img.shields.io/badge/binder-launch-00a2c7?style=for-the-badge&logo=jupyter&logoColor=white&labelColor=1c2024"></a>
      <br>
      <a href="https://pasteurlabs.ai/tesseract-hackathon-2026/"><img alt="Tesseract Hackathon 2026" src="https://img.shields.io/badge/Tesseract_Hackathon-2026-00a2c7?style=for-the-badge&labelColor=1c2024"></a>
      <a href="https://github.com/pasteurlabs/tesseract-core"><img alt="tesseract-core 1.11" src="https://img.shields.io/badge/tesseract--core-1.11-00a2c7?style=for-the-badge&labelColor=1c2024"></a>
      <a href="https://github.com/benvial/prismo/blob/main/app/pyproject.toml"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10+-00a2c7?style=for-the-badge&logo=python&logoColor=white&labelColor=1c2024"></a>
      <a href="https://github.com/benvial/prismo/blob/main/LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache_2.0-00a2c7?style=for-the-badge&labelColor=1c2024"></a>
    </p>

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
