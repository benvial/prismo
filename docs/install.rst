:description: How to install prismo and build its two Tesseract solver images.

.. _install:

Installation
============

.. rst-class:: lead

   The host app is a small Python package; the two solvers are Tesseract
   container images you build once.

----

Prerequisites
-------------

- Linux or macOS (Windows via WSL2), ``make``
- Docker (Tesseract builds and runs the solvers as containers)
- Python ≥ 3.10 in an active virtual environment
- ~10 GB of disk for the two images

Host app
--------

.. tab-set::
    :class: outline

    .. tab-item:: :iconify:`devicon:pypi` pip

        .. code-block:: bash

            git clone https://github.com/benvial/prismo && cd prismo
            make install          # pip install -e components/shared_code -e "app[dev]"

    .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            git clone https://github.com/benvial/prismo && cd prismo/app
            uv sync               # prismo_shared is a path dependency

This installs ``prismo_shared`` (the Pydantic schemas both solvers and the app
agree on) and the ``prismo`` CLI with JAX, NLopt, matplotlib and
``tesseract-core``. No solver runs on the host.

Solver images
-------------

.. code-block:: bash

    make julia-base chargetransport   # Julia 1.10 + precompiled ChargeTransport.jl (~15 min, once)
    make build                        # tesseract build both components (gyptis is a conda image)
    make test                         # component regression cases + host unit tests

``make images`` reports whether an image is older than the sources it was built
from. The Julia base image only needs rebuilding when
``components/tesseracts/chargetransport/julia_env/*.toml`` change.

Without installing anything
---------------------------

|binder| opens ``notebooks/prismo.ipynb`` on mybinder.org in a JupyterLab
where both solvers are installed *in-process*: gyptis + legacy FEniCS from
conda-forge and Julia 1.10 with the ChargeTransport.jl environment pinned to the
same ``Manifest.toml`` as the container image. Binder has no Docker, so that
session uses the ``make run`` path (the rib mesh authored by
``prismo.waveguide_mesh``, no containers) with about one CPU and 2 GB of RAM —
fine for the gradient check and a short optimization; the 200-iteration runs in
:doc:`results` take hours there. The image is described by ``binder/``:
``environment.yml`` (conda), ``Project.toml`` + ``Manifest.toml`` (Julia, kept
identical to the component's by a unit test) and ``postBuild`` (pip-installs the
app, warms the FEniCS and Julia caches).

.. |binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/benvial/prismo/main?urlpath=lab/tree/notebooks/prismo.ipynb

Docs
----

.. code-block:: bash

    pip install -e "app[docs]"
    make docs                         # docs/_build/html
