:description: Here is the guide on how to install prismo.

.. _install:

Installation
============

.. rst-class:: lead

   Install **prismo** as a Python package.

----

Prismo is a multi-Tesseract project: a monorepo that combines several
Tesseracts into a differentiable pipeline for topology optimization of a
reverse-biased PN-junction phase shifter. Each solver is a standalone
Tesseract with its own automatic differentiation strategy — eigen-adjoint
(gyptis) and discrete adjoint (ChargeTransport.jl).

package install
---------------

.. tab-set::
    :class: outline

    .. tab-item:: :iconify:`devicon:pypi` pip

        .. code-block:: bash

            pip install .

    .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            uv sync

docs install
------------

.. code-block:: bash

    uv sync --extra docs
    cd docs
    make html
