:description: Running the optimization, validating the gradient, reading the outputs.

Usage
=====

.. rst-class:: lead

   Everything runs through ``make``; each target is a thin wrapper around the
   ``prismo`` CLI.

----

.. code-block:: bash

    make validate-gradient-containers  # composed adjoint vs central finite differences
    make run-containers                # the optimization; outputs/ gets figures + checkpoint.json
    make animate                       # rebuild doping_evolution.{gif,mp4} from the checkpoint
    make probe-objective-containers RUN_ARGS="--design outputs/checkpoint.json"

Knobs are passed through ``RUN_ARGS`` (``prismo run --help`` lists them all):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Meaning
   * - ``--loss-weight w``
     - Optimize ``Δneff − w·α_mode`` (``w`` in n_eff per dB/cm). Default 0: loss is reported, not penalized.
   * - ``--seed u|lateral|vertical``
     - Initial junction topology. Default ``u``, the seed of the headline run;
       the MMA optimum is local, so this picks the basin.
   * - ``--contact-offset``, ``--domain-width``
     - Geometry of the shared mesh [µm].
   * - ``--mesh-size``
     - Characteristic mesh size [µm] of the shared mesh (coarser = faster).
   * - ``--mode-index k``
     - Which guided mode the eigensolve tracks (0 = fundamental).
   * - ``--r-min``, ``--move-limit``, ``--max-iter``, ``--ftol-rel``
     - Density-filter radius [µm] and MMA controls.

Outputs
-------

Every run prints Δn_eff (warm and cold re-solve), VπLπ, modal loss and
VπLπ·α, and writes to ``outputs/``:

- ``convergence.pdf`` — Δn_eff and VπLπ per iteration
- ``doping_field.pdf`` — initial vs optimized signed design field
- ``mode_field.pdf`` — ``|E|`` of the tracked mode
- ``depletion_field.pdf`` — carriers swept out between 0 V and −5 V under the mode
- ``bias_sweep.pdf`` — Δn_eff, α and VπLπ·α vs bias, seed vs optimized (``--bias-sweep-points 0`` skips it)
- ``gradient_validation.pdf`` — relative error of the adjoint vs step size
- ``loss_convergence.pdf``, ``tradeoff.pdf`` — loss-aware runs
- ``doping_evolution.{gif,mp4}`` — net doping at every evaluation
- ``checkpoint.json`` — best design + full history (``prismo animate`` replays it)

The vocabulary (design field, doping map, move limit, cold re-evaluation, …)
is defined once in the :doc:`glossary`; the reasoning behind the shared mesh,
the pivoting eigensolve and the loss model is in :doc:`design`.
