"""The Binder image pins the same Julia environment as the chargetransport component.

``binder/Project.toml`` and ``binder/Manifest.toml`` are copies of
``components/tesseracts/chargetransport/julia_env/`` (repo2docker only reads
them from the ``binder/`` directory, and ``JULIA_PROJECT`` points there at
runtime). The Binder copy adds a ``[compat]`` entry so repo2docker installs
Julia 1.10; everything else must match, or the in-process solver on Binder
runs against a different dependency graph than the tested container.
"""

from pathlib import Path

import tomllib

_ROOT = Path(__file__).resolve().parents[2]
_JULIA_ENV = _ROOT / "components" / "tesseracts" / "chargetransport" / "julia_env"
_BINDER = _ROOT / "binder"


def test_binder_project_pins_the_component_dependencies() -> None:
    component = tomllib.loads((_JULIA_ENV / "Project.toml").read_text())
    binder = tomllib.loads((_BINDER / "Project.toml").read_text())
    assert binder["deps"] == component["deps"]


def test_binder_project_pins_julia_1_10() -> None:
    binder = tomllib.loads((_BINDER / "Project.toml").read_text())
    # repo2docker treats a bare "1.10" as ^1.10 (latest 1.x); the tilde keeps 1.10.x.
    assert binder["compat"]["julia"] == "~1.10"


def test_binder_manifest_is_the_component_manifest() -> None:
    component = (_JULIA_ENV / "Manifest.toml").read_text()
    binder = (_BINDER / "Manifest.toml").read_text()
    assert binder == component
