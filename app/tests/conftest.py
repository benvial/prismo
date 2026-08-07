"""Pytest path setup for app tests.

Makes the ``app`` package and the shared-code package
``tesseract_photonic_waveguide_shared`` importable when running pytest from
the repository root without installing either package.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

for _path in (_ROOT / "app", _ROOT / "components" / "shared_code"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)
