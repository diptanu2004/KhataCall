"""Every module imports cleanly.

A syntax error in `scripts/record.py` once survived a green run of 138 tests,
because nothing imported it: the scripts are entry points, exercised by hand,
not by the suite. This is the cheapest possible guard against that — it catches
syntax errors, bad imports and typos in module-level code across the whole
tree, including the scripts.

It deliberately imports rather than merely compiling, so a missing dependency
or a broken import line fails here too. Importing must stay free of side
effects: no network, no audio device, no files written. If adding a module here
makes the suite do any of those, the module is doing too much at import time.
"""

import importlib
import pkgutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("app", "eval", "scripts")


def module_names() -> list[str]:
    names = []
    for package in PACKAGES:
        directory = ROOT / package
        if not directory.is_dir():
            continue
        names.append(package)
        for info in pkgutil.walk_packages([str(directory)], prefix=f"{package}."):
            names.append(info.name)
    return sorted(set(names))


@pytest.mark.parametrize("name", module_names())
def test_module_imports(name):
    importlib.import_module(name)


def test_the_scripts_are_actually_covered():
    """Guard the guard: if scripts stop being discovered, this test is useless."""
    names = module_names()
    for expected in ("scripts.record", "scripts.calibrate", "scripts.smoke", "scripts.usage"):
        assert expected in names, f"{expected} is not being import-checked"
