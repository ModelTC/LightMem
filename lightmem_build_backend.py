"""PEP 517 build backend wrapper for LightMem.

Goal
- `pip install .` (wheel build): keep default CMake options (ASan OFF).
- `pip install -e .` (editable / PEP 660): automatically enable ASan and use a
  separate CMake build directory to avoid cache contamination.

This delegates to `py_build_cmake.build` and only adjusts `config_settings` for
editable builds.
"""

from __future__ import annotations

from typing import Any

import py_build_cmake.build as _base


def _listify(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _with_editable_asan(config_settings: dict[str, Any] | None) -> dict[str, Any]:
    cfg: dict[str, Any] = dict(config_settings or {})

    injected_overrides = [
        "cmake.options.LIGHTMEM_ENABLE_ASAN=true",
        "cmake.build_type=RelWithDebInfo",
        'cmake.build_path=".py-build-cmake_cache/{build_config}-asan"',
    ]

    # py-build-cmake reads overrides from keys in this order:
    #   -o, o, override, --override
    # Put ours in `-o` so they run first, and user-provided `override=...`
    # can still overwrite them.
    existing = _listify(cfg.get("-o"))
    cfg["-o"] = injected_overrides + existing
    return cfg


# ---- Required PEP 517/660 hooks --------------------------------------------


def get_requires_for_build_wheel(config_settings=None):
    return _base.get_requires_for_build_wheel(config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return _base.build_wheel(wheel_directory, config_settings, metadata_directory)


def get_requires_for_build_sdist(config_settings=None):
    return _base.get_requires_for_build_sdist(config_settings)


def build_sdist(sdist_directory, config_settings=None):
    return _base.build_sdist(sdist_directory, config_settings)


def get_requires_for_build_editable(config_settings=None):
    return _base.get_requires_for_build_editable(_with_editable_asan(config_settings))


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    return _base.build_editable(
        wheel_directory, _with_editable_asan(config_settings), metadata_directory
    )
