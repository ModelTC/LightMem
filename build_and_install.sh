#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
set -euo pipefail

uv build \
  -C override=cmake.options.CMAKE_PREFIX_PATH=$(python -c "import torch;print(torch.utils.cmake_prefix_path,end='')") \
  -C override=cmake.options.CMAKE_BUILD_TYPE=Release
  # -C 'override=cmake.build_args=["--parallel","8"]'

pip uninstall -y kvcache
pip install dist/*.whl
