#!/usr/bin/env bash
set -euo pipefail

# Run a Python command with ASan/LSan configured for better results on Linux.
#
# Usage:
#   scripts/run_lsan.sh python tests/run_all.py
#   scripts/run_lsan.sh python tests/write.py

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <command...>" >&2
  exit 2
fi

# Try to locate an ASan runtime to preload.
asan_so=""

die() {
  echo "ERROR: $*" >&2
  exit 1
}

# Prevent mixing multiple ASan runtimes.
if [[ -n "${LD_PRELOAD:-}" ]] && echo ":${LD_PRELOAD}:" | grep -Eq ':(.*/)?libasan\.so(\.[0-9]+)*:|:(.*/)?libclang_rt\.asan[^:]*:'; then
  die "LD_PRELOAD already contains an ASan runtime; unset it to avoid mixing runtimes. (LD_PRELOAD='$LD_PRELOAD')"
fi

# Allow explicit override.
if [[ -n "${LIGHTMEM_ASAN_RUNTIME:-}" ]]; then
  if [[ ! -f "${LIGHTMEM_ASAN_RUNTIME}" ]]; then
    die "LIGHTMEM_ASAN_RUNTIME does not exist: ${LIGHTMEM_ASAN_RUNTIME}"
  fi
  asan_so="${LIGHTMEM_ASAN_RUNTIME}"
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

debug_log() {
  if [[ "${LIGHTMEM_ASAN_DEBUG:-}" == "1" ]]; then
    echo "[run_lsan] $*" >&2
  fi
}

find_ext_origin() {
  local py="$1"
  "$py" - <<'PY'
import importlib.util
import sys

spec = importlib.util.find_spec("light_mem.light_mem")
if spec is None or not getattr(spec, "origin", None):
    sys.exit(1)
print(spec.origin)
PY
}

detect_ext_so() {
  local py="$1"
  "$py" - <<'PY'
import glob
import os
import sys
import sysconfig

candidates = []

paths = []
try:
    p = sysconfig.get_paths()
    for k in ("platlib", "purelib"):
        v = p.get(k)
        if v:
            paths.append(v)
except Exception:
    pass

# Also consider repo-local layout (editable builds sometimes place .so next to sources).
repo_root = os.environ.get("LIGHTMEM_REPO_ROOT", "")
if repo_root:
    paths.append(os.path.join(repo_root, "python"))

seen = set()
for base in paths:
    if not base or base in seen:
        continue
    seen.add(base)
    candidates.extend(glob.glob(os.path.join(base, "light_mem", "light_mem*.so")))
    candidates.extend(glob.glob(os.path.join(base, "light_mem", "light_mem*.cpython-*.so")))

for p in candidates:
    if os.path.exists(p):
        print(p)
        sys.exit(0)

sys.exit(1)
PY
}

needed_entry() {
  local so_path="$1"
  if command -v readelf >/dev/null 2>&1; then
    readelf -d "$so_path" 2>/dev/null | awk -F'[][]' '/NEEDED/{print $2}' || true
  elif command -v objdump >/dev/null 2>&1; then
    objdump -p "$so_path" 2>/dev/null | awk '/NEEDED/{print $2}' || true
  else
    return 1
  fi
}

find_libasan_from_ldconfig() {
  local major="${1:-}"
  if ! command -v ldconfig >/dev/null 2>&1; then
    return 1
  fi
  if [[ -n "$major" ]]; then
    ldconfig -p 2>/dev/null | awk -v m="$major" '$1 ~ ("libasan\\.so\\." m "$") {print $NF; exit}'
  else
    ldconfig -p 2>/dev/null | awk '/libasan\.so/{print $NF; exit}'
  fi
}

find_libasan_in_conda() {
  local needed_name="$1"
  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    return 1
  fi
  local cand
  cand="$CONDA_PREFIX/lib/$needed_name"
  if [[ -f "$cand" ]]; then
    echo "$cand"
    return 0
  fi
  # Also accept libasan.so without major.
  if [[ "$needed_name" != "libasan.so" ]] && [[ -f "$CONDA_PREFIX/lib/libasan.so" ]]; then
    echo "$CONDA_PREFIX/lib/libasan.so"
    return 0
  fi
  return 1
}

find_libclang_rt_asan() {
  local needed_name="$1"
  local arch
  arch="$(uname -m)"

  # 1) If clang is present, prefer its resource dir.
  if command -v clang >/dev/null 2>&1; then
    local resdir cand
    resdir="$(clang --print-resource-dir 2>/dev/null || true)"
    if [[ -n "$resdir" ]]; then
      cand="$resdir/lib/linux/$needed_name"
      if [[ -f "$cand" ]]; then
        echo "$cand"
        return 0
      fi
      # Fallback: construct common name if NEEDED didn't include it.
      case "$arch" in
        x86_64) cand="$resdir/lib/linux/libclang_rt.asan-x86_64.so" ;;
        aarch64|arm64) cand="$resdir/lib/linux/libclang_rt.asan-aarch64.so" ;;
        *) cand="" ;;
      esac
      if [[ -n "$cand" && -f "$cand" ]]; then
        echo "$cand"
        return 0
      fi
    fi
  fi

  # 2) Search in conda prefix if available.
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    local found
    found="$(find "$CONDA_PREFIX" -path '*/lib/clang/*/lib/linux/*' -name "$needed_name" -print -quit 2>/dev/null || true)"
    if [[ -n "$found" && -f "$found" ]]; then
      echo "$found"
      return 0
    fi
  fi

  return 1
}

# Auto-detect and prefer the runtime that the built extension links against.
if [[ -z "$asan_so" ]]; then
  cmd0="$1"
  pybin=""
  if [[ "$cmd0" == "python" || "$cmd0" == python3* || "$cmd0" == */python* ]]; then
    pybin="$(command -v "$cmd0" 2>/dev/null || true)"
    [[ -z "$pybin" ]] && pybin="$cmd0"
  fi

  if [[ -n "$pybin" ]] && [[ -x "$pybin" ]]; then
    export LIGHTMEM_REPO_ROOT="$repo_root"
    # Prefer resolving the exact extension path without importing it.
    ext_so="$(find_ext_origin "$pybin" 2>/dev/null || true)"
    if [[ -z "$ext_so" ]]; then
      ext_so="$(detect_ext_so "$pybin" 2>/dev/null || true)"
    fi
    if [[ -n "$ext_so" && -f "$ext_so" ]]; then
      debug_log "extension: $ext_so"
      needed_list="$(needed_entry "$ext_so" | tr -d '\r' || true)"
      if [[ -n "$needed_list" ]]; then
        debug_log "NEEDED: $(echo "$needed_list" | tr '\n' ' ')"
      fi

      # Prefer clang compiler-rt ASan if the module needs it.
      clang_needed="$(echo "$needed_list" | awk '/^libclang_rt\.asan/{print; exit}')"
      if [[ -n "$clang_needed" ]]; then
        cand="$(find_libclang_rt_asan "$clang_needed" 2>/dev/null || true)"
        if [[ -n "$cand" && -f "$cand" ]]; then
          asan_so="$cand"
        fi
      fi

      # Otherwise, prefer matching libasan.so.<major> if present.
      if [[ -z "$asan_so" ]]; then
        asan_needed="$(echo "$needed_list" | awk '/^libasan\.so\./{print; exit}')"
        if [[ -n "$asan_needed" ]]; then
          major="$(echo "$asan_needed" | sed -E 's/^libasan\.so\.([0-9]+)$/\1/' )"
          cand="$(find_libasan_in_conda "$asan_needed" 2>/dev/null || true)"
          if [[ -z "$cand" ]]; then
            cand="$(find_libasan_from_ldconfig "$major" 2>/dev/null || true)"
          fi
          if [[ -n "$cand" && -f "$cand" ]]; then
            asan_so="$cand"
          fi
        fi
      fi
    fi
  fi
fi

if [[ -z "$asan_so" ]] && command -v gcc >/dev/null 2>&1; then
  cand="$(gcc -print-file-name=libasan.so 2>/dev/null || true)"
  if [[ -n "$cand" && "$cand" != "libasan.so" ]]; then
    asan_so="$cand"
  fi
fi

if [[ -z "$asan_so" ]] && command -v ldconfig >/dev/null 2>&1; then
  cand="$(ldconfig -p 2>/dev/null | awk '/libasan\.so/{print $NF; exit}')"
  if [[ -n "$cand" ]]; then
    asan_so="$cand"
  fi
fi

if [[ -z "$asan_so" ]] && command -v clang >/dev/null 2>&1; then
  resdir="$(clang --print-resource-dir 2>/dev/null || true)"
  if [[ -n "$resdir" ]]; then
    arch="$(uname -m)"
    case "$arch" in
      x86_64) cand="$resdir/lib/linux/libclang_rt.asan-x86_64.so" ;;
      aarch64|arm64) cand="$resdir/lib/linux/libclang_rt.asan-aarch64.so" ;;
      *) cand="" ;;
    esac
    if [[ -n "$cand" && -f "$cand" ]]; then
      asan_so="$cand"
    fi
  fi
fi

debug_log "selected runtime: $asan_so"

resolve_libstdcxx() {
  if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
    echo "$CONDA_PREFIX/lib/libstdc++.so.6"
    return 0
  fi
  if command -v ldconfig >/dev/null 2>&1; then
    ldconfig -p 2>/dev/null | awk '/libstdc\+\+\.so\.6/{print $NF; exit}'
    return 0
  fi
  return 1
}

if [[ -z "$asan_so" ]]; then
  die "Could not locate an ASan runtime. Set LIGHTMEM_ASAN_RUNTIME=/path/to/libasan.so (or libclang_rt.asan*.so)."
fi

# Reduce CPython allocator noise for leak checking.
export PYTHONMALLOC="${PYTHONMALLOC:-malloc}"

set_asan_option() {
  local key="$1"
  local value="$2"
  local base="${ASAN_OPTIONS:-}"
  local filtered=()
  local part
  IFS=':' read -r -a parts <<< "$base"
  for part in "${parts[@]}"; do
    [[ -z "$part" ]] && continue
    [[ "$part" == "$key="* ]] && continue
    filtered+=("$part")
  done
  filtered+=("$key=$value")
  ASAN_OPTIONS="$(IFS=:; echo "${filtered[*]}")"
  export ASAN_OPTIONS
}

# Default: enable leak checking; keep abort_on_error configurable.
export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=1:abort_on_error=1}"

# Suppress known CPython/torch/pybind leaks.
export LSAN_OPTIONS="${LSAN_OPTIONS:-suppressions=$repo_root/tools/lsan.supp}"

# Crash-simulation tests intentionally exit via os._exit()/SIGKILL-like paths.
# Leak checking is not meaningful there (no destructors/atexit), and can flip rc!=0.
disable_leaks="0"
for arg in "$@"; do
  case "$arg" in
    */tests/multi_node/test_09_crash_recovery_subprocess.py|*/tests/multi_node/worker_crash_node.py|*test_09_crash_recovery_subprocess.py|*worker_crash_node.py)
      disable_leaks="1"
      ;;
  esac
done

if [[ "$disable_leaks" == "1" ]]; then
  debug_log "disabling leak detection for crash-simulation test"
  set_asan_option "detect_leaks" "0"
fi

ld_preload="$asan_so"

preload_mode="${LIGHTMEM_PRELOAD_CXXABI:-auto}"

# Preload libstdc++ before ASan so __cxa_throw is resolvable when ASan sets up
# interceptors. This helps avoid:
#   CHECK failed: asan_interceptors.cpp:335 real___cxa_throw == 0
if [[ "$preload_mode" != "0" && "$preload_mode" != "false" && "$preload_mode" != "off" ]]; then
  if [[ "$preload_mode" == "1" || "$preload_mode" == "true" || "$preload_mode" == "on" || "$preload_mode" == "auto" ]]; then
    libstdcxx="$(resolve_libstdcxx 2>/dev/null || true)"
    if [[ -n "$libstdcxx" && -f "$libstdcxx" ]]; then
      debug_log "preloading libstdc++ after ASan: $libstdcxx"
      # ASan runtime must come first in the initial library list.
      ld_preload="$asan_so:$libstdcxx"
    else
      debug_log "libstdc++ not found for preload (mode=$preload_mode)"
    fi
  fi
fi

exec env LD_PRELOAD="$ld_preload" "$@"
