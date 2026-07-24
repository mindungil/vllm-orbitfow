#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: NVIDIA_USERSPACE_ROOT=<path> $0 <command> [args...]" >&2
    exit 2
fi

root="${NVIDIA_USERSPACE_ROOT:?NVIDIA_USERSPACE_ROOT must be set}"
libdir="${root}/root/usr/lib/x86_64-linux-gnu"
driver_version="${NVIDIA_DRIVER_VERSION:-}"

if [[ ! -r "${libdir}/libcuda.so.1" || ! -r "${libdir}/libnvidia-ml.so.1" ]]; then
    echo "No extracted NVIDIA compute libraries found under ${libdir}" >&2
    exit 1
fi
if [[ -n "${driver_version}" ]]; then
    for library in libcuda libnvidia-ml libnvidia-ptxjitcompiler; do
        if [[ ! -r "${libdir}/${library}.so.${driver_version}" ]]; then
            echo "Missing ${library}.so.${driver_version} under ${libdir}" >&2
            exit 1
        fi
    done
fi

export LD_LIBRARY_PATH="${libdir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "$@"
