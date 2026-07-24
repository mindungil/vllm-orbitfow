#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

package_version="${1:-580.159.03-1ubuntu1}"
destination="${2:-${HOME}/.local/nvidia-userspace-${package_version%%-*}}"
driver_version="${package_version%%-*}"
driver_branch="${driver_version%%.*}"
package="libnvidia-compute-${driver_branch}"
deb="${package}_${package_version}_amd64.deb"

mkdir -p "${destination}/packages" "${destination}/root"
cd "${destination}/packages"

if [[ ! -f "${deb}" ]]; then
    apt-get download "${package}=${package_version}"
fi
dpkg-deb -x "${deb}" "${destination}/root"

libdir="${destination}/root/usr/lib/x86_64-linux-gnu"
for library in \
    "${libdir}/libcuda.so.${driver_version}" \
    "${libdir}/libnvidia-ml.so.${driver_version}" \
    "${libdir}/libnvidia-ptxjitcompiler.so.${driver_version}"; do
    if [[ ! -r "${library}" ]]; then
        echo "Missing expected library: ${library}" >&2
        exit 1
    fi
done

printf 'Prepared NVIDIA %s user-space libraries at %s\n' \
    "${driver_version}" "${destination}"
printf 'Run commands with:\n'
printf '  NVIDIA_USERSPACE_ROOT=%q NVIDIA_DRIVER_VERSION=%q %q <command> [args...]\n' \
    "${destination}" "${driver_version}" \
    "$(dirname "$0")/run_with_nvidia_userspace.sh"
