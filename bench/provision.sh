#!/usr/bin/env bash
# Bring a fresh GPU box up to what bench/run.py needs. Idempotent — safe to
# re-run, and it fixes only what is actually broken.
#
#   curl -fsSL https://raw.githubusercontent.com/alexjaniak/nano-vllm/main/bench/provision.sh | sudo bash
#
# Vast VM templates can only use docker.io/vastai/kvm images, so the guest is
# whatever their tag ships and this closes the gap. Pin the tag rather than
# using @vastai-automatic-tag: auto-select hands you a different guest per
# rental, which is how a jammy box turns up when you asked for noble.
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "run as root (sudo bash provision.sh)"; exit 1; }

say() { printf '\n== %s\n' "$*"; }

# --- freeze the driver ------------------------------------------------------
# Do this before any apt work. jammy-security ships newer driver packages, and
# an apt-get update is enough for unattended-upgrades to install one in the
# background — userspace then jumps ahead of the loaded kernel module and
# nvidia-smi dies with "Driver/library version mismatch" until a DKMS rebuild
# and a reboot. The hold stays on afterwards; a benchmark box wants a driver
# that cannot move, same reason spec-v1.toml pins everything else.
say "freezing driver packages"
drv=$(dpkg -l | awk '/^ii +(nvidia|libnvidia)/ {print $2}' | grep -v container || true)
if [ -n "$drv" ]; then
  # shellcheck disable=SC2086
  apt-mark hold $drv >/dev/null
  echo "held $(echo "$drv" | wc -l) packages at $(cat /proc/driver/nvidia/version 2>/dev/null \
    | awk '{print $8}' || echo unknown)"
fi

# --- driver -----------------------------------------------------------------
say "driver"
if ! nvidia-smi -L >/dev/null 2>&1; then
  # Userspace and module out of step — the packages are usually fine, only the
  # built module is stale, so name the rebuild rather than the hardware.
  if nvidia-smi 2>&1 | grep -qi 'version mismatch'; then
    dkms_pkg=$(dpkg -l | awk '/^ii +nvidia-dkms/ {print $2}' | head -1)
    cat <<EOF
FATAL: driver userspace and the loaded kernel module disagree.
  loaded: $(cat /proc/driver/nvidia/version 2>/dev/null | awk '{print $8}' || echo '?')
  fix:    apt-get install --reinstall -y ${dkms_pkg:-nvidia-dkms-<branch>}
          update-initramfs -u && reboot
EOF
    exit 1
  fi
  # Consumer Blackwell (5090, sm_120) only works with the OPEN kernel modules.
  # Given the proprietary flavor the module loads, taints, then finds zero
  # devices — nvidia-smi says "No devices were found" while lspci shows the
  # card, so it reads like broken passthrough rather than a wrong package.
  if modinfo nvidia 2>/dev/null | grep -qi '^license: *NVIDIA'; then
    branch=$(modinfo nvidia | awk '/^version:/ {split($2, v, "."); print v[1]}')
    cat <<EOF
FATAL: proprietary NVIDIA modules installed; Blackwell needs the open ones.
  apt-mark unhold nvidia-driver-${branch} || true
  apt-get install -y nvidia-driver-${branch}-open
  reboot
Then re-run this script. Better: pick a VM image tag that ships open modules.
EOF
    exit 1
  fi
  echo "FATAL: no usable GPU. Check lspci for the card and dmesg for the driver."
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# --- container toolkit ------------------------------------------------------
# The quiet one: without it every engine container starts CPU-only, so a whole
# sweep completes and measures nothing.
say "container toolkit"
if docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L >/dev/null 2>&1; then
  echo "already working"
else
  # --yes: gpg prompts on an existing keyring, which would hang a curl|bash run.
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

# --- python -----------------------------------------------------------------
# run.py is stdlib-only but needs tomllib (3.11+). Jammy ships 3.10.
say "python"
if python3 -c 'import tomllib' 2>/dev/null; then
  PY=python3
else
  apt-get install -y -qq python3.11
  PY=python3.11
fi
$PY --version

# --- verify -----------------------------------------------------------------
# The driver can be upgraded out from under a loaded module by anything that
# runs apt, so re-check it here rather than trusting the earlier pass.
say "verify"
nvidia-smi -L
docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L
docker compose version
$PY -c 'import tomllib; print("tomllib ok")'
df -h --output=avail / | tail -1 | tr -d ' ' | awk '{print "disk free: " $1}'

printf '\nready:  %s bench/run.py --dry-run\n' "$PY"
