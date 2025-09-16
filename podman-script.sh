#!/usr/bin/env bash
set -e

MACHINE_NAME="podman-machine-default"

echo ">>> Stopping Podman machine (if running)..."
podman machine stop -f "$MACHINE_NAME" || true

echo ">>> Removing old Podman machine..."
podman machine rm -f "$MACHINE_NAME" || true

echo ">>> Initializing new Podman machine..."
podman machine init \
  --cpus=4 \
  --memory=8192 \
  --disk-size=20 \
  "$MACHINE_NAME"

echo ">>> Starting Podman machine..."
podman machine start "$MACHINE_NAME"

echo ">>> Setting default connection..."
podman system connection default "$MACHINE_NAME"

echo ">>> Podman machine reset complete!"
podman info | grep -A5 "host:"
