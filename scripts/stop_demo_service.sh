#!/usr/bin/env bash

# Stop the SplitOculo remote demo services started by start_demo_service.sh.
#
# Usage:
#   ./scripts/stop_demo_service.sh
#
# This delegates to the existing start/status/stop helper so that the same PID
# files and command-line ownership checks are used. It will stop only the
# SplitOculo cloud server and the ngrok process recorded by that helper.

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT_DIR/scripts/start_demo_service.sh" stop
