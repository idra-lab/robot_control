#!/usr/bin/env bash
# Launch the RL quadruped controller with real-time scheduling and, optionally, on an isolated core.
#
# The controller asks for SCHED_FIFO itself, but doing it from outside covers the startup path too -
# imports, the ONNX session, the ROS handshake - and works even where the in-process call is denied.
#
#   ./run_rl_controller.sh --real --input joy          # RT priority, no pinning
#   CPU=3 ./run_rl_controller.sh --real --input joy    # RT priority, pinned to core 3
#   PRIO=90 ./run_rl_controller.sh --real              # different priority
#
# Any argument is passed straight through to rl_quadruped_controller.py.
#
# ----------------------------------------------------------------------------------------------
# Core isolation (optional, biggest win on the real robot)
# ----------------------------------------------------------------------------------------------
# Pinning only helps if nothing else is allowed on that core. To reserve core 3 on a 4-core machine,
# add to the *host* kernel command line (/etc/default/grub, GRUB_CMDLINE_LINUX_DEFAULT) and reboot:
#
#   isolcpus=3 nohz_full=3 rcu_nocbs=3 irqaffinity=0-2
#
#     isolcpus      keeps the scheduler from putting anything else there
#     nohz_full     stops the periodic timer tick on that core
#     rcu_nocbs     moves RCU callback processing off it
#     irqaffinity   steers device interrupts to the other cores
#
# Then run with CPU=3. Verify with:
#   cat /sys/devices/system/cpu/isolated
#   chrt -p $(pgrep -f rl_quadruped_controller)
#   taskset -p $(pgrep -f rl_quadruped_controller)
#
# Note the container must be able to grant these: the locosim container already runs --privileged,
# which is what makes SCHED_FIFO available. Without it, expect the controller to warn and fall back
# to normal scheduling.
# ----------------------------------------------------------------------------------------------

set -euo pipefail

PRIO="${PRIO:-80}"
CPU="${CPU:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTROLLER="$SCRIPT_DIR/rl_quadruped_controller.py"

if [ ! -f "$CONTROLLER" ]; then
    echo "Cannot find $CONTROLLER" >&2
    exit 1
fi

CMD=()
if command -v chrt >/dev/null 2>&1; then
    CMD+=(chrt -f "$PRIO")
else
    echo "chrt not found; running without real-time priority" >&2
fi
if [ -n "$CPU" ]; then
    if command -v taskset >/dev/null 2>&1; then
        CMD+=(taskset -c "$CPU")
        echo "Pinning to CPU $CPU"
    else
        echo "taskset not found; running unpinned" >&2
    fi
fi

# One thread for the numeric libraries: the policy is a small MLP and extra threads only add jitter.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "Launching: ${CMD[*]} python3 $CONTROLLER $*"
exec "${CMD[@]}" python3 "$CONTROLLER" "$@"
