#!/usr/bin/env bash
# Launch the RL quadruped controller at SCHED_FIFO priority, optionally pinned to one core.
# Threads inherit the policy, so the whole start-up runs real-time too.
#
#   ./run_rl_controller.sh --robot go2 --real --input joy          # priority 80, no pinning
#   CPU=3 PRIO=90 ./run_rl_controller.sh --robot go2 --real        # pinned to core 3
#
# Arguments go to rl_quadruped_controller.py.  Pinning helps only on an isolated core (host kernel
# command line: isolcpus=3 nohz_full=3 rcu_nocbs=3 irqaffinity=0-2).  The container must run
# --privileged for SCHED_FIFO.

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

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "Launching: ${CMD[*]} python3 $CONTROLLER $*"
exec "${CMD[@]}" python3 "$CONTROLLER" "$@"
