#!/bin/bash
# @author: riccardobussola
# Runs a command in a throwaway container of the locosim image, from the host: training and
# evaluation without the lab container.  The command runs in robot_control/base_controllers, with
# ~/trento_lab_home mounted as /root (as in the lab container) and tqdm and onnx installed (training
# and the ONNX export need them); its output goes to stdout.
#
#   run_in_image.sh <command ...>
#   run_in_image.sh python3 components/safe_rl/value_function/train_value_function.py --data ...
#
# TRENTO_LAB_HOME (default ~/trento_lab_home), IMAGE (default mfocchi/trento_lab_framework:locosim)
# and CPUS (default 4: JAX takes every core otherwise, slowing simulations running alongside).
set -e
if [ $# -lt 1 ]; then sed -n 3,12p "$0"; exit 1; fi
IMAGE=${IMAGE:-mfocchi/trento_lab_framework:locosim}
LAB_HOME=${TRENTO_LAB_HOME:-$HOME/trento_lab_home}
cmd=$(printf '%q ' "$@")  # the arguments as given, quoting kept
# OUT, MODELS and FIGURES (run_pipeline.sh, summarize.sh) are passed on when set
docker run --rm --cpus "${CPUS:-4}" -v "$LAB_HOME":/root ${OUT:+-e OUT=$OUT} ${MODELS:+-e MODELS=$MODELS} \
  ${FIGURES:+-e FIGURES=$FIGURES} \
  -w /root/ros_ws/src/locosim/robot_control/base_controllers "$IMAGE" bash -c \
  "pip3 install -q tqdm onnx >/dev/null 2>&1; export LOCOSIM_DIR=/root/ros_ws/src/locosim PYTHONDONTWRITEBYTECODE=1 && \
   $cmd; status=\$?; chown -R $(id -u):$(id -g) components/safe_rl; exit \$status"
