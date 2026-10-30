#!/bin/bash
# @author: riccardobussola
# Runs saferl_controller_dataset.py in parallel Docker containers, one Gazebo (and ROS master) each.
# Run it on the host, not in the lab container.
#
#   run_batches.sh <first seed> <batches> <episodes per batch> <out folder> [more arguments]
#
# Batch i gets seed <first seed> + i - 1 and writes <out folder>/batch<i>/ and <out folder>/batch<i>.log;
# the out folder is relative to robot_control/base_controllers.  More arguments go to
# saferl_controller_dataset.py, e.g. --robot go2 --filter.  Example, from base_controllers, the
# report's 6 Aliengo training batches:
#
#   components/safe_rl/run_batches.sh 501 6 70 components/safe_rl/value_function/observation_datasets/aliengo/episodes
#
# TRENTO_LAB_HOME (default ~/trento_lab_home) is mounted as /root, as in the lab container; IMAGE
# defaults to mfocchi/trento_lab_framework:locosim.  docker ps lists the batches still running, as
# safe_rl_<out folder name>_<hash of the out path>_<seed>.
set -e
if [ $# -lt 4 ]; then sed -n 3,17p "$0"; exit 1; fi
IMAGE=${IMAGE:-mfocchi/trento_lab_framework:locosim}
LAB_HOME=${TRENTO_LAB_HOME:-$HOME/trento_lab_home}
first=$1; n=$2; episodes=$3; out=$4; shift 4
BC=/root/ros_ws/src/locosim/robot_control/base_controllers
mkdir -p "$LAB_HOME/ros_ws/src/locosim/robot_control/base_controllers/$out"
for i in $(seq 1 "$n"); do
  seed=$((first + i - 1))
  # generated_urdf on a tmpfs: every container writes the robot description there
  docker run -d --rm --name "safe_rl_$(basename "$out")_$(echo "$out" | md5sum | cut -c1-4)_$seed" -v "$LAB_HOME":/root \
    --tmpfs /root/ros_ws/src/locosim/robot_urdf/generated_urdf -w $BC "$IMAGE" bash -c \
    "source /opt/ros/noetic/setup.bash && source /root/ros_ws/install/setup.bash && \
     export LOCOSIM_DIR=/root/ros_ws/src/locosim ROS_HOSTNAME=localhost ROS_LOG_DIR=/tmp/roslog OMP_NUM_THREADS=1 && \
     export PYTHONPATH=/opt/ros/noetic/lib/python3.8/site-packages:\$LOCOSIM_DIR/robot_control:\$LOCOSIM_DIR/robot_control/base_controllers:\$PYTHONPATH && \
     python3 -u saferl_controller_dataset.py --episodes $episodes --seed $seed --out $out/batch$i --no-rviz $* \
       > $out/batch$i.log 2>&1; chown -R $(id -u):$(id -g) $out"
  echo "batch $i: seed $seed, $out/batch$i"
done
