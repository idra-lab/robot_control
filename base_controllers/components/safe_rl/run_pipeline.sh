#!/bin/bash
# @author: riccardobussola
# The value function study for one or more robots, from the host (README.md, "Run it all"): the ABS
# baseline (abs, ABS's reach-avoid value of the walking policy) and the paper's value function (mps, the
# safe policy's chance of reaching the invariant set) from the same pipeline, then the safety filter in
# closed loop with each.  Stages, in order:
#   data        fall-rule check (8 x FALL_EPISODES run to the ground); per method 6 batches of EPISODES
#               for training (the 6th stops it) and 2 for the test; mps_early, the ablation of the
#               handover time (the safe policy taking over from the push start, --switch-from push_start
#               --switch-window 0 0.7)
#   calib-data  the walking policy alone, every state at 100 Hz: 8 x THR_EPISODES runs of 20 s with the
#               evaluation's kicks (the threshold), 8 x MARGIN_EPISODES with light kicks (the margin)
#   train       MODELS/VF_<name>_<robot>.pkl and .onnx, unless the .onnx exists (delete it to retrain):
#               abs (ABS's reach-avoid training), mps (static equilibrium), mps_early (its dataset);
#               aliengo also mps_polygon (X_inv: capture point in the support polygon)
#   calibrate   threshold, margin and switching count of each model (calibrate_filter.py; abs: ABS's own
#               rule, mps: the walking rule over 3 evaluations), into VF_<name>_<robot>.json
#   evaluate    each model on its method's test batches (evaluate_value_function.py)
#   loop        closed loop, the same 8 x LOOP_EPISODES pushes (the evaluation's kicks) per arm, every
#               episode LOOP_LENGTH s long unless the robot leaves the safe set, the safe policy kept once
#               switched (the paper's simulations): the walking policy alone, the safe policy alone (from
#               the push start), abs at ABS's rule and at the walking rule (abs_walk), mps at its .json
#               and at the paper's percentile threshold (mps_pct), without the margin (mps_L0), switching
#               on a single value (mps_k1), switching back after 2 s (abs_back, mps_back), mps_early;
#               aliengo also mps_polygon
#   summary     summarize.sh: figures and numbers in FIGURES/
#
#   run_pipeline.sh [robot ...]        default: aliengo go2;  STAGES="train calibrate" for some stages
#
# A stage skips the batches already complete, so an interrupted run resumes; delete a folder to redo
# it.  About 3 h for the Aliengo (with the ablations) and 2.5 h for the Go2 on 24 cores, 8 simulations
# at a time in real time.  Seeds per robot: SEED (aliengo 1000, go2 2000) + 0 fall check, 100 abs, 900
# mps, 200 mps_early, 300 threshold runs, 400 margin runs, 500 closed loop.  Outputs, relative to
# robot_control/base_controllers: OUT (components/safe_rl/value_function/observation_datasets)/<robot>/,
# MODELS (.../value_function/models), FIGURES (components/safe_rl/report_figures).
set -e
SR_HOST=$(dirname "$(readlink -f "$0")")
BC_HOST=$(readlink -f "$SR_HOST/../..")
SR=components/safe_rl
VF=$SR/value_function
export OUT=${OUT:-$VF/observation_datasets}
export MODELS=${MODELS:-$VF/models}
export FIGURES=${FIGURES:-$SR/report_figures}
FALL_EPISODES=${FALL_EPISODES:-30}; EPISODES=${EPISODES:-70}; THR_EPISODES=${THR_EPISODES:-10}
MARGIN_EPISODES=${MARGIN_EPISODES:-30}; LOOP_EPISODES=${LOOP_EPISODES:-30}; LOOP_LENGTH=${LOOP_LENGTH:-7}
EPOCHS=${EPOCHS:-600}
STAGES=${STAGES:-"data calib-data train calibrate evaluate loop summary"}
ROBOTS=${*:-aliengo go2}
stamp() { echo "$(date +%T) $*"; }
stage() { [[ " $STAGES " == *" $1 "* ]]; }
wait_batches() { sleep 20; while [ -n "$(docker ps -q --filter name=safe_rl_)" ]; do sleep 10; done; }
# batches <first seed> <n> <episodes> <out> [args]: the batches of <out> not complete yet, then wait
batches() {
  local first=$1 n=$2 episodes=$3 out=$4; shift 4
  local done=1
  for i in $(seq 1 "$n"); do [ -f "$BC_HOST/$out/batch$i/observations_dataset_abs_test_episodes.csv" ] || done=0; done
  if [ $done = 1 ]; then echo "  $out: complete"; return; fi
  rm -rf "$BC_HOST/$out"
  "$SR_HOST/run_batches.sh" "$first" "$n" "$episodes" "$out" "$@" > /dev/null
  wait_batches
  echo "  $out: $(cat "$BC_HOST/$out"/batch*/observations_dataset_abs_test_episodes.csv | grep -vc episode) episodes"
}
npy() { (cd "$BC_HOST" && ls $1/batch*/observations_dataset_abs_test.npy); }
in_image() { "$SR_HOST/run_in_image.sh" "$@"; }
json() { python3 -c "import json,sys; d=json.load(open('$BC_HOST/$1')); print(eval('d' + sys.argv[1]))" "$2"; }

# CONCURRENT=1: another run_pipeline.sh (another robot) is running alongside; the two then wait on
# each other's batches, so two arms run at a time
if [ -z "$CONCURRENT" ] && [ -n "$(docker ps -q --filter name=safe_rl_)" ]; then
  echo "safe_rl_ containers are running (docker ps): wait for them, stop them, or set CONCURRENT=1"; exit 1
fi
for robot in $ROBOTS; do
  case $robot in
    aliengo) seed=${SEED:-1000}; kicks="1.2 1.4 1.6 1.8 2.0"; light=1.0; ablations=1;;
    go2) seed=${SEED:-2000}; kicks="1.0 1.2 1.4 1.6 1.8"; light=0.75; ablations=0;;
    *) seed=${SEED:?set SEED for $robot}; kicks=${KICKS:?set KICKS for $robot}; light=${LIGHT:-1.0}; ablations=0;;
  esac
  light_kicks=$(python3 -c "import numpy as np; print(' '.join(f'{k:.3f}' for k in np.linspace(0.05, $light, 12)))")
  D=$OUT/$robot
  R="--robot $robot"
  models="abs mps mps_early"
  [ $ablations = 1 ] && models="$models mps_polygon"

  if stage data; then
    stamp "$robot: data"
    batches $seed 8 $FALL_EPISODES $D/fall_check $R --fall-check
    batches $((seed + 100)) 6 $EPISODES $D/abs/episodes $R --method abs
    batches $((seed + 106)) 2 $EPISODES $D/abs/test $R --method abs
    batches $((seed + 900)) 6 $EPISODES $D/mps/episodes $R --method mps
    batches $((seed + 906)) 2 $EPISODES $D/mps/test $R --method mps
    batches $((seed + 200)) 6 $EPISODES $D/mps_early/episodes $R --method mps --switch-from push_start --switch-window 0 0.7
  fi

  if stage calib-data; then
    stamp "$robot: calibration runs"
    batches $((seed + 300)) 8 $THR_EPISODES $D/calibration/threshold $R --record-all --sample-rate 100 \
      --episode-length 20 --push-window 2 17 --kicks $kicks
    batches $((seed + 400)) 8 $MARGIN_EPISODES $D/calibration/margin $R --record-all --sample-rate 100 --kicks $light_kicks
  fi

  if stage train; then
    stamp "$robot: training $models"
    for name in $models; do
      case $name in
        abs) data=$D/abs; opts="--method abs";;
        mps) data=$D/mps; opts="--method mps";;
        mps_early) data=$D/mps_early; opts="--method mps";;
        mps_polygon) data=$D/mps; opts="--method mps --x-inv polygon";;
      esac
      [ -f "$BC_HOST/$MODELS/VF_${name}_$robot.onnx" ] && continue
      # three trainings at a time, 4 cores each
      in_image python3 $VF/train_value_function.py $opts --data $(npy $data/episodes | grep -v batch6) \
        --val $(npy $data/episodes | grep batch6) --model $MODELS/VF_${name}_$robot.pkl --epochs $EPOCHS \
        > "$BC_HOST/$D/train_$name.log" 2>&1 &
      [ $(jobs -r | wc -l) -ge 3 ] && wait -n
    done
    wait
    for name in $models; do echo "  $name: $(tr '\r' '\n' < "$BC_HOST/$D/train_$name.log" | grep -a "best at epoch" | tail -1)"; done
  fi

  if stage calibrate; then
    stamp "$robot: calibration"
    for name in $models; do
      in_image python3 $VF/calibrate_filter.py --model $MODELS/VF_${name}_$robot.onnx --write \
        --threshold-data $(npy $D/calibration/threshold) --margin-data $(npy $D/calibration/margin) \
        > "$BC_HOST/$D/calibrate_$name.log" 2>&1
      echo "  $name: threshold $(json $MODELS/VF_${name}_$robot.json "['threshold']"), margin $(json $MODELS/VF_${name}_$robot.json "['margin']")"
    done
  fi

  if stage evaluate; then
    stamp "$robot: evaluation"
    mkdir -p "$BC_HOST/$FIGURES"
    for name in $models; do
      method=${name%%_*}  # mps_* on the mps test batches
      levels=$([ $method = abs ] && echo "0.05 0.2 0.5" || echo "0.5 0.8 0.9")
      in_image python3 $VF/evaluate_value_function.py --model $MODELS/VF_${name}_$robot.onnx \
        --data $(npy $D/$method/test) --light-kick $light --thresholds $levels --plot $FIGURES/${robot}_vf_$name.png \
        > "$BC_HOST/$D/evaluate_$name.log" 2>&1
    done
  fi

  if stage loop; then
    stamp "$robot: closed loop"
    loop() {
      local name=$1; shift
      batches $((seed + 500)) 8 $LOOP_EPISODES $D/closed_loop/$name $R --kicks $kicks --episode-length $LOOP_LENGTH "$@"
    }
    filter() { echo "--filter $MODELS/VF_$1_$robot.onnx --no-switch-back"; }
    setting() { json $MODELS/VF_$1_$robot.json "$2"; }
    pct() { setting $1 "['rule_thresholds']['percentile']"; }
    loop alone
    loop safe --backup-delay 0
    for name in $models; do loop $name $(filter $name); done
    loop abs_walk $(filter abs) --threshold $(setting abs "['rule_thresholds']['walking']") --min-switch 3
    loop mps_pct $(filter mps) --threshold $(pct mps)
    loop mps_L0 $(filter mps) --threshold $(setting mps "['threshold']") --margin 0
    loop mps_k1 $(filter mps) --threshold $(setting mps "['rule_thresholds']['walking_k1']") \
      --margin $(setting mps "['margin_k1']") --min-switch 1
    for name in abs mps; do loop ${name}_back --filter $MODELS/VF_${name}_$robot.onnx; done
  fi

  if stage summary; then
    stamp "$robot: summary"
    in_image bash $SR/summarize.sh $robot > /dev/null
    stamp "$robot: done, $FIGURES/${robot}_summary.txt"
  fi
done
