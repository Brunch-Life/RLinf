#!/bin/bash
# Start Polymetis + GELLO ZMQ bridge on the current machine.
#
# Run this on EACH Franka controller node BEFORE launching RLinf.
# It starts 3 background services:
#   1. Polymetis robot server  (gRPC :50051, kHz joint control via libfranka)
#   2. Polymetis gripper server (gRPC :50052, Robotiq 2F)
#   3. GELLO ZMQ bridge        (ZMQ  :$ZMQ_PORT, exposes joint read/write)
#
# Usage:
#   bash start_polymetis.sh                           # all defaults
#   bash start_polymetis.sh --zmq-port 5555           # custom ZMQ port
#   bash start_polymetis.sh --gripper franka_hand     # use Franka hand
#   bash start_polymetis.sh --stop                    # kill all services
#
# Environment variables (override defaults):
#   ROBOT_IP            Franka robot IP          (default: 172.16.0.2)
#   ZMQ_PORT            ZMQ bridge port          (default: 5555)
#   GRIPPER_TYPE        robotiq_2f / franka_hand (default: robotiq_2f)
#   CONDA_ENV           conda env name           (default: polymetis-local)
#   CONDA_PATH          miniconda root           (default: ~/miniconda3)
#   GELLO_SOFTWARE_PATH gello_software dir       (default: ~/gello_software)

set -e

ROBOT_IP="${ROBOT_IP:-172.16.0.2}"
ZMQ_PORT="${ZMQ_PORT:-5555}"
GRIPPER_TYPE="${GRIPPER_TYPE:-robotiq_2f}"
CONDA_ENV="${CONDA_ENV:-polymetis-local}"
CONDA_PATH="${CONDA_PATH:-$HOME/miniconda3}"
GELLO_SOFTWARE_PATH="${GELLO_SOFTWARE_PATH:-$HOME/gello_software}"
LOG_DIR="/tmp/polymetis_logs"
STOP_ONLY=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --robot-ip)  ROBOT_IP="$2"; shift 2 ;;
        --zmq-port)  ZMQ_PORT="$2"; shift 2 ;;
        --gripper)   GRIPPER_TYPE="$2"; shift 2 ;;
        --conda-env) CONDA_ENV="$2"; shift 2 ;;
        --stop)      STOP_ONLY=1; shift ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

kill_existing() {
    echo "[*] Killing existing Polymetis / ZMQ processes..."
    pkill -f "run_server" 2>/dev/null || true
    pkill -f "launch_robot.py" 2>/dev/null || true
    pkill -f "launch_gripper.py" 2>/dev/null || true
    pkill -f "launch_nodes.py" 2>/dev/null || true
    sleep 1
}

if [ "$STOP_ONLY" -eq 1 ]; then
    kill_existing
    echo "[OK] All services stopped."
    exit 0
fi

mkdir -p "$LOG_DIR"
CONDA_SH="$CONDA_PATH/etc/profile.d/conda.sh"
if [ ! -f "$CONDA_SH" ]; then
    echo "ERROR: conda not found at $CONDA_PATH"
    exit 1
fi
if [ ! -d "$GELLO_SOFTWARE_PATH/experiments" ]; then
    echo "ERROR: gello_software not found at $GELLO_SOFTWARE_PATH"
    exit 1
fi

ACTIVATE="source $CONDA_SH && conda activate $CONDA_ENV"

kill_existing

echo "============================================"
echo "  Polymetis + GELLO ZMQ Bridge"
echo "============================================"
echo "  Robot IP:    $ROBOT_IP"
echo "  ZMQ Port:    $ZMQ_PORT"
echo "  Gripper:     $GRIPPER_TYPE"
echo "  Conda Env:   $CONDA_ENV"
echo "  GELLO Path:  $GELLO_SOFTWARE_PATH"
echo "============================================"

echo "[1/3] Starting Polymetis robot server (gRPC:50051)..."
bash -c "$ACTIVATE && launch_robot.py robot_client=franka_hardware" \
    > "$LOG_DIR/robot_server.log" 2>&1 &
PID_ROBOT=$!
sleep 3
if ! kill -0 $PID_ROBOT 2>/dev/null; then
    echo "FAILED — check $LOG_DIR/robot_server.log:"
    tail -10 "$LOG_DIR/robot_server.log"
    exit 1
fi
echo "  OK  (pid=$PID_ROBOT)"

echo "[2/3] Starting Polymetis gripper server (gRPC:50052)..."
bash -c "$ACTIVATE && launch_gripper.py gripper=$GRIPPER_TYPE" \
    > "$LOG_DIR/gripper_server.log" 2>&1 &
PID_GRIPPER=$!
sleep 2
if ! kill -0 $PID_GRIPPER 2>/dev/null; then
    echo "FAILED — check $LOG_DIR/gripper_server.log:"
    tail -10 "$LOG_DIR/gripper_server.log"
    exit 1
fi
echo "  OK  (pid=$PID_GRIPPER)"

echo "[3/3] Starting GELLO ZMQ bridge (ZMQ:$ZMQ_PORT)..."
bash -c "$ACTIVATE && cd $GELLO_SOFTWARE_PATH/experiments && python launch_nodes.py --robot=fr3 --robot_ip=127.0.0.1 --robot_port=$ZMQ_PORT --hostname=0.0.0.0" \
    > "$LOG_DIR/zmq_bridge.log" 2>&1 &
PID_ZMQ=$!
sleep 3
if ! kill -0 $PID_ZMQ 2>/dev/null; then
    echo "FAILED — check $LOG_DIR/zmq_bridge.log:"
    tail -10 "$LOG_DIR/zmq_bridge.log"
    exit 1
fi
echo "  OK  (pid=$PID_ZMQ)"

cat > "$LOG_DIR/pids" <<EOF
ROBOT=$PID_ROBOT
GRIPPER=$PID_GRIPPER
ZMQ=$PID_ZMQ
EOF

echo ""
echo "============================================"
echo "  All 3 services running!"
echo "  Robot:   pid=$PID_ROBOT"
echo "  Gripper: pid=$PID_GRIPPER"
echo "  ZMQ:     pid=$PID_ZMQ"
echo "  Logs:    $LOG_DIR/"
echo "============================================"
echo ""
echo "To stop:  bash $0 --stop"
