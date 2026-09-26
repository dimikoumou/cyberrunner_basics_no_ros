#!/bin/bash
# Keeps the ball balanced until stopped: relaunches pd_balance.py if it exits
# (e.g. U2D2 dropout). Stop with:  touch ~/cyberrunner_basics_no_ros/phase3_logs/STOP
# (the current run then ends when it next exits) or: pkill -f keep_balancing.sh; pkill -f pd_balance.py
cd "$(dirname "$0")/../rl_hw"
rm -f ../phase3_logs/STOP
while [ ! -f ../phase3_logs/STOP ]; do
    while ! ls /dev/tty.usbserial-* >/dev/null 2>&1; do sleep 2; done
    echo "=== launch $(date) ==="
    PD_SNAPSHOT_S=30 ../.venv/bin/python3 -u pd_balance.py 2000000 100000 0 2000000
    echo "=== pd_balance exited $(date) code $? ==="
    sleep 5
done
