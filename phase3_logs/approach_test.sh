#!/bin/bash
# Approach test: park the ball far away with a virtual goal, then run the real
# (sheet-detected) goal and report how smoothly the ball gets there.
cd "$(dirname "$0")/../rl_hw"
for start in "$@"; do
    echo "=== start at $start ==="
    PD_GOAL="$start,0.015" ../.venv/bin/python3 -u pd_balance.py 700 5 0 700 > /dev/null 2>&1
    ../.venv/bin/python3 -u pd_balance.py 2700 5 0 2700 2>&1 | grep -E "goal circle|longest|Traceback" 
    F=$(ls -t pd_logs/*.csv | head -1)
    ../.venv/bin/python3 ../phase3_logs/eval_smooth.py $F
    ../.venv/bin/python3 - "$F" <<'PY'
import csv, sys, numpy as np
r=list(csv.DictReader(open(sys.argv[1]))); g=lambda k: np.array([float(q[k]) if q[k] else np.nan for q in r])
t,d,x,y,gr=g("t"),g("dist"),g("xb"),g("yb"),g("goal_r")
R=np.nanmedian(gr); inside=np.flatnonzero(d<R)
print(f"   start dist {1000*d[0]:.0f} mm; reached region at {t[inside[0]]-t[0]:.1f}s" if len(inside) else "   never reached")
if len(inside):
    after=d[inside[0]:]; print(f"   after first arrival: max overshoot {1000*np.nanmax(after):.1f} mm, settled (median last 30s) {1000*np.nanmedian(d[t>t[-1]-30]):.1f} mm")
PY
done
