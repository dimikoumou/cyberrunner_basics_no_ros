#!/bin/bash
# Smooth-motion test: park the ball at START, then send it to TARGET (virtual goals),
# and measure how many motions/hops it takes. Usage: move_test.sh "sx,sy:tx,ty,r" ...
cd "$(dirname "$0")/../rl_hw"
for pair in "$@"; do
    start="${pair%%:*}"; target="${pair##*:}"
    PD_GOAL="$start,0.015" ../.venv/bin/python3 -u pd_balance.py 500 5 0 500 > /dev/null 2>&1
    PD_ODIL="$ODIL" PD_HYBRID="${HYBRID:-1}" PD_POLICY="$POLICY" PD_GOAL="$target" ../.venv/bin/python3 -u pd_balance.py 800 5 0 800 > /tmp/move_run.log 2>&1; grep -E "  move |ended early|implausible ball jump" /tmp/move_run.log | head -6
    F=$(ls -t pd_logs/*.csv | head -1)
    ../.venv/bin/python3 - "$F" "$pair" /tmp/move_run.log <<'PY'
import csv, sys, numpy as np, re
r=list(csv.DictReader(open(sys.argv[1]))); g=lambda k: np.array([float(q[k]) if q[k] else np.nan for q in r])
t,d,vx,vy,ph,fd=g("t"),g("dist"),g("vx"),g("vy"),g("kicking"),g("ball_found"); t=t-t[0]
R=float(sys.argv[2].split(":")[1].split(",")[2]); sp=np.hypot(vx,vy)
log=open(sys.argv[3]).read(); moves=len(re.findall(r"  move \d+:", log)); early=len(re.findall(r"ended early", log))
inside=np.flatnonzero((d<R)&(fd==1)); ta=t[inside[0]] if len(inside) else np.nan
# hops before arrival: separate still periods (>=0.3 s, speed<10mm/s) while outside the target
pre=(t<ta) if np.isfinite(ta) else np.ones(len(t),bool)
still=(sp<0.01)&pre&(d>R); hops=0; run=0.0
for i in range(1,len(t)):
    run = run + (t[i]-t[i-1]) if still[i] else 0.0
    if still[i] and run>=0.3 and not (still[i-1] and run-(t[i]-t[i-1])>=0.3): hops+=1
pulses=int(((ph[:-1]==0)&(ph[1:]==1)).sum())
a0,a1=g("a0"),g("a1"); jerk=float(np.nanmean(np.diff(np.clip(a0,-1,1))**2+np.diff(np.clip(a1,-1,1))**2))
post=t>ta if np.isfinite(ta) else np.zeros(len(t),bool)
print(f"{sys.argv[2]}: start {1000*d[0]:.0f}mm | moves {moves} (early end {early}), hops before arrival {hops}, pulses {pulses} | "
      f"arrived {ta:.1f}s | after: max {1000*np.nanmax(d[post]) if post.any() else float('nan'):.0f}mm, "
      f"in target {100*np.nanmean(d[post]<R) if post.any() else 0:.0f}%, final {1000*np.nanmedian(d[t>t[-1]-5]):.1f}mm | jerk {jerk:.4f}")
PY
done
