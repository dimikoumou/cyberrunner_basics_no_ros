"""Smoothness metrics for a pd_balance run CSV: how directly the ball reaches the
goal region and how quietly it stays there."""
import csv, sys, numpy as np
for fn in sys.argv[1:]:
    r = list(csv.DictReader(open(fn)))
    g = lambda k: np.array([float(q[k]) if q.get(k) not in (None, "") else np.nan for q in r])
    t, d, ic, fd, k, vx, vy = g("t"), g("dist"), g("in_circle"), g("ball_found"), g("kicking"), g("vx"), g("vy")
    rad = np.nanmedian(g("goal_r")) if "goal_r" in r[0] else np.nan
    first = np.flatnonzero(ic == 1)
    t_arrive = t[first[0]] - t[0] if len(first) else np.nan
    # exits: detected frames outside after having been inside
    exits, inside = 0, False
    for i in range(len(t)):
        if ic[i] == 1: inside = True
        elif fd[i] == 1 and inside: exits += 1; inside = False
    m = t > (t[0] + (t_arrive if np.isfinite(t_arrive) else 0))
    dur = max(1e-6, t[-1] - t[m][0]) if m.any() else np.nan
    km = k[m]; kicks = int(((km[:-1] == 0) & (km[1:] >= 1)).sum()) if m.any() else 0
    sp = np.hypot(vx, vy)
    print(f"{fn.split('/')[-1]}: goal r={1000*rad:.1f}mm | first arrival {t_arrive:.1f}s | after arrival: in region {100*np.nanmean(ic[m]):.0f}%, "
          f"exits {exits} ({exits/dur*60:.1f}/min), kicks {kicks/dur*60:.1f}/min, median dist {1000*np.nanmedian(d[m]):.1f}mm, "
          f"p90 {1000*np.nanpercentile(d[m],90):.1f}mm, still {100*np.mean(sp[m]<0.01):.0f}%")
