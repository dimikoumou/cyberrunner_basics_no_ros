"""Tilt-servo tracking: lag-aligned std of (measured - target) tilt per axis."""
import csv, sys, numpy as np
for fn in sys.argv[1:]:
    r = list(csv.DictReader(open(fn)))
    g = lambda k: np.array([float(q[k]) if q.get(k) not in (None, "") else np.nan for q in r])
    a0, a1 = np.clip(g("a0"), -1, 1), np.clip(g("a1"), -1, 1)
    al, be = np.degrees(g("alpha")), np.degrees(g("beta"))
    tb, ta = a0 * 5.0, -a1 * 5.0          # target minus the constant level offset
    out = []
    for name, tgt, meas in (("alpha", ta, al), ("beta", tb, be)):
        best = None
        for lag in range(0, 9):
            e = meas[lag:] - tgt[:len(tgt) - lag]
            e = e - np.nanmean(e)
            s = np.nanstd(e)
            if best is None or s < best[1]: best = (lag, s)
        out.append(f"{name}: lag {best[0]} steps, tracking err std {best[1]:.2f} deg")
    print(fn.split('/')[-1] + ": " + " | ".join(out))
