import csv, sys, numpy as np
for fn in sys.argv[1:]:
    r=list(csv.DictReader(open(fn)))
    t=np.array([float(q["t"]) for q in r]); ic=np.array([int(q["in_circle"]) for q in r]); fd=np.array([int(q.get("ball_found",1)) for q in r])
    d=np.array([float(q["dist"]) for q in r]); ep=np.array([int(q["episode"]) for q in r]); st=[q["status"] for q in r]
    best=0; s=None
    for i in range(len(t)):
        if ic[i]==1:
            if s is None or ep[i]!=ep[s]: s=i
            best=max(best,t[i]-t[s])
        elif fd[i]==1 or st[i]=="ball_lost": s=None
    f=fd==1; late=f&(t>t[0]+20)
    print(f"{fn.split('/')[-1]}: {t[-1]-t[0]:.0f}s, in-circle {100*ic.mean():.1f}% | after first 20s {100*ic[t>t[0]+20].mean():.1f}% | median dist {1000*np.median(d[late]):.1f}mm p90 {1000*np.percentile(d[late],90):.1f}mm | longest hold {best:.2f}s | ball_lost {st.count('ball_lost')} | episodes {ep.max()}")
