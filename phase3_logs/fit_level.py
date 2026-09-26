import numpy as np, csv, sys
from numpy.polynomial import polynomial as P
r=list(csv.DictReader(open(sys.argv[1])))
g=lambda k,f=float: np.array([f(q[k]) for q in r])
t,x,y,al,be,fd,ep,a0,a1,d=g("t"),g("xb"),g("yb"),np.degrees(g("alpha")),np.degrees(g("beta")),g("ball_found",int),g("episode",int),g("a0"),g("a1"),g("dist")
W=4; ax=np.full(len(t),np.nan); ay=ax.copy()
for i in range(W,len(t)-W):
    s=slice(i-W,i+W+1)
    if fd[s].min()==0 or ep[s].min()!=ep[s].max(): continue
    tt=t[s]-t[i]; ax[i]=2*np.polyfit(tt,x[s],2)[0]; ay[i]=2*np.polyfit(tt,y[s],2)[0]
ok=np.isfinite(ax)&(np.abs(x)<0.09)&(np.abs(y)<0.075)
px=np.polyfit(be[ok],ax[ok],1); py=np.polyfit(al[ok],ay[ok],1)
h=ok.nonzero()[0]; mid=h[len(h)//2]
def b0(m): p=np.polyfit(be[m],ax[m],1); return -p[1]/p[0]
def a0f(m): p=np.polyfit(al[m],ay[m],1); return -p[1]/p[0]
m1=ok&(np.arange(len(t))<mid); m2=ok&(np.arange(len(t))>=mid)
print(f"n={ok.sum()} beta0={-px[1]/px[0]:+.2f} (halves {b0(m1):+.2f}/{b0(m2):+.2f}, r={np.corrcoef(be[ok],ax[ok])[0,1]:+.2f})  alpha0={-py[1]/py[0]:+.2f} (halves {a0f(m1):+.2f}/{a0f(m2):+.2f}, r={np.corrcoef(al[ok],ay[ok])[0,1]:+.2f})")
near=fd.astype(bool)&(d<0.07)
print(f"near-goal mean err ex={1000*(x[near]+0.0078).mean():+.1f}mm ey={1000*(y[near]-0.0060).mean():+.1f}mm  sd {1000*x[near].std():.1f}/{1000*y[near].std():.1f}mm  frac near={near.mean():.2f}  frac |a0|>=0.79 {np.mean(np.abs(a0[near])>=0.79):.2f}")
