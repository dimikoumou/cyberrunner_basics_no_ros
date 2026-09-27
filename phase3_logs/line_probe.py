import json, time, urllib.request, math, sys, numpy as np
def post(d): urllib.request.urlopen(urllib.request.Request("http://localhost:8000/cmd", data=json.dumps(d).encode(), method="POST"))
post(json.loads(sys.argv[1])); dur=float(sys.argv[2])
t0=time.time(); rows=[]
while time.time()-t0<dur:
    s=json.load(urllib.request.urlopen("http://localhost:8000/state"))
    if s["ball"]: rows.append((time.time()-t0, s["ball"][0], s["ball"][1], s["dist"] if s["dist"] is not None else np.nan, s.get("line_msg","")))
    time.sleep(0.2)
a=np.array([r[:4] for r in rows]); t,x,y,off=a.T
ang=np.unwrap(np.arctan2(y,x)); sp=np.hypot(np.diff(x),np.diff(y))/np.diff(t)
still=np.mean(sp<0.005)
print(f"{sys.argv[1]}: {len(t)} samples; loop progress {np.degrees(ang[-1]-ang[0]):.0f} deg ({(ang[-1]-ang[0])/(2*np.pi):.2f} laps); "
      f"median speed {100*np.median(sp):.1f} cm/s; still {100*still:.0f}% of samples; off-line median {1000*np.nanmedian(off):.1f} mm p90 {1000*np.nanpercentile(off,90):.1f} mm; last msg: {rows[-1][4]}")
