import cv2, csv
rows=list(csv.DictReader(open("state_est/markers.csv")))
# pick the rig camera: the first one that really delivers 1920x1080 (indices move on re-plug)
for idx in range(4):
    cap=cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION); cap.set(3,1920); cap.set(4,1080)
    ok,f=cap.read()
    if ok and f is not None and f.shape[1]==1920: break
    cap.release()
cv2.namedWindow("CyberRunner live (q to quit)", cv2.WINDOW_NORMAL); cv2.resizeWindow("CyberRunner live (q to quit)", 1280, 720)
while True:
    ok,f=cap.read()
    if not ok: continue
    for r in rows:
        col=(0,255,0) if r["corner_type"]=="outer" else (0,140,255)
        cv2.circle(f,(int(r["x"]),int(r["y"])),22,col,2)
    cv2.imshow("CyberRunner live (q to quit)", f)
    if cv2.waitKey(1)&0xFF==ord('q'): break
