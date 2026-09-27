"""
Local web UI for the ball-on-plate controller (2026-09-26).

Open http://localhost:8000 while pd_balance.py runs with PD_UI=1:
  - live camera view with the ball, the target and the target region drawn on it
  - Start / Stop balancing
  - mode "sheet": the ball goes to the red region drawn on the sheet (re-detected
    live, so sheets can be swapped mid-run)
  - mode "click": click anywhere on the plate in the live view and the ball goes there

The controller loop (pd_balance.py) owns the rig; this module only shares frames and
state with the browser and queues the browser's commands for the loop to apply.
Standard library only (http.server), so there is nothing extra to install.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

STREAM_FPS = 15
FRAME_W, FRAME_H = 640, 360

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CyberRunner control</title>
<style>
  :root { --bg:#f6f5f2; --panel:#ffffff; --ink:#1d1d1f; --muted:#6b6b70; --line:#dcdad4;
          --accent:#c8322d; --go:#2f7d4f; --stop:#9a2b27; }
  @media (prefers-color-scheme: dark) { :root { --bg:#161617; --panel:#202022; --ink:#ececee;
          --muted:#9c9ca3; --line:#34343a; --accent:#ef5a53; --go:#4fae78; --stop:#e0625c; } }
  * { box-sizing:border-box; }
  body { margin:0; font:15px/1.45 -apple-system, system-ui, sans-serif; background:var(--bg); color:var(--ink); }
  main { max-width:1100px; margin:0 auto; padding:20px 16px 32px; }
  h1 { font-size:20px; margin:0 0 4px; } .sub { color:var(--muted); margin:0 0 16px; }
  .grid { display:grid; grid-template-columns: minmax(0,1fr) 300px; gap:16px; align-items:start; }
  @media (max-width: 820px) { .grid { grid-template-columns: 1fr; } }
  .view { position:relative; background:#000; border-radius:10px; overflow:hidden; border:1px solid var(--line); }
  .view img { display:block; width:100%; height:auto; cursor:crosshair; }
  .hint { position:absolute; left:10px; bottom:10px; background:rgba(0,0,0,.55); color:#fff;
          padding:4px 8px; border-radius:6px; font-size:13px; }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; }
  .panel + .panel { margin-top:12px; }
  .panel h2 { font-size:13px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin:0 0 10px; }
  .row { display:flex; gap:8px; flex-wrap:wrap; }
  button { font:inherit; padding:9px 14px; border-radius:8px; border:1px solid var(--line);
           background:var(--panel); color:var(--ink); cursor:pointer; flex:1; }
  button.go { background:var(--go); border-color:var(--go); color:#fff; }
  button.stop { background:var(--stop); border-color:var(--stop); color:#fff; }
  button.on { outline:2px solid var(--accent); outline-offset:-2px; font-weight:600; }
  dl { display:grid; grid-template-columns:auto 1fr; gap:4px 12px; margin:0; }
  dt { color:var(--muted); } dd { margin:0; font-variant-numeric:tabular-nums; }
  .ok { color:var(--go); font-weight:600; } .bad { color:var(--stop); font-weight:600; }
  .msg { color:var(--muted); font-size:13px; min-height:1.2em; margin-top:8px; }
</style></head>
<body><main>
  <h1>CyberRunner</h1>
  <p class="sub">Ball-on-plate control. Pick a mode, press Start, and click the board in click mode.</p>
  <div class="grid">
    <div class="view">
      <img id="cam" alt="Live view of the plate">
      <div class="hint" id="hint">Loading…</div>
    </div>
    <div>
      <div class="panel">
        <h2>Balancing</h2>
        <div class="row">
          <button class="go" id="start">Start</button>
          <button class="stop" id="stop">Stop</button>
        </div>
        <div class="msg" id="msg"></div>
      </div>
      <div class="panel">
        <h2>Target</h2>
        <div class="row">
          <button id="m_sheet">Red region on sheet</button>
          <button id="m_click">Click to target</button>
          <button id="m_line">Follow red line</button>
          <button id="m_path">Draw path</button>
        </div>
        <div class="row" id="pathrow" style="margin-top:8px">
          <button id="p_go">Go</button>
          <button id="p_clear">Clear path</button>
        </div>
      </div>
      <div class="panel">
        <h2>Controller</h2>
        <div class="row">
          <button id="c_classic">Classic</button>
          <button id="c_learned">Learned (RL)</button>
          <button id="c_odil">ODIL</button>
        </div>
        <label style="display:block;margin-top:8px;font-size:13px;color:var(--muted)">
          <input type="checkbox" id="c_hybrid" checked> classic settle near the target (hybrid)
        </label>
      </div>
      <div class="panel">
        <h2>Hole &amp; reload</h2>
        <div class="row">
          <button id="h_drop">Drop into hole</button>
          <button id="h_drop5">Drop test &times;5</button>
          <button id="h_detect">Find holes</button>
        </div>
        <label style="display:block;margin-top:8px;font-size:13px;color:var(--muted)">
          <input type="checkbox" id="h_reload" checked> auto-reload with the elevator when the ball falls in
        </label>
        <div class="msg" id="h_status">–</div>
      </div>
      <div class="panel">
        <h2>Ball elevator</h2>
        <div class="row">
          <button class="go" id="e_on">On</button>
          <button class="stop" id="e_off">Off</button>
          <button id="e_dir">Forward</button>
          <button id="e_max">Max</button>
        </div>
        <label style="display:block;margin-top:10px;color:var(--muted);font-size:13px">
          Speed: <span id="e_speed_lbl">12 (2.7 rpm)</span>
          <input id="e_speed" type="range" min="0" max="1000" value="336" style="width:100%">
        </label>
        <div class="msg" id="e_status">off</div>
      </div>
      <div class="panel">
        <h2>Status</h2>
        <dl>
          <dt>Running</dt><dd id="s_run">–</dd>
          <dt>Mode</dt><dd id="s_mode">–</dd>
          <dt>Ball</dt><dd id="s_ball">–</dd>
          <dt>Target</dt><dd id="s_goal">–</dd>
          <dt>Distance</dt><dd id="s_dist">–</dd>
          <dt>In target</dt><dd id="s_in">–</dd>
          <dt>Hold</dt><dd id="s_hold">–</dd>
          <dt>Controller</dt><dd id="s_ctrl">–</dd>
          <dt>Loop</dt><dd id="s_hz">–</dd>
        </dl>
      </div>
    </div>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
async function cmd(body) {
  try {
    const r = await fetch('/cmd', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const j = await r.json(); $('msg').textContent = j.msg || '';
  } catch (e) { $('msg').textContent = 'Controller not reachable.'; }
}
$('start').onclick = () => cmd({cmd:'start'});
$('stop').onclick = () => cmd({cmd:'stop'});
$('m_sheet').onclick = () => cmd({cmd:'mode', mode:'sheet'});
$('m_click').onclick = () => cmd({cmd:'mode', mode:'click'});
$('m_line').onclick = () => cmd({cmd:'mode', mode:'line'});
$('m_path').onclick = () => cmd({cmd:'mode', mode:'path'});
$('p_go').onclick = () => cmd({cmd:'path_go'});
$('p_clear').onclick = () => cmd({cmd:'path_clear'});
let eDir = 1, eTimer = null, eMax = 1620;
// logarithmic slider: 0..1000 -> 1..eMax units (fine control at low speed, full speed at the right end)
const eUnits = () => Math.max(1, Math.round(Math.exp(Math.log(eMax) * parseInt($('e_speed').value, 10) / 1000)));
const eLabel = () => { $('e_speed_lbl').textContent = `${eUnits()} (${(eUnits() * 0.229).toFixed(1)} rpm)`; };
$('e_speed').addEventListener('input', () => {
  eLabel(); clearTimeout(eTimer);
  eTimer = setTimeout(() => cmd({cmd:'elevator', op:'speed', units:eUnits(), dir:eDir}), 150);
});
$('e_on').onclick = () => cmd({cmd:'elevator', op:'on', units:eUnits(), dir:eDir});
$('e_max').onclick = () => {
  $('e_speed').value = 1000; eLabel();
  cmd({cmd:'elevator', op:'speed', units:eMax, dir:eDir});
};
$('e_off').onclick = () => cmd({cmd:'elevator', op:'off'});
$('e_dir').onclick = () => {
  eDir = -eDir; $('e_dir').textContent = eDir > 0 ? 'Forward' : 'Reverse';
  cmd({cmd:'elevator', op:'speed', units:eUnits(), dir:eDir});
};
$('h_drop').onclick = () => cmd({cmd:'drop_test', n:1});
$('h_drop5').onclick = () => cmd({cmd:'drop_test', n:5});
$('h_detect').onclick = () => cmd({cmd:'holes_detect'});
$('h_reload').onchange = () => cmd({cmd:'auto_reload', on:$('h_reload').checked});
$('c_classic').onclick = () => cmd({cmd:'controller', which:'classic'});
$('c_learned').onclick = () => cmd({cmd:'controller', which:'learned'});
$('c_odil').onclick = () => cmd({cmd:'controller', which:'odil'});
$('c_hybrid').onchange = () => cmd({cmd:'hybrid', on:$('c_hybrid').checked});
$('cam').addEventListener('click', ev => {
  const r = ev.target.getBoundingClientRect();
  cmd({cmd:'click', u:(ev.clientX - r.left) / r.width, v:(ev.clientY - r.top) / r.height});
});
const mm = v => (v * 1000).toFixed(1) + ' mm';
function nextFrame() {
  const img = new Image();
  let done = false;
  const next = ok => { if (done) return; done = true; setTimeout(nextFrame, ok ? 80 : 500); };
  img.onload = () => { $('cam').src = img.src; next(true); };
  img.onerror = () => next(false);
  setTimeout(() => next(false), 3000);   // never let one hung request stop the view
  img.src = '/frame.jpg?t=' + Date.now();
}
nextFrame();
let eSynced = false;
async function poll() {
  try {
    const s = await (await fetch('/state')).json();
    $('s_run').innerHTML = s.running ? '<span class="ok">balancing</span>' : '<span class="bad">stopped</span>';
    $('s_mode').textContent = {click:'click to target', sheet:'red region on sheet', line:'follow red line', path:'drawn path', drop:'drop into hole'}[s.mode] || s.mode;
    $('m_sheet').classList.toggle('on', s.mode === 'sheet');
    $('m_click').classList.toggle('on', s.mode === 'click');
    $('m_line').classList.toggle('on', s.mode === 'line');
    $('m_path').classList.toggle('on', s.mode === 'path');
    $('pathrow').style.display = s.mode === 'path' ? 'flex' : 'none';
    $('c_classic').classList.toggle('on', s.controller !== 'learned');
    $('c_learned').classList.toggle('on', (s.controller || '').startsWith('learned'));
    $('c_odil').classList.toggle('on', (s.controller || '').startsWith('odil'));
    if (s.hybrid !== undefined) $('c_hybrid').checked = s.hybrid;
    $('s_ball').textContent = s.ball ? `(${mm(s.ball[0])}, ${mm(s.ball[1])})` : 'not visible';
    $('s_goal').textContent = s.goal ? `(${mm(s.goal[0])}, ${mm(s.goal[1])}), r ${mm(s.goal_r)}` : 'none';
    $('s_dist').textContent = s.dist != null ? mm(s.dist) : '–';
    $('s_in').innerHTML = s.in_target ? '<span class="ok">yes</span>' : 'no';
    $('s_hold').textContent = s.hold != null ? s.hold.toFixed(1) + ' s' : '–';
    $('s_hz').textContent = s.hz ? s.hz.toFixed(0) + ' Hz' : '–';
    if (s.elev_max_units && s.elev_max_units !== eMax) { eMax = s.elev_max_units; eLabel(); }
    if (!eSynced && s.elev_units) {
      eSynced = true;
      $('e_speed').value = Math.round(1000 * Math.log(Math.max(1, s.elev_units)) / Math.log(eMax));
      eDir = s.elev_dir || 1; $('e_dir').textContent = eDir > 0 ? 'Forward' : 'Reverse'; eLabel();
    }
    if (s.elev_on !== undefined) {
      const live = s.elev_vel_rpm != null ? `, actual ${s.elev_vel_rpm.toFixed(1)} rpm, ${s.elev_current_ma} mA` : '';
      $('e_status').innerHTML = s.elev_on ? `<span class="ok">running</span> at ${s.elev_rpm_set} rpm${live}`
        : (s.elev_msg && s.elev_msg !== 'stopped' ? s.elev_msg : 'off') + live;
    }
    if (s.auto_reload !== undefined) $('h_reload').checked = s.auto_reload;
    if (s.hole_msg !== undefined) $('h_status').textContent = s.hole_msg;
    $('s_ctrl').textContent = (s.controller || 'classic').endsWith('+classic settle') ? 'classic (near-field settle)'
      : s.controller === 'learned' ? 'learned RL' : s.controller === 'odil' ? 'ODIL' : 'classic';
    $('hint').textContent = s.mode === 'click' ? 'Click the board to send the ball there'
      : s.mode === 'path' ? (s.line_msg || 'Click points to draw a path, then press Go')
      : s.mode === 'line' ? (s.line_msg || 'Following the red line')
      : s.mode === 'drop' ? (s.hole_msg || 'Rolling the ball into the hole')
      : (s.goal ? 'Following the red region' : 'No red region found');
  } catch (e) { $('hint').textContent = 'Controller not reachable'; }
  setTimeout(poll, 250);
}
poll();
</script></body></html>
"""


class UIServer:
    def __init__(self, port=8000):
        self.port = port
        self._lock = threading.Lock()
        self._jpeg = None
        self._jpeg_seq = 0
        self._last_pub = 0.0
        self.state = {"running": False, "mode": "sheet", "ball": None, "goal": None, "goal_r": None,
                      "dist": None, "in_target": False, "hold": None, "hz": None, "controller": "classic"}
        self.overlay = {"contour_px": None, "goal_px": None, "goal_r_px": None, "ball_px": None,
                        "path_px": None, "path_closed": False, "holes_px": None}
        self._cmds = []
        ui = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                elif self.path.startswith("/state"):
                    with ui._lock:
                        body = json.dumps(ui.state).encode()
                    self._send(200, body, "application/json")
                elif self.path.startswith("/frame.jpg"):
                    # single latest frame: the page polls this (MJPEG <img> streams stall in
                    # Safari after a controller restart and never recover)
                    with ui._lock:
                        jpg = ui._jpeg
                    if jpg is None:
                        self._send(503, b"no frame yet", "text/plain")
                    else:
                        self.send_response(200)
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(jpg)))
                        self.send_header("Cache-Control", "no-store")
                        self.end_headers()
                        self.wfile.write(jpg)
                elif self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    seen = -1
                    try:
                        while True:
                            with ui._lock:
                                jpg, seq = ui._jpeg, ui._jpeg_seq
                            if jpg is None or seq == seen:
                                time.sleep(0.02)
                                continue
                            seen = seq
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        return
                else:
                    self._send(404, b"not found", "text/plain")

            def do_POST(self):
                if self.path != "/cmd":
                    self._send(404, b"not found", "text/plain")
                    return
                try:
                    n = int(self.headers.get("Content-Length", "0"))
                    c = json.loads(self.rfile.read(n) or b"{}")
                except (ValueError, json.JSONDecodeError):
                    self._send(400, b'{"msg":"bad request"}', "application/json")
                    return
                with ui._lock:
                    ui._cmds.append(c)
                msg = {"start": "Starting…", "stop": "Stopping…", "mode": f"Mode: {c.get('mode')}",
                       "click": "Target sent", "elevator": f"Elevator: {c.get('op')}"}.get(c.get("cmd"), "")
                self._send(200, json.dumps({"msg": msg}).encode(), "application/json")

        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        print(f"UI running at http://localhost:{port}")

    # ---- used by the control loop ----------------------------------------
    def pop_commands(self):
        with self._lock:
            c, self._cmds = self._cmds, []
        return c

    def set_state(self, **kw):
        with self._lock:
            self.state.update(kw)

    def set_overlay(self, **kw):
        with self._lock:
            self.overlay.update(kw)

    def publish_frame(self, frame):
        """Called for every processed camera frame (from the env); throttled."""
        now = time.time()
        if now - self._last_pub < 1.0 / STREAM_FPS or frame is None:
            return
        self._last_pub = now
        with self._lock:
            ov, st = dict(self.overlay), dict(self.state)
        img = frame.copy()
        if ov.get("contour_px") is not None and len(ov["contour_px"]) >= 3:
            cv2.polylines(img, [np.asarray(ov["contour_px"]).astype(np.int32)], True, (255, 0, 255), 1, cv2.LINE_AA)
        if ov.get("path_px") is not None and len(ov["path_px"]) >= 2:
            cv2.polylines(img, [np.asarray(ov["path_px"]).astype(np.int32)], bool(ov.get("path_closed")),
                          (255, 0, 255), 1, cv2.LINE_AA)
        for hx, hy, hr, kr in (ov.get("holes_px") or []):
            c = (int(round(hx)), int(round(hy)))
            cv2.circle(img, c, max(2, int(round(hr))), (0, 0, 255), 1, cv2.LINE_AA)       # the hole
            cv2.circle(img, c, max(3, int(round(kr))), (0, 140, 255), 1, cv2.LINE_AA)     # keep-out zone
        if ov.get("goal_px") is not None:
            gx, gy = (int(round(v)) for v in ov["goal_px"])
            if ov.get("goal_r_px"):
                cv2.circle(img, (gx, gy), max(2, int(round(ov["goal_r_px"]))), (0, 200, 0), 1, cv2.LINE_AA)
            cv2.drawMarker(img, (gx, gy), (0, 200, 0), cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
        if ov.get("ball_px") is not None and np.all(np.isfinite(ov["ball_px"])):
            by, bx = ov["ball_px"]
            cv2.circle(img, (int(round(bx)), int(round(by))), 9, (0, 220, 255), 1, cv2.LINE_AA)
        label = ("BALANCING" if st.get("running") else "STOPPED") + "  |  " + {
            "click": "click to target", "sheet": "red region", "line": "follow red line",
            "path": "drawn path", "drop": "drop into hole"}.get(st.get("mode"), "")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (4, 4), (12 + tw, 12 + th), (0, 0, 0), -1)
        cv2.putText(img, label, (8, 8 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._lock:
                self._jpeg = jpg.tobytes()
                self._jpeg_seq += 1

    def close(self):
        try:
            self._httpd.shutdown()
        except Exception:
            pass
