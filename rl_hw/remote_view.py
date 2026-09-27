#!/usr/bin/env python3
"""
View-only live view for a phone (2026-09-27): relays the controller UI's latest camera
frame and status (http://localhost:8000) to the local network, WITHOUT any controls.
Every request needs the key from the link (?k=...), so only people you give the link to
can look.

  python3 remote_view.py [key] [port]      -> open http://<this Mac's IP>:<port>/?k=<key>
"""
import json
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

KEY = sys.argv[1] if len(sys.argv) > 1 else "change-me"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8001
SRC = "http://127.0.0.1:8000"

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CyberRunner view</title>
<style>
 :root { --bg:#f6f5f2; --ink:#1d1d1f; --muted:#6b6b70; }
 @media (prefers-color-scheme: dark) { :root { --bg:#161617; --ink:#ececee; --muted:#9a9aa0; } }
 body { margin:0; background:var(--bg); color:var(--ink); font:15px -apple-system, system-ui, sans-serif; }
 main { max-width:760px; margin:0 auto; padding:12px 16px; }
 img { width:100%; border-radius:8px; background:#000; display:block; }
 #s { margin-top:10px; color:var(--muted); line-height:1.5; }
 b { color:var(--ink); }
</style></head><body><main>
<img id="cam" alt="Live view of the plate">
<div id="s">connecting…</div>
</main><script>
const K = new URLSearchParams(location.search).get('k') || '';
function frame() {
  const i = new Image(); let done = false;
  const next = ok => { if (done) return; done = true; setTimeout(frame, ok ? 120 : 800); };
  i.onload = () => { document.getElementById('cam').src = i.src; next(true); };
  i.onerror = () => next(false); setTimeout(() => next(false), 4000);
  i.src = '/frame.jpg?k=' + encodeURIComponent(K) + '&t=' + Date.now();
}
async function state() {
  try {
    const s = await (await fetch('/state?k=' + encodeURIComponent(K))).json();
    const mm = v => (v * 1000).toFixed(0) + ' mm';
    document.getElementById('s').innerHTML =
      `<b>${s.running ? 'balancing' : 'stopped'}</b> · ${s.controller || 'classic'} · ` +
      `ball ${s.ball ? '(' + mm(s.ball[0]) + ', ' + mm(s.ball[1]) + ')' : '<b>not visible</b>'} · ` +
      `distance ${s.dist != null ? mm(s.dist) : '–'} · in target ${s.in_target ? 'yes' : 'no'} · ` +
      `elevator ${s.elev_on ? '<b>on</b>' : 'off'}` + (s.hole_msg ? ' · ' + s.hole_msg : '');
  } catch (e) { document.getElementById('s').textContent = 'controller not reachable'; }
  setTimeout(state, 1000);
}
frame(); state();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if parse_qs(u.query).get("k", [""])[0] != KEY:
            self._send(403, b"forbidden", "text/plain")
            return
        try:
            if u.path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif u.path == "/frame.jpg":
                with urllib.request.urlopen(SRC + "/frame.jpg", timeout=3) as r:
                    self._send(200, r.read(), "image/jpeg")
            elif u.path == "/state":
                with urllib.request.urlopen(SRC + "/state", timeout=3) as r:
                    s = json.load(r)
                keep = ("running", "controller", "ball", "dist", "in_target", "elev_on", "hole_msg", "mode")
                self._send(200, json.dumps({k: s.get(k) for k in keep}).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")
        except (OSError, ValueError):
            self._send(503, b"controller not reachable", "text/plain")


if __name__ == "__main__":
    print(f"view-only live view on port {PORT} (key required)")
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
