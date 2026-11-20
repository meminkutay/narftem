"""nartfem Strain Calculator - a small local application around nartfem_strains.

Start it (python app.py, or the packaged executable): it serves one page on 127.0.0.1 and opens
it in the default browser.  Nothing leaves the computer.  The process ends when the page is
closed (or with the Quit button).
"""
from __future__ import annotations
import json
import os
import socket
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import nartfem_strains as ns                     # noqa: E402
from app_page import PAGE                        # noqa: E402
from app_examples import EXAMPLES                # noqa: E402

APP_NAME = "nartfem Strain Calculator"
IDLE_EXIT_S = 240          # quit when the page has not been heard from for this long (and nothing is running)


class Job:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.cancel = False
        self.log = []
        self.done = 0
        self.total = 0
        self.result = None
        self.error = None
        self.id = 0
        self.t0 = 0.0
        self.elapsed = 0.0

    def start(self, data):
        with self.lock:
            if self.running:
                return False
            self.running, self.cancel = True, False
            self.log, self.done, self.total = [], 0, 0
            self.result, self.error = None, None
            self.id += 1
            self.t0 = time.time()
        threading.Thread(target=self._work, args=(data,), daemon=True).start()
        return True

    def _log(self, *a):
        self.log.append(" ".join(str(x) for x in a))

    def _progress(self, done, total):
        self.done, self.total = done, total

    def _work(self, data):
        try:
            self.result = ns.run_all(data, log=self._log, progress=self._progress, cancelled=lambda: self.cancel)
        except ns.InputError as e:
            self.error = str(e)
        except Exception as e:                                # no traceback: nothing about the code is shown
            self.error = "The analysis failed (%s: %s). Check the input values." % (type(e).__name__, e)
        finally:
            self.elapsed = time.time() - self.t0
            self.running = False

    def status(self, since):
        return dict(id=self.id, running=self.running, done=self.done, total=self.total,
                    log=self.log[since:], n_log=len(self.log), error=self.error,
                    elapsed=(time.time() - self.t0) if self.running else self.elapsed,
                    result=None if self.running else self.result)


JOB = Job()
LAST_SEEN = [time.time()]
SERVER = [None]


def parse_json(text):
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, "The input is not valid JSON: %s at line %d, column %d." % (e.msg, e.lineno, e.colno)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json", code=200):
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def do_GET(self):
        LAST_SEEN[0] = time.time()
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/favicon.ico":
            self.send_response(204); self.send_header("Content-Length", "0"); self.end_headers()
        elif path == "/api/examples":
            self._send(dict(examples=EXAMPLES, version=ns.__version__))
        elif path == "/api/job":
            since = 0
            if "since=" in self.path:
                try:
                    since = int(self.path.split("since=")[1].split("&")[0])
                except ValueError:
                    since = 0
            self._send(JOB.status(since))
        else:
            self._send(dict(error="not found"), code=404)

    def do_POST(self):
        LAST_SEEN[0] = time.time()
        path = self.path.split("?")[0]
        body = self._body()
        if path == "/api/run":
            data, err = parse_json(body.get("text", ""))
            if err:
                self._send(dict(ok=False, error=err)); return
            try:
                n = len(ns.expand_sweep(data)) if isinstance(data, dict) else 0     # early input check
                if isinstance(data, dict):
                    ns.build_load(data.get("load"))
            except ns.InputError as e:
                self._send(dict(ok=False, error=str(e))); return
            if not JOB.start(data):
                self._send(dict(ok=False, error="An analysis is already running.")); return
            self._send(dict(ok=True, id=JOB.id, total=n))
        elif path == "/api/stop":
            JOB.cancel = True
            self._send(dict(ok=True))
        elif path == "/api/ping":
            self._send(dict(ok=True))
        elif path == "/api/quit":
            self._send(dict(ok=True))
            threading.Thread(target=_shutdown, daemon=True).start()
        else:
            self._send(dict(error="not found"), code=404)


def _shutdown():
    time.sleep(0.3)
    if SERVER[0]:
        SERVER[0].shutdown()


def _watchdog():
    while True:
        time.sleep(5)
        if not JOB.running and time.time() - LAST_SEEN[0] > IDLE_EXIT_S:
            _shutdown(); return


def _free_port(preferred=8765):
    for port in list(range(preferred, preferred + 20)) + [0]:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", port))
            port = s.getsockname()[1]
            s.close()
            return port
        except OSError:
            continue
    raise RuntimeError("no free port")


def main(open_browser=True, port=None):
    port = port or _free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    SERVER[0] = srv
    url = "http://127.0.0.1:%d/" % port
    try:
        print("%s is running at %s\nLeave this window open while you work; close the browser page to quit." % (APP_NAME, url), flush=True)
    except Exception:
        pass                                                   # windowed build: no console
    threading.Thread(target=_watchdog, daemon=True).start()
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    srv.server_close()


def selftest():
    """Run every example once and print the critical strains (checks a packaged build)."""
    for ex in EXAMPLES:
        out = ns.run_all(ex["data"], log=lambda *a: None)
        r = out["rows"][0]
        print("%-48s cases %2d   tensile %8.1f   subgrade %9.1f microstrain" % (
            ex["label"][:48], len(out["rows"]), r["eps_h_max"], r["eps_zz_subgrade"]), flush=True)
    print("self-test finished.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest(); sys.exit(0)
    main(open_browser="--no-browser" not in sys.argv,
         port=int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else None)
