#!/usr/bin/env python3
"""webgpu-relay — job relay between the control API (Muse) and browser
agents (the WebGPU FDTD test bench page), over HTTP long-polling.

Agents (browsers):
  POST /agent/register {label, adapter, userAgent} -> {sessionId, agentKey}
  POST /agent/poll     {sessionId, agentKey} -> {job|null}   (long-poll 25 s)
  POST /agent/event    {sessionId, agentKey, jobId, line}   (stream a log line)
  POST /agent/result   {sessionId, agentKey, jobId, status, result}

Control (Authorization: Bearer <token from TOKEN_FILE>):
  GET  /api/sessions         -> live sessions with adapter info
  POST /api/jobs {sessionId, job} -> {jobId}
  GET  /api/jobs/<jobId>     -> full job record

GET /api/health -> {ok: true} (unauthenticated)

State is in memory only; a restart drops sessions (agents re-register
automatically) and queued jobs.
"""
import json
import os
import secrets
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("WEBGPU_RELAY_PORT", "8177"))
TOKEN_FILE = os.environ.get("WEBGPU_RELAY_TOKEN_FILE", "/opt/webgpu-relay/token")
POLL_SECONDS = 25
LIVE_SECONDS = 90
MAX_BODY = 4 * 1024 * 1024
MAX_LOG_LINES = 500

with open(TOKEN_FILE) as f:
    CONTROL_TOKEN = f.read().strip()

lock = threading.Lock()
cond = threading.Condition(lock)
sessions = {}  # sessionId -> {agentKey, label, adapter, userAgent, connectedAt, lastSeen, queue[], busy, currentJob}
jobs = {}      # jobId -> {id, sessionId, job, status, submittedAt, startedAt, finishedAt, result, logs[]}


def now():
    return time.time()


def live_session(sid):
    s = sessions.get(sid)
    if s and now() - s["lastSeen"] <= LIVE_SECONDS:
        return s
    return None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # quiet
        pass

    # -- plumbing ---------------------------------------------------------
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ValueError("body too large")
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw or b"{}")

    def _authed(self):
        auth = self.headers.get("Authorization") or ""
        return auth == "Bearer " + CONTROL_TOKEN

    def _agent(self, data):
        s = sessions.get(data.get("sessionId") or "")
        if not s or not secrets.compare_digest(s["agentKey"], data.get("agentKey") or ""):
            return None
        s["lastSeen"] = now()
        return s

    # -- routes -----------------------------------------------------------
    def do_GET(self):
        if self.path == "/api/health":
            return self._send(200, {"ok": True, "sessions": len(sessions), "jobs": len(jobs)})
        if self.path == "/api/sessions":
            if not self._authed():
                return self._send(401, {"error": "bad token"})
            with lock:
                out = []
                for sid, s in sessions.items():
                    out.append({
                        "sessionId": sid, "label": s["label"], "adapter": s["adapter"],
                        "pageVersion": s.get("pageVersion"),
                        "userAgent": s["userAgent"], "connectedAt": s["connectedAt"],
                        "lastSeen": s["lastSeen"], "live": now() - s["lastSeen"] <= LIVE_SECONDS,
                        "busy": s["busy"], "currentJob": s["currentJob"],
                        "queued": len(s["queue"]),
                    })
                out.sort(key=lambda x: x["connectedAt"])
            return self._send(200, {"sessions": out})
        if self.path.startswith("/api/jobs/"):
            if not self._authed():
                return self._send(401, {"error": "bad token"})
            jid = self.path[len("/api/jobs/"):]
            with lock:
                j = jobs.get(jid)
                if not j:
                    return self._send(404, {"error": "no such job"})
                return self._send(200, j)
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            data = self._body()
        except Exception as e:
            return self._send(400, {"error": str(e)})

        if self.path == "/agent/register":
            prev = data.get("resumeSessionId") or ""
            with lock:
                old = sessions.get(prev)
                if old and secrets.compare_digest(old["agentKey"], data.get("agentKey") or ""):
                    old["lastSeen"] = now()
                    old["label"] = str(data.get("label") or old["label"])[:120]
                    old["adapter"] = data.get("adapter")
                    old["pageVersion"] = str(data.get("pageVersion") or old.get("pageVersion") or "")[:40]
                    old["userAgent"] = str(data.get("userAgent") or old["userAgent"])[:400]
                    return self._send(200, {"sessionId": prev,
                                            "agentKey": old["agentKey"],
                                            "resumed": True})
            sid = uuid.uuid4().hex[:12]
            with lock:
                sessions[sid] = {
                    "agentKey": secrets.token_hex(16),
                    "label": str(data.get("label") or "")[:120],
                    "adapter": data.get("adapter"),
                    "pageVersion": str(data.get("pageVersion") or "")[:40],
                    "userAgent": str(data.get("userAgent") or "")[:400],
                    "connectedAt": now(), "lastSeen": now(),
                    "queue": [], "busy": False, "currentJob": None,
                }
                key = sessions[sid]["agentKey"]
            return self._send(200, {"sessionId": sid, "agentKey": key})

        if self.path == "/agent/poll":
            with lock:
                s = self._agent(data)
                if not s:
                    return self._send(401, {"error": "bad session"})
                deadline = now() + POLL_SECONDS
                while not s["queue"] and now() < deadline:
                    cond.wait(timeout=max(0.1, deadline - now()))
                    s["lastSeen"] = now()
                job = s["queue"].pop(0) if s["queue"] else None
                if job:
                    s["busy"] = True
                    s["currentJob"] = job["id"]
                    rec = jobs.get(job["id"])
                    if rec:
                        rec["status"] = "running"
                        rec["startedAt"] = now()
            return self._send(200, {"job": job})

        if self.path == "/agent/event":
            with lock:
                s = self._agent(data)
                if not s:
                    return self._send(401, {"error": "bad session"})
                rec = jobs.get(data.get("jobId") or "")
                if rec and len(rec["logs"]) < MAX_LOG_LINES:
                    rec["logs"].append(str(data.get("line") or "")[:500])
            return self._send(200, {"ok": True})

        if self.path == "/agent/result":
            with lock:
                s = self._agent(data)
                if not s:
                    return self._send(401, {"error": "bad session"})
                rec = jobs.get(data.get("jobId") or "")
                if rec:
                    rec["status"] = data.get("status") or "done"
                    rec["result"] = data.get("result")
                    rec["finishedAt"] = now()
                s["busy"] = False
                s["currentJob"] = None
                cond.notify_all()
            return self._send(200, {"ok": True})

        if self.path == "/api/jobs":
            if not self._authed():
                return self._send(401, {"error": "bad token"})
            sid = data.get("sessionId") or ""
            job = data.get("job")
            if not isinstance(job, dict) or not job.get("type"):
                return self._send(400, {"error": "job must be an object with a type"})
            with lock:
                s = live_session(sid)
                if not s:
                    return self._send(409, {"error": "session not live", "sessionId": sid})
                jid = uuid.uuid4().hex[:12]
                payload = dict(job)
                payload["id"] = jid
                jobs[jid] = {
                    "id": jid, "sessionId": sid, "job": job, "status": "queued",
                    "submittedAt": now(), "startedAt": None, "finishedAt": None,
                    "result": None, "logs": [],
                }
                if len(jobs) > 300:  # prune oldest finished
                    done = [k for k, v in sorted(jobs.items(), key=lambda kv: kv[1]["submittedAt"])
                            if v["status"] in ("done", "error")]
                    for k in done[: len(jobs) - 300]:
                        jobs.pop(k, None)
                s["queue"].append(payload)
                cond.notify_all()
            return self._send(200, {"jobId": jid})

        return self._send(404, {"error": "not found"})


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"webgpu-relay listening on 127.0.0.1:{PORT}", flush=True)
    server.serve_forever()
