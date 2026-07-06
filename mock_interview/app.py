# Local mock-interview server: serves the UI, runs candidate code in a real
# Python subprocess, and grades stages against the hidden tests in problems.py.
#
#   python3 app.py          -> http://localhost:5050

import json
import os
import subprocess
import sys
import tempfile
import time

from flask import Flask, jsonify, request, send_from_directory

from problems import SESSIONS
from remediation import REMEDIATION, tag_for, STAGE_TAGS
from resume_sessions import VERBAL_SESSIONS
from variants import CODING_VARIANTS, VERBAL_VARIANTS

# attach variations: variant 0 is the base stage; the UI rotates by attempt count
for _s in SESSIONS:
    for _st in _s["stages"]:
        _st["variants"] = CODING_VARIANTS.get(_st["id"], [])
for _s in VERBAL_SESSIONS:
    for _st in _s["stages"]:
        _st["variants"] = VERBAL_VARIANTS.get(_st["id"], [])

ALL_SESSIONS = SESSIONS + VERBAL_SESSIONS

ROOT = os.path.dirname(os.path.abspath(__file__))
SESS_DIR = os.path.join(ROOT, "sessions")
ATTEMPTS = os.path.join(SESS_DIR, "attempts.jsonl")
VERBAL_LOG = os.path.join(SESS_DIR, "verbal.jsonl")
MARKER = "___RESULTS___"

# explicit instance_path: Flask otherwise falls back to os.getcwd(), which may
# be unreadable when launched by an external runner
app = Flask(__name__, static_folder=None, instance_path=ROOT)

HARNESS_HEAD = '''\
import numpy as np
import json as _json
import traceback as _tb
_results = []
def _check(name, fn):
    try:
        fn()
        _results.append({"name": name, "ok": True, "msg": ""})
    except AssertionError as e:
        _results.append({"name": name, "ok": False, "msg": str(e) or "assertion failed"})
    except Exception:
        _results.append({"name": name, "ok": False, "msg": _tb.format_exc(limit=2)})
'''


def run_python(source, timeout=15):
    """Run source in a fresh interpreter; returns (stdout, stderr, timed_out)."""
    fd, path = tempfile.mkstemp(suffix=".py", prefix="mock_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(source)
        try:
            p = subprocess.run(
                [sys.executable, path],
                capture_output=True, text=True, timeout=timeout,
            )
            return p.stdout, p.stderr, False
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
            return out, err + f"\n[timed out after {timeout}s -- infinite loop?]", True
    finally:
        os.unlink(path)


def find_stage(session_id, stage_id):
    for s in SESSIONS:
        if s["id"] == session_id:
            for st in s["stages"]:
                if st["id"] == stage_id:
                    return st
    return None


def grade(code, tests, timeout=15):
    source = (
        HARNESS_HEAD
        + "\n# ---- candidate code ----\n" + code
        + "\n\n# ---- hidden tests ----\n" + tests
        + f'\nprint("{MARKER}" + _json.dumps(_results))\n'
    )
    stdout, stderr, timed_out = run_python(source, timeout)
    if MARKER in stdout:
        head, tail = stdout.rsplit(MARKER, 1)
        results = json.loads(tail.strip())
        return {"ok": True, "results": results, "stdout": head, "stderr": stderr}
    # candidate code crashed before tests could run (syntax error etc.)
    return {"ok": False, "results": [], "stdout": stdout, "stderr": stderr,
            "timed_out": timed_out}


@app.get("/")
def home():
    return send_from_directory(ROOT, "index.html")


@app.get("/problems")
def problems():
    return jsonify(ALL_SESSIONS)


@app.post("/log_verbal")
def log_verbal():
    body = request.get_json(force=True)
    rec = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": body.get("session_id"),
        "stage_id": body.get("stage_id"),
        "topic": body.get("topic"),
        "title": body.get("title"),
        "score": int(body.get("score", 0)),
    }
    os.makedirs(SESS_DIR, exist_ok=True)
    with open(VERBAL_LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return jsonify({"ok": True})


@app.post("/run")
def run():
    code = request.get_json(force=True).get("code", "")
    stdout, stderr, timed_out = run_python(code)
    return jsonify({"stdout": stdout, "stderr": stderr, "timed_out": timed_out})


def log_attempt(session_id, stage_id, code, graded, variant=0):
    """Append one submit attempt to attempts.jsonl for the gap report."""
    if graded["ok"]:
        failed = [{"name": t["name"], "tag": tag_for(stage_id, t["name"]),
                   "msg": t["msg"].strip().splitlines()[-1][:300] if t["msg"] else ""}
                  for t in graded["results"] if not t["ok"]]
        tags_seen = sorted({tag_for(stage_id, t["name"]) for t in graded["results"]})
        n_pass = sum(1 for t in graded["results"] if t["ok"])
    else:  # crashed before tests ran
        failed = [{"name": "(code crashed before tests ran)", "tag": "crash-discipline",
                   "msg": (graded.get("stderr") or "").strip().splitlines()[-1][:300]
                          if graded.get("stderr") else ""}]
        tags_seen = ["crash-discipline"]
        n_pass = 0
    rec = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "session_id": session_id, "stage_id": stage_id, "variant": variant,
        "n_pass": n_pass, "n_fail": len(failed),
        "failed": failed, "tags_seen": tags_seen,
        "code": code[:20000],
    }
    os.makedirs(SESS_DIR, exist_ok=True)
    with open(ATTEMPTS, "a") as f:
        f.write(json.dumps(rec) + "\n")


@app.post("/submit")
def submit():
    body = request.get_json(force=True)
    stage = find_stage(body.get("session_id"), body.get("stage_id"))
    if stage is None:
        return jsonify({"error": "unknown stage"}), 404
    vi = int(body.get("variant", 0))
    tests = stage["tests"] if vi == 0 else stage["variants"][vi - 1]["tests"]
    graded = grade(body.get("code", ""), tests)
    log_attempt(body.get("session_id"), body.get("stage_id"),
                body.get("code", ""), graded, variant=vi)
    return jsonify(graded)


def load_attempts():
    if not os.path.exists(ATTEMPTS):
        return []
    out = []
    with open(ATTEMPTS) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def stage_title(stage_id):
    for s in ALL_SESSIONS:
        for st in s["stages"]:
            if st["id"] == stage_id:
                return f"{s['title']} — {st['title']}"
    return stage_id


def load_verbal():
    if not os.path.exists(VERBAL_LOG):
        return []
    out = []
    with open(VERBAL_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def verbal_counts():
    counts = {}
    for r in load_verbal():
        sid = r.get("stage_id")
        if sid:
            counts[sid] = counts.get(sid, 0) + 1
    return counts


def resume_readiness():
    """Per-topic average self-score from verbal deep-dive sessions.
    For repeated attempts of the same question, only the LATEST score counts
    (you're supposed to improve), but the attempt count is kept."""
    latest = {}
    counts = {}
    for r in load_verbal():
        if not r.get("topic") or not r.get("stage_id"):
            continue
        latest[r["stage_id"]] = r
        counts[r["stage_id"]] = counts.get(r["stage_id"], 0) + 1
    topics = {}
    for sid, r in latest.items():
        t = topics.setdefault(r["topic"], {"scores": [], "questions": []})
        t["scores"].append(r["score"])
        t["questions"].append({"stage_id": sid, "title": r.get("title", sid),
                               "score": r["score"], "attempts": counts[sid]})
    out = []
    for topic, t in topics.items():
        qs = sorted(t["questions"], key=lambda q: q["score"])
        out.append({
            "topic": topic,
            "avg": round(sum(t["scores"]) / len(t["scores"]), 2),
            "n": len(t["scores"]),
            "weakest": qs[:3],
        })
    out.sort(key=lambda x: x["avg"])
    return out


@app.get("/gaps")
def gaps():
    attempts = load_attempts()
    tags = {}   # tag -> aggregate
    stages = {}  # stage_id -> {attempts, passes}
    for a in attempts:
        st = stages.setdefault(a["stage_id"], {"attempts": 0, "clean": 0})
        st["attempts"] += 1
        if a["n_fail"] == 0:
            st["clean"] += 1
        for t in a["tags_seen"]:
            tags.setdefault(t, {"fails": 0, "seen": 0, "stages": set(),
                                "examples": [], "last": ""})["seen"] += 1
        for fl in a["failed"]:
            g = tags[fl["tag"]]
            g["fails"] += 1
            g["stages"].add(a["stage_id"])
            g["last"] = a["ts"]
            ex = (fl["name"] + (" — " + fl["msg"] if fl["msg"] else "")).strip()
            if ex not in g["examples"]:
                g["examples"] = (g["examples"] + [ex])[-3:]
    # hints used per stage from saved debriefs -> attributed to the stage's tag
    hints_by_tag = {}
    if os.path.isdir(SESS_DIR):
        for name in os.listdir(SESS_DIR):
            if name.endswith(".json"):
                try:
                    with open(os.path.join(SESS_DIR, name)) as f:
                        d = json.load(f)
                    for stg in d.get("stages", []):
                        t = STAGE_TAGS.get(stg.get("id"), "general")
                        hints_by_tag[t] = hints_by_tag.get(t, 0) + int(stg.get("hints", 0))
                except (json.JSONDecodeError, OSError):
                    pass
    weak, clean = [], []
    for tag, g in tags.items():
        hints = hints_by_tag.get(tag, 0)
        entry = {
            "tag": tag, "fails": g["fails"], "hints": hints,
            "stages": sorted(g["stages"]), "last": g["last"],
            "examples": g["examples"],
            "remediation": REMEDIATION.get(tag),
        }
        if g["fails"] > 0 or hints > 0:
            weak.append(entry)
        else:
            clean.append(tag)
    weak.sort(key=lambda e: -(e["fails"] * 2 + e["hints"]))
    return jsonify({
        "total_attempts": len(attempts),
        "total_fails": sum(a["n_fail"] for a in attempts),
        "weak": weak,
        "clean": sorted(clean),
        "stage_stats": [
            {"stage_id": sid, "title": stage_title(sid), **v}
            for sid, v in sorted(stages.items())
        ],
        "resume": resume_readiness(),
        # attempt counts per stage, used by the UI to rotate variants
        "counts": {**{sid: v["attempts"] for sid, v in stages.items()},
                   **verbal_counts()},
    })


@app.get("/gap_report")
def gap_report():
    return send_from_directory(ROOT, "gap_report.html")


@app.get("/history")
def history_page():
    return send_from_directory(ROOT, "history.html")


@app.get("/history_data")
def history_data():
    attempts = []
    for i, a in enumerate(load_attempts()):
        attempts.append({
            "idx": i, "ts": a.get("ts", ""), "session_id": a.get("session_id"),
            "stage_id": a.get("stage_id"), "title": stage_title(a.get("stage_id", "")),
            "variant": a.get("variant", 0),
            "n_pass": a.get("n_pass", 0), "n_fail": a.get("n_fail", 0),
        })
    verbal = []
    for i, v in enumerate(load_verbal()):
        verbal.append({
            "idx": i, "ts": v.get("ts", ""), "stage_id": v.get("stage_id"),
            "title": v.get("title", ""), "topic": v.get("topic", ""),
            "score": v.get("score", 0),
        })
    debriefs = []
    if os.path.isdir(SESS_DIR):
        for name in sorted(os.listdir(SESS_DIR)):
            if name.endswith(".json") and os.path.isfile(os.path.join(SESS_DIR, name)):
                try:
                    with open(os.path.join(SESS_DIR, name)) as f:
                        d = json.load(f)
                    debriefs.append({
                        "file": name, "ts": d.get("saved_at", ""),
                        "title": d.get("title", name),
                        "passed": d.get("passed", 0), "total": d.get("total", 0),
                        "elapsed": d.get("elapsed", ""),
                    })
                except (json.JSONDecodeError, OSError):
                    debriefs.append({"file": name, "ts": "", "title": name + " (unreadable)",
                                     "passed": 0, "total": 0, "elapsed": ""})
    return jsonify({"attempts": attempts, "verbal": verbal, "debriefs": debriefs})


def _archive_lines(kind, records):
    """Append deleted jsonl records to sessions/archive/deleted_<kind>.jsonl."""
    dest_dir = os.path.join(SESS_DIR, "archive")
    os.makedirs(dest_dir, exist_ok=True)
    with open(os.path.join(dest_dir, f"deleted_{kind}.jsonl"), "a") as f:
        for r in records:
            r = dict(r)
            r["_deleted_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(json.dumps(r) + "\n")


def _delete_jsonl(path, kind, wanted):
    """Remove records by (idx, ts) pairs; mismatched pairs are skipped.
    Deleted records are archived. Returns number removed."""
    if not os.path.exists(path) or not wanted:
        return 0
    with open(path) as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    kill = set()
    for w in wanted:
        i = int(w.get("idx", -1))
        if 0 <= i < len(lines):
            try:
                rec = json.loads(lines[i])
            except json.JSONDecodeError:
                continue
            if rec.get("ts", "") == w.get("ts", ""):   # guard against index drift
                kill.add(i)
    if not kill:
        return 0
    _archive_lines(kind, [json.loads(lines[i]) for i in sorted(kill)])
    keep = [ln for i, ln in enumerate(lines) if i not in kill]
    with open(path, "w") as f:
        f.write("\n".join(keep) + ("\n" if keep else ""))
    return len(kill)


@app.post("/delete_history")
def delete_history():
    body = request.get_json(force=True)
    n_att = _delete_jsonl(ATTEMPTS, "attempts", body.get("attempts", []))
    n_ver = _delete_jsonl(VERBAL_LOG, "verbal", body.get("verbal", []))
    n_deb = 0
    dest_dir = os.path.join(SESS_DIR, "archive")
    for name in body.get("debriefs", []):
        # filenames only — no path components
        if os.path.sep in name or ".." in name or not name.endswith(".json"):
            continue
        path = os.path.join(SESS_DIR, name)
        if os.path.isfile(path):
            os.makedirs(dest_dir, exist_ok=True)
            os.rename(path, os.path.join(dest_dir, name))
            n_deb += 1
    return jsonify({"deleted": {"attempts": n_att, "verbal": n_ver, "debriefs": n_deb},
                    "archived_to": "sessions/archive/"})


@app.post("/clear_history")
def clear_history():
    """Archive all logged history (attempts, verbal scores, debriefs) into
    sessions/archive/<timestamp>/ — resets the gap report, picker stats, and
    variant rotation without destroying anything."""
    if not os.path.isdir(SESS_DIR):
        return jsonify({"archived": 0, "to": None})
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(SESS_DIR, "archive", stamp)
    moved = 0
    for name in os.listdir(SESS_DIR):
        path = os.path.join(SESS_DIR, name)
        if os.path.isfile(path) and (name.endswith(".jsonl") or name.endswith(".json")):
            os.makedirs(dest, exist_ok=True)
            os.rename(path, os.path.join(dest, name))
            moved += 1
    return jsonify({"archived": moved, "to": f"sessions/archive/{stamp}/" if moved else None})


@app.get("/study")
def study():
    return send_from_directory(ROOT, "study.html")


@app.post("/save_session")
def save_session():
    os.makedirs(SESS_DIR, exist_ok=True)
    body = request.get_json(force=True)
    body["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    # "_file" lets the client update its auto-saved debrief in place instead
    # of creating a duplicate (which would double-count hints in /gaps)
    name = body.pop("_file", None)
    if not name or os.path.sep in name or ".." in name:
        name = time.strftime("%Y%m%d_%H%M%S_") + str(body.get("session_id", "x")) + ".json"
    path = os.path.join(SESS_DIR, name)
    with open(path, "w") as f:
        json.dump(body, f, indent=2)
    return jsonify({"saved": name})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"Mock interview simulator -> http://localhost:{port}")
    app.run(port=port, debug=False)
