"""End-to-end smoke test for mcp_server.py over real stdio JSON-RPC.

Checks three things:
  1. the protocol handshake works and all six tools are advertised;
  2. tools actually run (list_audio_devices, then transcribe_file on a fixture);
  3. **every byte the server writes to stdout is valid JSON-RPC** — stdout is the
     transport, so a stray print anywhere in the dependency tree breaks the server.
     This is the regression that would otherwise be found only by a confused client.

usage:  .venv-mcp/bin/python tests/smoke_mcp.py
"""
import json
import os
import queue
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(REPO, ".venv-mcp", "bin", "python")
FIXTURE = os.path.join(REPO, "tests", "fixtures", "speech.wav")

EXPECTED_TOOLS = {
    "list_audio_devices", "start_recording", "get_status",
    "stop_recording", "transcribe_file", "check_audio_setup",
}

failures = []


def _payload(response):
    """Unwrap a tools/call result into the dict the tool returned."""
    content = response.get("result", {}).get("content", [])
    for block in content:
        if block.get("type") == "text":
            try:
                return json.loads(block["text"])
            except json.JSONDecodeError:
                pass
    return response.get("result", {}).get("structuredContent", {}) or {}


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def main():
    if not os.path.exists(FIXTURE):
        sys.exit(f"missing fixture {FIXTURE} — run tests/make_fixture.sh first")

    proc = subprocess.Popen(
        [PYTHON, os.path.join(REPO, "mcp_server.py")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, cwd=REPO)

    raw_lines = []
    inbox = queue.Queue()

    def reader():
        for line in proc.stdout:
            raw_lines.append(line)
            inbox.put(line)
        inbox.put(None)

    threading.Thread(target=reader, daemon=True).start()

    def send(msg):
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()

    def await_id(want_id, timeout=180):
        """Read until the response with this id arrives."""
        while True:
            line = inbox.get(timeout=timeout)
            if line is None:
                raise RuntimeError("server closed stdout early")
            if not line.strip():
                continue
            msg = json.loads(line)          # raises if stdout was polluted
            if msg.get("id") == want_id:
                return msg

    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                         "clientInfo": {"name": "smoke", "version": "0"}}})
        init = await_id(1, timeout=60)
        check("initialize", "result" in init,
              init.get("result", {}).get("serverInfo", {}).get("name", ""))

        send({"jsonrpc": "2.0", "method": "notifications/initialized"})

        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        listed = await_id(2, timeout=60)
        names = {t["name"] for t in listed.get("result", {}).get("tools", [])}
        check("tools/list advertises all six", names == EXPECTED_TOOLS,
              f"got {sorted(names)}")

        send({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
              "params": {"name": "list_audio_devices", "arguments": {}}})
        devices = await_id(3, timeout=60)
        check("list_audio_devices runs",
              not devices.get("result", {}).get("isError", False),
              str(devices.get("result"))[:120])

        send({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
              "params": {"name": "transcribe_file",
                         "arguments": {"path": FIXTURE, "language": "english",
                                       "model": "base", "wait_seconds": 150}}})
        got = await_id(4, timeout=300)
        blob = json.dumps(got.get("result", {}))
        check("transcribe_file returns the transcript",
              "quick brown fox" in blob.lower(), blob[:160])

        # Recording cycle against BlackHole: a virtual device that is silent
        # unless audio is routed into it, so this needs no microphone.
        send({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
              "params": {"name": "start_recording",
                         "arguments": {"device": "BlackHole", "live": True,
                                       "model": "tiny", "max_seconds": 20}}})
        started = _payload(await_id(5, timeout=120))
        session_id = started.get("session_id")
        check("start_recording returns a session immediately",
              bool(session_id) and started.get("status") == "recording",
              f"{session_id} on {started.get('device')} @ "
              f"{started.get('sample_rate')}Hz")

        time.sleep(3)
        send({"jsonrpc": "2.0", "id": 6, "method": "tools/call",
              "params": {"name": "get_status",
                         "arguments": {"session_id": session_id}}})
        status = _payload(await_id(6, timeout=60))
        check("get_status reports a live recording",
              status.get("status") == "recording"
              and status.get("elapsed_seconds", 0) >= 2,
              f"elapsed={status.get('elapsed_seconds')}s "
              f"levels={status.get('levels')}")

        send({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
              "params": {"name": "stop_recording",
                         "arguments": {"session_id": session_id}}})
        stopped = _payload(await_id(7, timeout=180))
        wav = stopped.get("wav_path", "")
        check("stop_recording finalises and leaves a wav",
              stopped.get("status") == "complete" and os.path.exists(wav),
              f"{os.path.basename(wav)} "
              f"({os.path.getsize(wav) if os.path.exists(wav) else 0} bytes)")
        if wav and os.path.exists(wav):
            os.remove(wav)

        send({"jsonrpc": "2.0", "id": 8, "method": "tools/call",
              "params": {"name": "get_status",
                         "arguments": {"session_id": "nope_1234"}}})
        missing = await_id(8, timeout=30)
        check("unknown session is reported as a tool error",
              missing.get("result", {}).get("isError") is True,
              str(missing.get("result"))[:100])

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    # The whole point: nothing but JSON-RPC may reach stdout.
    bad = []
    for line in raw_lines:
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            bad.append(line[:100])
    check("stdout carries only JSON-RPC", not bad, f"{len(bad)} bad line(s): {bad[:3]}")

    stderr = proc.stderr.read()
    print(f"\n[stderr from server]\n{stderr.strip()[-600:] or '(empty)'}")

    print(f"\n{'ALL PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
