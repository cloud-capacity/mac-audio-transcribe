"""MCP server exposing the Mac recorder and Whisper transcriber as tools.

Recording is long-running but MCP calls are request/response, so recordings are
modelled as sessions: start_recording returns immediately with an id, get_status
reports progress (including the live transcript as it accumulates), and
stop_recording finalises. Every tool returns quickly.

Run over stdio; see README for registering it with Claude Desktop.
"""
import datetime
import os
import sys
import threading
import time
import uuid
from typing import Any, Optional

from mcp.server import MCPServer

import audio_core as core
import check_audio

mcp = MCPServer("mac-audio-transcribe")

OUTPUT_DIR = os.environ.get(
    "TRANSCRIBE_OUTPUT_DIR", os.path.dirname(os.path.abspath(__file__)))
MAX_RECORDING_SECONDS = int(os.environ.get("TRANSCRIBE_MAX_SECONDS", "3600"))

_SESSIONS: dict[str, "Session"] = {}
_SESSIONS_LOCK = threading.Lock()
# One Whisper pass at a time: two concurrent decodes contend for the same cores
# (or the same GPU) and both end up slower than either alone.
_TRANSCRIBE_LOCK = threading.Lock()


class Session:
    def __init__(self, kind: str, max_seconds: int = MAX_RECORDING_SECONDS):
        self.id = f"{kind[:3]}_{uuid.uuid4().hex[:8]}"
        self.kind = kind
        self.status = "recording" if kind == "recording" else "transcribing"
        self.created = time.time()
        self.max_seconds = max_seconds
        self.recorder: Optional[core.Recorder] = None
        self.wav_path: Optional[str] = None
        self.transcript = ""
        self.detected_language: Optional[str] = None
        self.compute_device: Optional[str] = None
        self.model: Optional[str] = None
        self.note: Optional[str] = None
        self.stop_reason: Optional[str] = None
        self.error: Optional[str] = None
        self.done = threading.Event()
        self.lock = threading.Lock()


def _get(session_id: str) -> Session:
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(session_id)
    if session is None:
        known = ", ".join(_SESSIONS) or "none"
        raise ValueError(f"No session {session_id!r}. Known sessions: {known}")
    return session


def _active_recording() -> Optional[Session]:
    with _SESSIONS_LOCK:
        for s in _SESSIONS.values():
            if s.kind == "recording" and s.status == "recording":
                return s
    return None


def _finish_recording(session: Session, reason: str) -> None:
    """Stop capture and drain the worker. Idempotent."""
    with session.lock:
        if session.status != "recording":
            return
        session.status = "stopping"
    try:
        session.recorder.stop()
        session.transcript = session.recorder.transcript
        session.stop_reason = reason
        session.status = "complete"
    except Exception as exc:
        session.error = f"{type(exc).__name__}: {exc}"
        session.status = "error"
    finally:
        session.done.set()


def _watchdog(session: Session) -> None:
    """Enforce the recording's limits — nothing else is looping over this session."""
    while session.status == "recording":
        time.sleep(0.5)
        rec = session.recorder
        if rec is None:
            return
        if rec.elapsed > session.max_seconds:
            _finish_recording(session, f"max_seconds ({session.max_seconds}) reached")
            return
        if rec.seconds_since_activity > core.SILENCE_TIMEOUT:
            _finish_recording(
                session, f"no audio for {core.SILENCE_TIMEOUT}s — auto-stopped")
            return


def _recording_status(session: Session) -> dict[str, Any]:
    rec = session.recorder
    out: dict[str, Any] = {
        "session_id": session.id,
        "kind": "recording",
        "status": session.status,
        "wav_path": session.wav_path,
        "model": session.model,
        "compute_device": session.compute_device,
        "transcript": session.transcript or (rec.transcript if rec else ""),
    }
    if rec is not None:
        out.update({
            "elapsed_seconds": round(rec.elapsed, 1),
            "levels": [round(float(v), 5) for v in rec.levels],
            "silent_for_seconds": round(rec.seconds_since_activity, 1),
            "lag_seconds": round(rec.lag, 1),
            "segments": rec.segmenter.segment_count if rec.segmenter else 0,
            "buffer_overruns": rec.xrun_count,
        })
    for key in ("stop_reason", "note", "error"):
        if getattr(session, key):
            out[key] = getattr(session, key)
    return out


def _job_status(session: Session) -> dict[str, Any]:
    out: dict[str, Any] = {
        "session_id": session.id,
        "kind": "transcription",
        "status": session.status,
        "elapsed_seconds": round(time.time() - session.created, 1),
        "model": session.model,
        "compute_device": session.compute_device,
        "transcript": session.transcript,
    }
    for key in ("detected_language", "note", "error"):
        if getattr(session, key):
            out[key] = getattr(session, key)
    return out


# ----------------------------------------------------------------------- tools

@mcp.tool()
def list_audio_devices() -> dict:
    """List audio input devices available for recording.

    Use this before start_recording so you can pass a device by name. Indices
    differ between machines; names do not.
    """
    cpu_brand, device = core.pick_compute_device()
    models = core.models_for(device)
    return {
        "devices": core.list_input_devices(),
        "cpu": cpu_brand,
        "compute_device": device,
        "default_live_model": models["live"],
        "default_final_model": models["final"],
    }


@mcp.tool()
def start_recording(
    device: Optional[str] = None,
    language: Optional[str] = None,
    live: bool = True,
    model: Optional[str] = None,
    max_seconds: int = MAX_RECORDING_SECONDS,
) -> dict:
    """Start recording audio. Returns immediately with a session id.

    Poll get_status(session_id) to watch the transcript build up, then call
    stop_recording(session_id) when finished. Recording also stops on its own
    after max_seconds, or after 60 seconds of silence.

    device: name substring (preferred) or index from list_audio_devices; the
        system default input is used when omitted.
    language: spoken language such as "english". Omit to auto-detect.
    live: transcribe while recording. Set false to only capture audio, then use
        transcribe_file on the returned wav_path afterwards.
    model: Whisper model for the live pass; defaults to the machine's setting.
    """
    existing = _active_recording()
    if existing is not None:
        raise ValueError(
            f"Recording {existing.id} is already running — stop it first "
            "(only one recording at a time).")

    resolved = core.resolve_device(device)
    sample_rate = core.choose_sample_rate(resolved, live)
    _, compute_device = core.pick_compute_device()
    model_name = model or core.models_for(compute_device)["live"]

    session = Session("recording", max_seconds=max(1, int(max_seconds)))
    session.compute_device = compute_device

    loaded = None
    if live:
        loaded = core.load_model(model_name, compute_device)
        session.model = model_name
        session.compute_device = loaded.device
        if loaded.fell_back:
            session.note = loaded.reason

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session.wav_path = os.path.join(OUTPUT_DIR, f"recording_{stamp}.wav")

    session.recorder = core.Recorder(
        resolved, sample_rate, session.wav_path,
        live=live, language=language, loaded_model=loaded)
    try:
        session.recorder.start()
    except Exception as exc:
        session.status = "error"
        session.error = f"{type(exc).__name__}: {exc}"
        raise

    with _SESSIONS_LOCK:
        _SESSIONS[session.id] = session
    threading.Thread(target=_watchdog, args=(session,), daemon=True).start()

    result = {
        "session_id": session.id,
        "status": "recording",
        "device": resolved["name"],
        "channels": resolved["channels"],
        "sample_rate": sample_rate,
        "live": live,
        "model": session.model,
        "compute_device": session.compute_device,
        "wav_path": session.wav_path,
        "stops_automatically_after_seconds": session.max_seconds,
    }
    if session.note:
        result["note"] = session.note
    return result


@mcp.tool()
def get_status(session_id: str) -> dict:
    """Progress of a recording or a transcription job.

    For a recording: elapsed time, input levels, how far the transcriber is
    behind, and the transcript so far. For a transcription job: whether it has
    finished, and the text once it has.
    """
    session = _get(session_id)
    return (_recording_status(session) if session.kind == "recording"
            else _job_status(session))


@mcp.tool()
def stop_recording(session_id: str) -> dict:
    """Stop a recording and return its transcript and audio file path.

    The transcript is the live one. For a more accurate pass, call
    transcribe_file on the returned wav_path with a larger model.
    """
    session = _get(session_id)
    if session.kind != "recording":
        raise ValueError(f"{session_id} is a transcription job, not a recording.")
    _finish_recording(session, "stopped by request")
    status = _recording_status(session)
    status["hint"] = (
        f"For a more accurate transcript, call transcribe_file({session.wav_path!r}).")
    return status


@mcp.tool()
def transcribe_file(
    path: str,
    language: Optional[str] = None,
    model: Optional[str] = None,
    wait_seconds: int = 60,
) -> dict:
    """Transcribe an existing audio file (wav, mp3, m4a — anything ffmpeg reads).

    Waits up to wait_seconds for the result and returns the transcript inline.
    If it takes longer, returns a session_id to poll with get_status instead, so
    long files never block the call.
    """
    full = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(full):
        raise ValueError(f"No such file: {full}")

    _, compute_device = core.pick_compute_device()
    model_name = model or core.models_for(compute_device)["final"]

    session = Session("transcription")
    session.model = model_name
    session.compute_device = compute_device
    with _SESSIONS_LOCK:
        _SESSIONS[session.id] = session

    def run():
        try:
            with _TRANSCRIBE_LOCK:
                loaded = core.load_model(model_name, compute_device)
                session.compute_device = loaded.device
                if loaded.fell_back:
                    session.note = loaded.reason
                result = core.transcribe_audio(loaded, full, language=language)
            session.transcript = (result.get("text") or "").strip()
            session.detected_language = result.get("language")
            session.status = "complete"
        except Exception as exc:
            session.error = f"{type(exc).__name__}: {exc}"
            session.status = "error"
        finally:
            session.done.set()

    threading.Thread(target=run, daemon=True).start()

    if session.done.wait(timeout=max(0, wait_seconds)):
        out = _job_status(session)
        out["source"] = full
        return out

    return {
        "session_id": session.id,
        "status": "transcribing",
        "source": full,
        "model": model_name,
        "message": (f"Still running after {wait_seconds}s — poll "
                    f"get_status({session.id!r})."),
    }


@mcp.tool()
def check_audio_setup(device: Optional[str] = None, seconds: int = 5) -> dict:
    """Listen briefly and report which channels of a device carry audio.

    Diagnoses a BlackHole + Aggregate Device meeting setup: channels 1-2 should
    carry the far end and channel 3 your microphone. Play some audio and speak
    while this runs. Records nothing to disk.
    """
    return check_audio.probe(device, seconds=max(1, min(int(seconds), 60)))


if __name__ == "__main__":
    # Nothing may be written to stdout: it is the JSON-RPC transport. audio_core
    # never prints, and every Whisper call passes verbose=None (the only silent
    # setting — see transcribe_audio). Diagnostics go to stderr.
    print(f"mac-audio-transcribe MCP server — output dir {OUTPUT_DIR}", file=sys.stderr)
    mcp.run()
