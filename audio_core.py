"""Recording and transcription core, shared by the CLI and the MCP server.

This module deliberately never prints and never prompts. `transcribe.py` owns the
terminal UI; `mcp_server.py` speaks JSON-RPC over stdout, where a stray write would
corrupt the transport. Keeping all I/O in the front-ends is what makes one
implementation safe to share.
"""
import contextlib
import platform
import queue
import subprocess
import threading
import time
import wave

import numpy as np
import sounddevice as sd
import whisper

# ---------- Configuration ----------
SAMPLE_RATE = 44100
REFRESH_HZ = 20
SILENCE_TIMEOUT = 60
SILENCE_THRESHOLD = 0.005

# --- Live transcription ---
WHISPER_RATE = 16000        # Whisper always works at 16 kHz internally
LIVE_SILENCE_SPLIT = 0.4    # seconds of silence that ends a segment
LIVE_MIN_SEGMENT = 1.0      # never transcribe anything shorter than this
LIVE_MAX_SEGMENT = 8.0      # force a cut if someone talks non-stop
LIVE_OVERLAP = 0.25         # audio carried into the next segment

# Model choice depends on the machine. The CPU row is measured on an Intel
# i5-10500: `base` runs at RTF 0.07-0.20 with stable latency, while `small` is
# erratic (0.19-1.91) because Whisper's temperature-fallback retries can make a
# 5 s chunk take 9.5 s. The MPS row is a starting point only — re-benchmark on
# Apple Silicon and adjust (see README).
MODEL_DEFAULTS = {
    "cpu": {"live": "base", "final": "small"},
    "mps": {"live": "small", "final": "medium"},
}
# -----------------------------------


def models_for(device):
    """Live and final model names for a compute device."""
    return MODEL_DEFAULTS.get(device, MODEL_DEFAULTS["cpu"])


# ---------------------------------------------------------------- compute device

def _cpu_brand():
    for mib in ("machdep.cpu.brand_string", "hw.model"):
        try:
            r = subprocess.run(["sysctl", "-n", mib], capture_output=True,
                               text=True, timeout=2)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        except Exception:
            continue
    return "Unknown"


def pick_compute_device():
    """Return (cpu_brand, device) where device is 'mps' or 'cpu'."""
    cpu_brand = _cpu_brand()
    try:
        import torch
        mps_available = (torch.backends.mps.is_available()
                         if hasattr(torch.backends, "mps") else False)
    except Exception:
        mps_available = False

    is_apple_silicon = platform.machine() == "arm64" or "Apple" in cpu_brand
    device = "mps" if (is_apple_silicon and mps_available) else "cpu"
    return cpu_brand, device


# ---------------------------------------------------------------- input devices

def list_input_devices():
    """Every input-capable device, with whether it will accept 16 kHz."""
    devices = sd.query_devices()
    out = []
    for idx, d in enumerate(devices):
        if d["max_input_channels"] < 1:
            continue
        channels = d["max_input_channels"]
        try:
            sd.check_input_settings(device=idx, channels=channels,
                                    dtype="float32", samplerate=WHISPER_RATE)
            accepts_16k = True
        except Exception:
            accepts_16k = False
        out.append({
            "index": idx,
            "name": d["name"],
            "channels": channels,
            "default_samplerate": int(d["default_samplerate"]),
            "accepts_16k": accepts_16k,
        })
    return out


def resolve_device(spec=None):
    """Resolve a device index, a name substring, or None to a device dict.

    Names are preferred over indices because indices differ between machines —
    the same aggregate device is not index 3 on both an iMac and a MacBook.
    """
    inputs = list_input_devices()
    if not inputs:
        raise RuntimeError("No audio input devices found.")

    if spec is None:
        try:
            default_idx = sd.default.device[0]
        except Exception:
            default_idx = None
        for d in inputs:
            if d["index"] == default_idx:
                return d
        return inputs[0]

    if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
        idx = int(spec)
        for d in inputs:
            if d["index"] == idx:
                return d
        raise ValueError(
            f"No input device with index {idx}. Available: "
            + ", ".join(f"{d['index']}={d['name']!r}" for d in inputs))

    matches = [d for d in inputs if spec.lower() in d["name"].lower()]
    if not matches:
        raise ValueError(
            f"No input device matching {spec!r}. Available: "
            + ", ".join(repr(d["name"]) for d in inputs))
    if len(matches) > 1:
        raise ValueError(
            f"{spec!r} matches several devices: "
            + ", ".join(repr(d["name"]) for d in matches))
    return matches[0]


def choose_sample_rate(device, live):
    """Pick the capture rate for a device dict.

    In live mode prefer 16 kHz when the device allows it: that is what Whisper
    wants anyway, so nothing is resampled and the WAV is ~3x smaller. Devices with
    a fixed rate keep their own and get resampled per segment. Classic mode keeps
    the historical 44.1 kHz so recordings are unchanged.
    """
    if live and device["accepts_16k"]:
        return WHISPER_RATE
    if live:
        return device["default_samplerate"]
    return SAMPLE_RATE


# ---------------------------------------------------------------- audio helpers

_taps = None


def resample_to_16k(x, src_rate):
    """Downsample mono float32 audio to 16 kHz (anti-alias filter + linear interp)."""
    global _taps
    if src_rate == WHISPER_RATE:
        return x.astype(np.float32, copy=False)
    if _taps is None:
        num_taps = 63
        cutoff = 0.45 * WHISPER_RATE / src_rate   # normalised to src_rate
        n = np.arange(num_taps) - (num_taps - 1) / 2
        h = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(num_taps)
        _taps = (h / h.sum()).astype(np.float32)
    x = np.convolve(x, _taps, mode="same")
    n_out = int(round(len(x) * WHISPER_RATE / src_rate))
    idx = np.linspace(0, len(x) - 1, n_out)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


def strip_overlap(prev, new, max_words=8):
    """Drop words the previous segment already ended with (the LIVE_OVERLAP tail
    gets transcribed twice, once at the end of a segment and once at the start
    of the next)."""
    def norm(words):
        return [w.lower().strip(".,!?;:") for w in words]

    prev_words, new_words = prev.split(), new.split()
    for k in range(min(max_words, len(prev_words), len(new_words)), 0, -1):
        if norm(prev_words[-k:]) == norm(new_words[:k]):
            return " ".join(new_words[k:])
    return new


# ---------------------------------------------------------------- whisper models

class LoadedModel:
    """A Whisper model plus the device it actually ended up on."""

    def __init__(self, name, model, device, fell_back=False, reason=None):
        self.name = name
        self.model = model
        self.device = device
        self.fell_back = fell_back
        self.reason = reason


_MODELS = {}
_MODEL_LOCK = threading.RLock()


def load_model(name, device=None):
    """Load (and cache) a Whisper model, falling back to CPU if the GPU path fails.

    The README has always claimed this fallback exists; until Apple Silicon entered
    the picture the `mps` branch never actually ran, so it was never needed. It is
    real now.
    """
    if device is None:
        device = pick_compute_device()[1]
    key = (name, device)
    with _MODEL_LOCK:
        if key in _MODELS:
            return _MODELS[key]
        try:
            loaded = LoadedModel(name, whisper.load_model(name, device=device), device)
        except Exception as exc:
            if device == "cpu":
                raise
            loaded = LoadedModel(
                name, whisper.load_model(name, device="cpu"), "cpu",
                fell_back=True,
                reason=f"loading on {device} failed ({exc.__class__.__name__}: {exc})")
        _MODELS[key] = loaded
        return loaded


def _demote_to_cpu(loaded, exc):
    """Reload a model on the CPU after a GPU decode failure."""
    with _MODEL_LOCK:
        if loaded.device == "cpu":
            return
        loaded.model = whisper.load_model(loaded.name, device="cpu")
        loaded.device = "cpu"
        loaded.fell_back = True
        loaded.reason = (f"transcribing on GPU failed "
                         f"({exc.__class__.__name__}: {exc}); switched to CPU")
        _MODELS[(loaded.name, "cpu")] = loaded


def transcribe_audio(loaded, audio, language=None, initial_prompt=None, fp16=False):
    """Run Whisper, keeping it silent and surviving a GPU decode failure.

    verbose=None is the silent setting and the only safe one here: verbose=False
    *enables* Whisper's tqdm bar (it uses `disable=verbose is not False`), and any
    non-None value enables its "Detected language" print. Either would corrupt the
    terminal meter, and would corrupt the MCP stdio transport outright.
    """
    kwargs = {"fp16": fp16, "verbose": None, "condition_on_previous_text": False}
    if language:
        kwargs["language"] = language
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt[-200:]

    try:
        return loaded.model.transcribe(audio, **kwargs)
    except Exception as exc:
        if loaded.device == "cpu":
            raise
        # Whisper's decoder has historically hit unimplemented MPS ops well after
        # the model itself loads cleanly, so guard the first decode too.
        _demote_to_cpu(loaded, exc)
        return loaded.model.transcribe(audio, **kwargs)


# ---------------------------------------------------------------- segmentation

class LiveSegmenter:
    """Cuts a stream into utterances on silence and transcribes each one.

    Fixed-size chunks cut words in half, so segments end after LIVE_SILENCE_SPLIT
    seconds of quiet; a talker who never pauses is cut at LIVE_MAX_SEGMENT. Each
    segment carries LIVE_OVERLAP seconds of the previous one so a clipped word
    survives, and the resulting duplicate words are stripped from the text.
    """

    def __init__(self, sample_rate, loaded_model, language=None, emit=None):
        self.sample_rate = sample_rate
        self.model = loaded_model
        self.emit = emit or (lambda kind, text: None)

        self._buf = []
        self._buf_len = 0
        self._silence_run = 0
        self._have_speech = False
        self._context = ""
        self._lang = language
        self._overlap = int(LIVE_OVERLAP * sample_rate)
        self._max_len = int(LIVE_MAX_SEGMENT * sample_rate)

        self.transcribed_secs = 0.0
        self.segment_count = 0
        self.lines = []

    def _take_segment(self):
        """Pop everything buffered, keeping a short tail for the next segment."""
        seg = np.concatenate(self._buf)
        tail = (seg[-self._overlap:].copy() if len(seg) > self._overlap
                else np.zeros(0, dtype=np.float32))
        self._buf = [tail] if len(tail) else []
        self._buf_len = len(tail)
        self._silence_run = 0
        self._have_speech = False
        return seg

    def _flush(self):
        if not self._buf:
            return
        seg = self._take_segment()
        secs = len(seg) / self.sample_rate

        try:
            result = transcribe_audio(
                self.model, resample_to_16k(seg, self.sample_rate),
                language=self._lang, initial_prompt=self._context or None)
        except Exception as exc:
            self.emit("note", f"live transcription failed: {exc}")
            self.transcribed_secs += secs
            return

        self.transcribed_secs += secs
        self.segment_count += 1
        if self._lang is None:
            # Pin the language after the first detection so it can't flip mid-recording.
            self._lang = result.get("language")
            if self._lang:
                self.emit("note", f"detected language: {self._lang}")

        text = (result.get("text") or "").strip()
        if text and self._context:
            text = strip_overlap(self._context, text).strip()
        if text:
            self._context = (self._context + " " + text).strip()
            self.lines.append(text)
            self.emit("text", text)

    def feed(self, block, rms):
        self._buf.append(block)
        self._buf_len += len(block)
        if rms > SILENCE_THRESHOLD:
            self._silence_run = 0
            self._have_speech = True
        else:
            self._silence_run += len(block)

        if not self._have_speech and self._buf_len > self._max_len:
            # Nothing but silence so far — drop it instead of asking Whisper to
            # transcribe it (it hallucinates on silence).
            dropped = self._take_segment()
            self.transcribed_secs += len(dropped) / self.sample_rate
            return

        secs = self._buf_len / self.sample_rate
        if self._have_speech and (
            (self._silence_run / self.sample_rate >= LIVE_SILENCE_SPLIT
             and secs >= LIVE_MIN_SEGMENT)
            or secs >= LIVE_MAX_SEGMENT
        ):
            self._flush()

    def finish(self):
        if self._have_speech:
            self._flush()

    @property
    def transcript(self):
        return "\n".join(self.lines)


# ---------------------------------------------------------------- recorder

_STOP = object()


class Recorder:
    """Owns the input stream, the WAV file, and the transcription worker.

    The audio callback only produces — it writes frames, stores levels, and hands
    blocks to the worker. Doing model work or any I/O there causes input overflows.
    """

    def __init__(self, device, sample_rate, wav_path, live=False,
                 language=None, loaded_model=None):
        self.device = device
        self.channels = device["channels"]
        self.sample_rate = sample_rate
        self.wav_path = wav_path
        self.live = live
        self.language = language
        self.model = loaded_model

        self.segmenter = None
        self.captured_secs = 0.0
        self.xrun_count = 0
        self.start_time = None
        self.last_activity = None
        self.error = None

        self._levels = np.zeros(self.channels, dtype=np.float32)
        self._mean_level = 0.0
        self._audio_q = queue.Queue()
        self._ui_q = queue.Queue()
        self._wf = None
        self._stream = None
        self._worker = None
        self._stopped = False

    # -- state readable by a UI or a status tool -------------------------------

    @property
    def levels(self):
        return self._levels

    @property
    def mean_level(self):
        return self._mean_level

    @property
    def elapsed(self):
        return 0.0 if self.start_time is None else time.time() - self.start_time

    @property
    def seconds_since_activity(self):
        return 0.0 if self.last_activity is None else time.time() - self.last_activity

    @property
    def lag(self):
        """Seconds of captured audio the transcriber has not caught up with."""
        if not self.segmenter:
            return 0.0
        return max(0.0, self.captured_secs - self.segmenter.transcribed_secs)

    @property
    def lines(self):
        return list(self.segmenter.lines) if self.segmenter else []

    @property
    def transcript(self):
        return self.segmenter.transcript if self.segmenter else ""

    def drain_text(self):
        """Pop (kind, text) items for a UI to display. Not needed to collect text."""
        out = []
        while True:
            try:
                out.append(self._ui_q.get_nowait())
            except queue.Empty:
                return out

    # -- lifecycle -------------------------------------------------------------

    def _emit(self, kind, text):
        self._ui_q.put((kind, text))

    def _callback(self, indata, frames_count, time_info, status):
        if status:
            # Printing here would corrupt the meter (and the MCP transport);
            # report the count when the recording ends instead.
            self.xrun_count += 1

        self._wf.writeframes(np.int16(indata * 32767).tobytes())

        rms = np.sqrt(np.mean(indata ** 2, axis=0))
        mean_rms = float(np.mean(rms))
        self._levels = rms
        self._mean_level = mean_rms
        self.captured_secs += frames_count / self.sample_rate

        if mean_rms > SILENCE_THRESHOLD:
            self.last_activity = time.time()

        if self.live:
            mono = indata[:, 0] if self.channels == 1 else indata.mean(axis=1)
            self._audio_q.put((np.array(mono, dtype=np.float32), mean_rms))

    def _run_worker(self):
        while True:
            item = self._audio_q.get()
            if item is _STOP:
                self.segmenter.finish()
                return
            block, rms = item
            try:
                self.segmenter.feed(block, rms)
            except Exception as exc:      # keep the recording alive regardless
                self.error = f"{exc.__class__.__name__}: {exc}"
                self._emit("note", f"live worker error: {self.error}")

    def start(self):
        self._wf = wave.open(self.wav_path, "wb")
        self._wf.setnchannels(self.channels)
        self._wf.setsampwidth(2)
        self._wf.setframerate(self.sample_rate)

        if self.live:
            self.segmenter = LiveSegmenter(self.sample_rate, self.model,
                                           language=self.language, emit=self._emit)
            self._worker = threading.Thread(target=self._run_worker, daemon=True)
            self._worker.start()

        self.start_time = time.time()
        self.last_activity = time.time()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            device=self.device["index"],
            callback=self._callback,
        )
        self._stream.start()

    def stop(self, timeout=300, on_progress=None):
        """Stop capture, drain the worker, close the WAV. Safe to call twice."""
        if self._stopped:
            return
        self._stopped = True

        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.stop()
                self._stream.close()
            self._stream = None

        if self._worker is not None:
            self._audio_q.put(_STOP)
            deadline = time.time() + timeout
            while self._worker.is_alive() and time.time() < deadline:
                if on_progress:
                    on_progress()
                self._worker.join(0.2)
            self._worker = None

        if self._wf is not None:
            with contextlib.suppress(Exception):
                self._wf.close()
            self._wf = None
