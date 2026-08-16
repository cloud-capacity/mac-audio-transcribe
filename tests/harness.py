"""Regression harness for transcribe.py's live-transcription path.

Runs the real script against a stubbed `sounddevice` and a stubbed `whisper`, so the
whole record -> segment -> transcribe -> shutdown flow is exercised deterministically
with no microphone, no model download, and no wall-clock dependency on inference.

What it protects: the segmentation boundaries. Whisper is replaced by a stub that
reports the duration of every segment handed to it, so any change to the silence
splitting, the LIVE_MAX_SEGMENT forced cut, or the LIVE_OVERLAP carry shows up as a
different sequence of durations.

usage:  python tests/harness.py <native_rate> <channels> [mode]
          native_rate 16000 -> device accepts 16 kHz (no resampling)
          native_rate 44100 -> device is fixed at 44.1 kHz (resample path)
          mode        1 = record then transcribe, 2 = live (default)

Known-good output (both rates, mode 2):
  2.40 / 4.25 / 8.01 / 8.01 / 8.01 / 2.97 / 2.75 s, total 36.40 s
"""
import io
import os
import runpy
import signal
import sys
import threading
import time
import types

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "transcribe.py")
sys.path.insert(0, REPO)  # so the script under test can import audio_core

NATIVE = int(sys.argv[1]) if len(sys.argv) > 1 else 16000
CHANNELS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
MODE = sys.argv[3] if len(sys.argv) > 3 else "2"
SPEED = 12.0  # generate audio this much faster than real time

seen = []  # (duration_seconds, kwargs) per transcribe call


# ---------------- fake sounddevice ----------------
fake_sd = types.ModuleType("sounddevice")
_devices = [
    {"name": "Fake Output", "max_input_channels": 0, "max_output_channels": 2,
     "default_samplerate": float(NATIVE)},
    {"name": "Fake Mic", "max_input_channels": CHANNELS, "max_output_channels": 0,
     "default_samplerate": float(NATIVE)},
]
fake_sd.query_devices = lambda: _devices


def check_input_settings(device=None, channels=None, dtype=None, samplerate=None, **kw):
    if samplerate is not None and int(samplerate) != NATIVE:
        raise ValueError("Invalid sample rate")


fake_sd.check_input_settings = check_input_settings
fake_sd.sleep = lambda ms: time.sleep(ms / 1000.0)

# (seconds, is_speech) — exercises: short utterance, longer utterance, a talker who
# never pauses (forces the max-length cut three times), and a final short burst.
SCENARIO = [(2.0, True), (1.0, False), (3.0, True), (1.0, False),
            (25.0, True), (1.5, False), (1.0, True), (1.0, False)]


class InputStream:
    def __init__(self, samplerate, channels, dtype, device, callback, **kw):
        self.rate = int(samplerate)
        self.channels = channels
        self.callback = callback
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _block(self, n, speech, phase):
        t = (np.arange(n) + phase) / self.rate
        if speech:
            sig = 0.25 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 1400 * t)
        else:
            sig = 0.0005 * np.sin(2 * np.pi * 50 * t)
        return np.repeat(sig.astype(np.float32)[:, None], self.channels, axis=1)

    def _run(self):
        bs = int(self.rate * 0.02)  # 20 ms blocks
        phase = 0
        for secs, speech in SCENARIO:
            for _ in range(int(secs / 0.02)):
                if self._stop.is_set():
                    return
                self.callback(self._block(bs, speech, phase), bs, None, None)
                phase += bs
                time.sleep(0.02 / SPEED)
        os.kill(os.getpid(), signal.SIGINT)

    def start(self):
        self._t.start()

    def stop(self):
        self._stop.set()

    def close(self):
        self._stop.set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *a):
        self.stop()


fake_sd.InputStream = InputStream
fake_sd.default = types.SimpleNamespace(device=(1, 0))
sys.modules["sounddevice"] = fake_sd


# ---------------- fake whisper ----------------
fake_whisper = types.ModuleType("whisper")


class FakeModel:
    def __init__(self, name):
        self.name = name

    def transcribe(self, audio, **kwargs):
        if isinstance(audio, str):  # the final full-file pass
            return {"text": "final full-file transcript", "language": "english"}
        dur = len(audio) / 16000.0
        seen.append((dur, kwargs))
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        time.sleep(0.05)
        return {"text": f"[seg {len(seen)}: {dur:.2f}s peak={peak:.2f}]",
                "language": "english"}


fake_whisper.load_model = lambda name, device=None: FakeModel(name)
sys.modules["whisper"] = fake_whisper

sys.stdin = io.StringIO("1\n" + MODE + "\n0\n")

try:
    runpy.run_path(SCRIPT, run_name="__main__")
except SystemExit as e:
    print(f"\n[harness] SystemExit: {e}")

print("\n[harness] segments handed to whisper:")
for i, (dur, kw) in enumerate(seen, 1):
    prompt = kw.get("initial_prompt")
    print(f"  {i}: {dur:6.2f}s  lang={kw.get('language')!r}  "
          f"verbose={kw.get('verbose')!r}  "
          f"prompt={'…' + prompt[-30:] if prompt else None!r}")

boundaries = " / ".join(f"{d:.2f}" for d, _ in seen)
print(f"[harness] BOUNDARIES: {boundaries}")
print(f"[harness] TOTAL: {sum(d for d, _ in seen):.2f}s "
      f"(scenario had {sum(s for s, _ in SCENARIO):.1f}s)")

# Whisper's tqdm bar and its language print are enabled by verbose=False and by any
# non-None verbose respectively — only verbose=None is silent. Guard against a
# regression that would corrupt both the terminal meter and the MCP stdio transport.
bad = [i for i, (_, kw) in enumerate(seen, 1) if kw.get("verbose") is not None]
print(f"[harness] VERBOSE_OK: {not bad}" + (f" (segments {bad} not None)" if bad else ""))
