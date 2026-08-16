"""Interactive recorder and transcriber.

All the recording and transcription logic lives in audio_core.py, which is shared
with mcp_server.py. This file owns the terminal UI: prompts, the level meter, and
the files written next to the recording.
"""
import datetime
import shutil
import signal
import sys
import time

import audio_core as core

stop_flag = False


def handle_sigint(sig, frame):
    global stop_flag
    stop_flag = True


signal.signal(signal.SIGINT, handle_sigint)

# --- Language setup ---
print("🌍 Transcription language:")
print("1️⃣  Auto-detect (recommended)")
print("2️⃣  Manually specify language")
choice = input("Select 1 or 2 [1]: ").strip() or "1"
if choice == "2":
    lang = input("Enter spoken language (e.g., english): ").strip().lower() or "english"
    autodetect = False
else:
    autodetect = True
    lang = None
print(f"🗣️  Mode: {'Auto-detect' if autodetect else lang.title()}\n")

# --- Transcription mode ---
print("⚡ Transcription mode:")
print("1️⃣  Record first, transcribe afterwards")
print("2️⃣  Live — show text while recording")
mode = input("Select 1 or 2 [1]: ").strip() or "1"
live_mode = mode == "2"
print(f"⚡ Mode: {'Live' if live_mode else 'Record then transcribe'}")

# --- Device selection ---
input_devices = core.list_input_devices()
if not input_devices:
    print("❌ No audio input devices found.")
    sys.exit(1)

print("\n🎙️  Available input devices:\n")
for i, d in enumerate(input_devices):
    print(f"[{i}] {d['name']} ({d['channels']} ch)")

sel = input("\nSelect device index [0]: ").strip()
selected = int(sel or 0)
device = input_devices[selected]
CHANNELS = device["channels"]

print(f"\n🎚  Using device: {device['name']}")
print(f"🎚  Device supports {CHANNELS} channel(s)")

SAMPLE_RATE = core.choose_sample_rate(device, live_mode)
if live_mode:
    print(f"🎚  Recording at {SAMPLE_RATE} Hz"
          + ("" if SAMPLE_RATE == core.WHISPER_RATE
             else " (resampled to 16 kHz for Whisper)"))
print()

# --- Compute device ---
cpu_brand, device_to_use = core.pick_compute_device()
models = core.models_for(device_to_use)

print(f"💻 Detected CPU: {cpu_brand}")
if device_to_use == "mps":
    print("🚀 Using GPU acceleration (MPS)")
else:
    print("⚙️  Using CPU (MPS not available on Intel Macs)")
print()

# The live model has to be resident before the stream opens — loading it takes
# a few seconds and would otherwise stall the first segments.
live_model = None
if live_mode:
    print(f"🧠 Loading live model ({models['live']}) on {device_to_use.upper()}…"
          " (first run downloads it)")
    live_model = core.load_model(models["live"], device_to_use)
    if live_model.fell_back:
        print(f"⚠️  {live_model.reason}")

# --- File setup ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"recording_{timestamp}.wav"
print(f"💾 Writing live audio to {filename}\n")

recorder = core.Recorder(device, SAMPLE_RATE, filename, live=live_mode,
                         language=lang, loaded_model=live_model)


def bar(level, width):
    level = max(0, min(level, 1))
    width = max(4, width)
    filled = int(level * width)
    return "█" * filled + "-" * (width - filled)


def drain_text():
    """Print finished transcript lines above the meter."""
    for kind, line in recorder.drain_text():
        prefix = "🌐 " if kind == "note" else ""
        sys.stdout.write("\r\033[K" + prefix + line + "\n")
        sys.stdout.flush()


def draw_meter():
    cols = shutil.get_terminal_size((80, 20)).columns
    mm, ss = divmod(int(recorder.elapsed), 60)

    lag_txt = ""
    if live_mode:
        lag = recorder.lag
        if lag > 3:
            lag_txt = f" ⏳ +{int(lag)}s"

    # Reserve the lag field's space even when it is empty, so the bars keep a
    # constant width instead of resizing whenever the indicator appears.
    width = cols // 2 - 15 - (10 if live_mode else 0)
    if CHANNELS == 1:
        # MONO DEVICE
        mono_bar = bar(min(recorder.mean_level * 20, 1.0), width)
        sys.stdout.write(f"\r⏱️ {mm:02}:{ss:02}  🎧 [{mono_bar}]{lag_txt} ")
    else:
        # STEREO DEVICE
        left = min(recorder.levels[0] * 20, 1.0)
        right = min(recorder.levels[1] * 20, 1.0)
        sys.stdout.write(
            f"\r⏱️ {mm:02}:{ss:02}  🎧 L:[{bar(left, width)}] R:[{bar(right, width)}]{lag_txt} "
        )
    sys.stdout.flush()


print("🎙️  Recording started — press Ctrl +C or stay silent for 60s to stop.\n")

try:
    recorder.start()
    while not stop_flag:
        if recorder.seconds_since_activity > core.SILENCE_TIMEOUT:
            print(f"\n🕓 No audio for {core.SILENCE_TIMEOUT} s — auto-stopping.")
            stop_flag = True
            break
        drain_text()
        draw_meter()
        time.sleep(1.0 / core.REFRESH_HZ)
except KeyboardInterrupt:
    pass
finally:
    if live_mode:
        print("\n⏳ Finishing live transcription…")
    recorder.stop(on_progress=drain_text)
    drain_text()
    print(f"\n⏹️  Recording finished and saved to {filename}")

if recorder.xrun_count:
    print(f"⚠️  {recorder.xrun_count} audio buffer overrun(s) — "
          "the recording may have small gaps.")

# --- Save the live transcript ---
live_txt = None
if live_mode and recorder.lines:
    live_txt = filename.replace(".wav", ".live.txt")
    with open(live_txt, "w") as f:
        f.write(recorder.transcript + "\n")
    print(f"📝 Live transcript saved as {live_txt}")

# --- Whisper transcription ---
# Ctrl+C should be able to abort the (slow) final pass now that recording is done.
signal.signal(signal.SIGINT, signal.SIG_DFL)

print(f"\n🧠 Transcribing with Whisper ({models['final']} model, "
      f"{device_to_use.upper()} device)…")
final_model = core.load_model(models["final"], device_to_use)
if final_model.fell_back:
    print(f"⚠️  {final_model.reason}")

try:
    result = core.transcribe_audio(final_model, filename, language=lang)
except KeyboardInterrupt:
    print("\n⏹️  Final transcription cancelled.")
    if live_txt:
        print(f"📝 The live transcript is still available at {live_txt}")
    sys.exit(1)

if autodetect:
    print(f"🌐 Detected language: {result.get('language','unknown')}")

txt = filename.replace(".wav", ".txt")
with open(txt, "w") as f:
    f.write(result["text"])

print(f"✅ Transcript saved as {txt}\n")
print("Preview:\n" + result["text"][:400] + "...")
