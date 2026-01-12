import sounddevice as sd
import numpy as np
import wave
import datetime
import shutil
import sys
import time
import signal
import whisper

# ---------- Configuration ----------
SAMPLE_RATE = 44100
REFRESH_HZ = 20
SILENCE_TIMEOUT = 60
SILENCE_THRESHOLD = 0.005
# -----------------------------------

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

# --- Device selection ---
devices = sd.query_devices()
input_devices = [d for d in devices if d["max_input_channels"] > 0]

print("\n🎙️  Available input devices:\n")
for i, d in enumerate(input_devices):
    print(f"[{i}] {d['name']} ({d['max_input_channels']} ch)")

sel = input("\nSelect device index [0]: ").strip()
selected = int(sel or 0)

device_info = input_devices[selected]
device_index = devices.index(device_info)

# --- AUTO-DETECT CHANNEL COUNT ---
CHANNELS = device_info["max_input_channels"]

if CHANNELS < 1:
    print("❌ This device has no input channels.")
    sys.exit(1)

print(f"\n🎚  Using device: {device_info['name']}")
print(f"🎚  Device supports {CHANNELS} channel(s)\n")

# --- File setup ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"recording_{timestamp}.wav"
print(f"💾 Writing live audio to {filename}\n")

wf = wave.open(filename, "wb")
wf.setnchannels(CHANNELS)
wf.setsampwidth(2)
wf.setframerate(SAMPLE_RATE)

term_width = shutil.get_terminal_size((80, 20)).columns
last_update = 0
last_activity = time.time()
start_time = time.time()

def bar(level, width=term_width//2 - 15):
    level = max(0, min(level, 1))
    filled = int(level * width)
    return "█" * filled + "-" * (width - filled)

def callback(indata, frames_count, time_info, status):
    global last_update, last_activity
    if status:
        print(status, file=sys.stderr)

    # Write raw audio to file
    wf.writeframes(np.int16(indata * 32767).tobytes())

    # Compute RMS loudness
    rms = np.sqrt(np.mean(indata**2, axis=0))
    mean_rms = float(np.mean(rms))

    # Silence detection
    if mean_rms > SILENCE_THRESHOLD:
        last_activity = time.time()

    # Update visualization
    now = time.time()
    if now - last_update >= 1.0 / REFRESH_HZ:
        elapsed = int(now - start_time)
        mm, ss = divmod(elapsed, 60)

        if CHANNELS == 1:
            # MONO DEVICE
            mono_bar = bar(min(mean_rms * 20, 1.0))
            sys.stdout.write(f"\r⏱️ {mm:02}:{ss:02}  🎧 [{mono_bar}] ")
        else:
            # STEREO DEVICE
            left = min(rms[0] * 20, 1.0)
            right = min(rms[1] * 20, 1.0)
            sys.stdout.write(
                f"\r⏱️ {mm:02}:{ss:02}  🎧 L:[{bar(left)}] R:[{bar(right)}] "
            )

        sys.stdout.flush()
        last_update = now

print("🎙️  Recording started — press Ctrl +C or stay silent for 60s to stop.\n")

try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        device=device_index,
        callback=callback,
    ):
        while not stop_flag:
            if time.time() - last_activity > SILENCE_TIMEOUT:
                print("\n🕓 No audio for 60 s — auto-stopping.")
                stop_flag = True
                break
            sd.sleep(100)

except KeyboardInterrupt:
    pass
finally:
    wf.close()
    print(f"\n⏹️  Recording finished and saved to {filename}")

# --- Whisper transcription ---
print("🧠 Transcribing with Whisper (MPS if available)…")
model = whisper.load_model("small", device="mps")
if autodetect:
    result = model.transcribe(filename, fp16=False)
    print(f"🌐 Detected language: {result.get('language','unknown')}")
else:
    result = model.transcribe(filename, fp16=False, language=lang)

txt = filename.replace(".wav", ".txt")
with open(txt, "w") as f:
    f.write(result["text"])

print(f"✅ Transcript saved as {txt}\n")
print("Preview:\n" + result["text"][:400] + "...")
