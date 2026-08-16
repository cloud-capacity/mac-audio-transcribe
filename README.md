# Audio Transcription Tool

A Python script for recording audio from your Mac's microphone and transcribing it using OpenAI's Whisper model. Automatically detects your Mac's architecture and uses GPU acceleration (MPS) on Apple Silicon Macs or CPU processing on Intel Macs.

## Features

- Real-time audio recording with visual level meters
- **Live transcription** — see text appear while you are still recording
- Automatic silence detection (stops after 60 seconds of silence)
- Multi-channel audio support (mono and stereo)
- Language auto-detection or manual language selection
- OpenAI Whisper transcription with automatic device selection:
  - GPU acceleration (MPS) on Apple Silicon Macs for faster processing
  - CPU processing on Intel Macs (fully supported)
- Interactive device selection
- Automatic CPU detection and device selection, with CPU fallback if the GPU path fails
- **MCP server** — drive it from Claude or another agent (see [MCP Server](#mcp-server-use-it-from-claude-and-other-agents))

## Prerequisites

- macOS 10.15 or later (tested on both Apple Silicon and Intel Macs)
- Python 3.8 or higher
- pip (Python package installer)
- Homebrew (package manager for macOS)

### Installing Homebrew

If you don't have Homebrew installed, run this command in Terminal:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the on-screen instructions to complete the installation. After installation, you may need to add Homebrew to your PATH. The installer will provide instructions specific to your system.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd transcribe
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv .venv
   ```

3. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

4. **Install system dependencies via Homebrew:**
   ```bash
   brew install portaudio
   ```

   Note: PortAudio is required for the `sounddevice` package. Installing it via Homebrew ensures proper system-level support.

5. **Install BlackHole (Optional - for capturing audio from other applications):**
   ```bash
   brew install blackhole-2ch
   ```
   
   **Why install BlackHole?** BlackHole is a virtual audio driver that allows you to capture audio from other applications (like Microsoft Teams, Zoom, Spotify, etc.) instead of just your microphone. This is especially useful for transcribing meetings, calls, or any audio playing on your Mac.
   
   After installation, you'll need to:
   - Restart your Mac (or log out and back in) for BlackHole to appear as an audio device
   - In System Preferences > Sound > Output, select "BlackHole 2ch" as your output device
   - In your application (e.g., Microsoft Teams), set the output to "BlackHole 2ch"
   - When running the transcription script, select "BlackHole 2ch" as your input device
   
   For more channels (up to 16), you can install `blackhole-16ch` instead:
   ```bash
   brew install blackhole-16ch
   ```

6. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   This will install OpenAI Whisper and all required dependencies. The script will automatically detect your Mac's architecture (Intel or Apple Silicon) and select the appropriate compute device.

## Usage

1. **Activate your virtual environment** (if not already active):
   ```bash
   source .venv/bin/activate
   ```

2. **Run the script:**
   ```bash
   python transcribe.py
   ```

3. **Follow the prompts:**
   - Choose language detection mode (auto-detect recommended or manual)
   - Choose transcription mode:
     - **1 — Record first, transcribe afterwards:** quiet while recording, one accurate pass at the end
     - **2 — Live:** text appears on screen as you speak (see [Live Transcription](#live-transcription))
   - Select your audio input device from the list
     - For microphone input: Select your built-in microphone or external mic
     - For app or browser audio (YouTube, Teams, Zoom): Select "BlackHole 2ch", or your Aggregate Device to also capture your own voice — see [Capturing Audio from Applications](#capturing-audio-from-applications)
   - Start speaking or play audio - the script will record audio
   - Press `Ctrl+C` to stop recording manually, or wait for 60 seconds of silence

4. **Output:**
   - Audio is saved as `recording_YYYYMMDD_HHMMSS.wav`
   - Transcript is saved as `recording_YYYYMMDD_HHMMSS.txt`
   - In live mode, the running preview is also kept as `recording_YYYYMMDD_HHMMSS.live.txt`
   - A preview of the transcript is displayed in the terminal

## Live Transcription

Live mode shows text while you are still recording, instead of making you wait until the end.

It uses **two models**, because the model that is accurate enough for the final transcript is too
slow to keep up with a microphone in real time:

| Stage | Model on Intel (CPU) | When it runs |
|-------|----------------------|--------------|
| Live preview | `base` | during recording, one chunk at a time |
| Final transcript | `small` | once, over the whole WAV, after you stop |

Both come from `MODEL_DEFAULTS` in `audio_core.py`, which is keyed on the compute
device — Apple Silicon gets its own row (see [Configuration](#configuration)).

The saved `.txt` always comes from the accurate final pass, so live mode never costs you quality.
The live text is saved separately as `.live.txt`.

**How the audio is split.** Sending Whisper fixed-size chunks cuts words in half, so the script
instead cuts on natural pauses: a chunk ends after `LIVE_SILENCE_SPLIT` seconds of silence, and if
you talk non-stop it is forced to cut at `LIVE_MAX_SEGMENT` seconds. Each chunk is given the tail of
the previous transcript as context, and repeated words at the seams are removed automatically.

**What to expect.** Text appears roughly a second after you pause. Measured on a 6-core Intel
i5-10500 (CPU only), `base` transcribes a chunk in 0.4–1.3 s — about 7–14x faster than real time —
so it keeps up comfortably. If it ever falls behind, a `⏳ +12s` indicator appears next to the level
meter; no audio is lost, the preview just trails, and the final pass is unaffected.

**Note:** live mode records at 16 kHz when the device supports it (Whisper's own rate, so nothing is
lost and the WAV is ~3x smaller). Fixed-rate devices such as BlackHole keep their own rate and are
resampled in memory. The `base` model (~145 MB) downloads automatically the first time you use live
mode.

## Capturing Audio from Applications

Sending your output straight to BlackHole works, but you go deaf while recording —
BlackHole is a virtual cable, not a speaker. The fix is a **Multi-Output Device**, which
sends audio to BlackHole *and* your speakers at once.

There are two recipes, and it is worth being clear which one you need: capturing what a
**video or a browser plays** takes one virtual device; capturing **both sides of a call**
takes two, because your own voice never passes through BlackHole.

Both are built in **Audio MIDI Setup** (`Lyd- og MIDI-opsætning`) with the **+** button
at the bottom left.

### Recipe A — system audio (YouTube, browser, any app)

Create a **Multi-Output Device**:

- tick your real output (speakers or headphones) **and** BlackHole 2ch
- put the real output **first** — the top device is the clock master
- tick **Drift Correction** on BlackHole only, never on the master
- name it something like `Meeting Out`

Then set **System Settings → Sound → Output** to it, and record from **BlackHole 2ch**.
Most apps, including Chrome, have no per-app output picker and simply follow the system
default; a tab already playing may need reloading to notice the change.

Headphones are optional here — nothing is recording your microphone, so there is no echo
path to worry about.

### Recipe B — both sides of a meeting (Teams, Zoom)

BlackHole only ever carries the *far* end. To get your own voice too, add an **Aggregate
Device** that merges BlackHole with your microphone:

- tick **BlackHole 2ch** and your microphone
- set **Clock Source** to the microphone — a physical device makes the better clock, and
  a Bluetooth one makes the worst
- tick **Drift Correction** on BlackHole
- name it `Meeting In`

That yields a 3-channel input: channels 1–2 are the far end, channel 3 is you.

| Setting | Value |
|---------|-------|
| App speaker (Teams/Zoom) | `Meeting Out` |
| App microphone | your real mic — **not** BlackHole |
| Script input device | `Meeting In` |

The script handles the 3-channel input unchanged: it auto-detects the channel count and
live mode averages them into the mono signal Whisper wants, so both sides land in one
transcript.

**Use headphones for this one.** On speakers your microphone re-records the far end, so
every remote voice appears twice in the transcript, slightly offset — it measurably hurts
the output.

### Verifying the routing

```bash
python check_audio.py "BlackHole"      # recipe A
python check_audio.py "Meeting In" 10  # recipe B
```

Play some audio (and speak, for recipe B) while it runs. It reports which channels carried
signal and, for a 3-channel aggregate, whether each side of the conversation arrived. Five
seconds here beats discovering a silent recording an hour later. Via the MCP server, the
same check is the `check_audio_setup` tool.

| Symptom | Cause |
|---------|-------|
| Nothing at all | system output isn't the Multi-Output device, or the wrong input was selected |
| Only your voice | the app's speaker isn't set to the Multi-Output device |
| Only the far end | the microphone isn't in the Aggregate Device |
| Remote voice doubled | you're on speakers, not headphones |
| Sides drift apart late in a long call | drift correction is off |

### Two things that will catch you out

**The volume keys stop working** when a Multi-Output Device is selected. That is a
CoreAudio limitation, not a misconfiguration — set the level on the sub-device inside
Audio MIDI Setup, or in the app.

**An app's own volume slider is applied before the audio reaches BlackHole.** Turning
YouTube's player down to listen quietly records a correspondingly weak signal. Leave the
app at full volume and control loudness on the sub-device instead.

When you're finished, set **Output** back to your normal speakers so the volume keys work
again.

## Configuration

You can modify these constants in `audio_core.py` to customize behavior (they are
shared by the CLI and the MCP server):

- `SAMPLE_RATE = 44100` - Audio sample rate (Hz); live mode drops to 16000 when the device allows it
- `REFRESH_HZ = 20` - Visual update frequency
- `SILENCE_TIMEOUT = 60` - Seconds of silence before auto-stop
- `SILENCE_THRESHOLD = 0.005` - Silence detection sensitivity (lower = more sensitive); also decides
  where live chunks are split

Live transcription:

- `MODEL_DEFAULTS` - which models to use, keyed on the compute device. The `cpu` row
  (`base` live, `small` final) is measured on an Intel i5-10500; the `mps` row is a
  starting point that should be re-benchmarked on Apple Silicon. Drop the live model
  to `"tiny"` if it can't keep up
- `LIVE_SILENCE_SPLIT = 0.4` - Seconds of silence that ends a chunk
- `LIVE_MIN_SEGMENT = 1.0` - Shortest chunk that will be transcribed
- `LIVE_MAX_SEGMENT = 8.0` - Forced cut when someone talks non-stop; this is the worst-case delay
  before text appears
- `LIVE_OVERLAP = 0.25` - Seconds of audio carried into the next chunk so cut-off words survive

## Troubleshooting

### Homebrew Not Found
If you get a "command not found: brew" error, you need to install Homebrew first. See the Prerequisites section above.

### PortAudio Issues
If you get errors related to PortAudio when installing `sounddevice`, ensure PortAudio is installed via Homebrew:
```bash
brew install portaudio
pip install --upgrade sounddevice
```

### No Audio Input Devices Found
- Check System Preferences > Security & Privacy > Microphone
- Ensure your application has microphone permissions
- Try running the script again after granting permissions
- If using BlackHole, ensure it's installed and you've restarted your Mac after installation

### BlackHole Not Appearing as Audio Device
- After installing BlackHole, you must restart your Mac (or log out and back in) for it to appear
- Verify installation: `brew list | grep blackhole`
- Check System Preferences > Sound to see if BlackHole appears in the list
- If it still doesn't appear, try reinstalling: `brew reinstall blackhole-2ch`

### Whisper Model Download
The Whisper model ("small") will be downloaded automatically on first run. This may take a few minutes and requires an internet connection. The model is cached for future use. Live mode additionally downloads the smaller "base" model (~145 MB) the first time it runs.

### Live Transcription Falls Behind
If the `⏳ +Ns` indicator next to the level meter keeps growing, the machine can't transcribe as fast
as it records. Set the `live` entry for your device in `MODEL_DEFAULTS` (`audio_core.py`) to
`"tiny"`. Nothing is lost when this happens — the audio is still recorded in full and the final
transcript is unaffected.

### "Audio buffer overrun" Warning After Recording
The audio input queue overflowed, usually because the machine was busy, so the WAV may have small
gaps. Try a smaller live model in `MODEL_DEFAULTS`, or use mode 1 (record first, transcribe
afterwards).

### Live Text Looks Worse Than the Final Transcript
Expected. The live preview runs a smaller model on short chunks with no knowledge of what comes
next; a word may be cut at a chunk boundary. The saved `.txt` comes from the accurate pass over the
complete recording — use `.live.txt` only as a running preview.

### Device Selection
The script automatically detects your Mac's CPU architecture:
- **Apple Silicon Macs**: Uses GPU acceleration (MPS - Metal Performance Shaders) for faster transcription
- **Intel Macs**: Uses CPU processing (fully supported and functional)

You'll see output like:
```
💻 Detected CPU: Intel Core i5-xxxx
⚙️  Using CPU (MPS not available on Intel Macs)
```

or

```
💻 Detected CPU: Apple M1 Pro
🚀 Using GPU acceleration (MPS)
```

### MPS (Metal Performance Shaders) Issues
On Apple Silicon, if loading a model on MPS fails — or if the *first decode* fails, which is
the more common shape, since Whisper's decoder has historically hit unimplemented MPS ops well
after the model loads cleanly — the model is reloaded on the CPU and the run continues. The
fallback is reported rather than silent: the CLI prints the reason, and the MCP tools return it
in a `note` field alongside the `compute_device` actually used. Slower, but functional.

### Permission Denied Errors
If you see permission errors when writing files, ensure you have write permissions in the current directory.

## System Requirements

- macOS 10.15 or later
- Python 3.8+ for the CLI; **Python 3.12** for the MCP server (see [Setup](#setup))
- At least 2GB free disk space (for Whisper models)
- Microphone access permissions
- **Apple Silicon Macs**: MPS (Metal Performance Shaders) support for GPU acceleration
- **Intel Macs**: CPU processing (fully supported)

## Notes

- Recordings are saved in the same directory as the script
- The final transcript uses the "small" Whisper model by default (good balance of speed and accuracy); live previews use "base"
- For better accuracy, set the `final` entry in `MODEL_DEFAULTS` to "medium" or "large" (slower but more accurate)
- The script automatically detects your Mac's architecture and selects the optimal compute device
- Apple Silicon Macs benefit from GPU acceleration (MPS), while Intel Macs use CPU processing
- Audio files are saved as 16-bit WAV files

## MCP Server (use it from Claude and other agents)

`mcp_server.py` exposes the recorder and transcriber over the
[Model Context Protocol](https://modelcontextprotocol.io), so Claude can list your
audio devices, start and stop recordings, and transcribe files.

### Tools

| Tool | What it does |
|------|--------------|
| `list_audio_devices` | Input devices with channel counts and whether they accept 16 kHz. Call this first — device *names* are stable across machines, indices are not |
| `start_recording` | Starts recording and returns a `session_id` immediately. Accepts a device name substring, language, live on/off, and a model override |
| `get_status` | Elapsed time, input levels, how far the transcriber is behind, and the transcript so far |
| `stop_recording` | Stops and returns the live transcript plus the `.wav` path |
| `transcribe_file` | Transcribes any existing audio file. Returns inline if it finishes within `wait_seconds` (default 60), otherwise a session to poll |
| `check_audio_setup` | Listens briefly and reports which channels carry signal — diagnoses the BlackHole/Aggregate routing above |

Recording is a **session** rather than one blocking call, because a `record_and_transcribe`
tool that ran for half an hour would exceed every client's timeout. `stop_recording`
returns the *live* transcript; for the more accurate pass, call `transcribe_file` on
the `.wav` path it hands back.

### Setup

The MCP SDK needs Python ≥3.10 while the CLI's venv is on 3.9, so the server gets its
own environment. **Python 3.12 is the only version that works on both an Intel and an
Apple Silicon Mac**: torch's last x86 macOS build (2.2.2) ships wheels up to cp312,
and on arm64 the current torch covers 3.10–3.14.

```bash
brew install python@3.12
python3.12 -m venv .venv-mcp
.venv-mcp/bin/pip install -r requirements-mcp.txt
```

**Run this on each machine separately.** `.venv-mcp` holds platform-specific binaries
(x86 torch 2.2.2 vs arm64 torch 2.13), so a venv synced between machines via git,
iCloud or Dropbox will break on one of them. It is in `.gitignore` for that reason.
`torch` is deliberately unpinned so each machine resolves its own wheel. The other
three constraints in `requirements-mcp.txt` all carry `platform_machine == "x86_64"`
markers, so they apply on the Intel Mac and are a no-op on Apple Silicon:

| Pin | Why it exists |
|-----|---------------|
| `numpy<2` | torch 2.2.2 was compiled against NumPy 1.x. Under NumPy 2 every `torch.from_numpy` raises `RuntimeError: Numpy is not available`, which breaks Whisper entirely — and it surfaces only as a warning at import, then fails at the first tensor conversion |
| `llvmlite<=0.45.1` | later releases publish arm64-only macOS wheels, so pip falls back to an sdist and building llvmlite from source needs a full LLVM toolchain |
| `numba<=0.62.1` | same, and it must stay compatible with the llvmlite cap |

Every one of these looks like stale pinning. Don't remove one without installing on
**both** machines — each was found by an install that failed or a runtime that broke.

### Registering with Claude Desktop

Add an `mcpServers` entry to `~/Library/Application Support/Claude/claude_desktop_config.json`,
keeping any keys already in the file. Paths are absolute and differ per machine:

```json
{
  "mcpServers": {
    "mac-audio-transcribe": {
      "command": "/Users/you/Development/mac-audio-transcribe/.venv-mcp/bin/python",
      "args": ["/Users/you/Development/mac-audio-transcribe/mcp_server.py"]
    }
  }
}
```

Restart Claude Desktop afterwards. **macOS attributes microphone access to the app that
launched the server**, so the permission prompt comes from Claude Desktop, not Terminal —
grant it on the first recording.

### Notes

- Recording only ever starts on an explicit `start_recording` call. Sessions stop
  themselves after `max_seconds` (default 3600) or 60 seconds of silence, so a
  forgotten session can't record indefinitely.
- Every response includes the `.wav` path, so recordings are easy to find and delete.
  Set `TRANSCRIBE_OUTPUT_DIR` to write them somewhere other than the repo.
- One recording at a time, and one Whisper pass at a time — concurrent decodes just
  contend for the same cores and finish slower.

## Project layout

| File | Purpose |
|------|---------|
| `audio_core.py` | Recording and transcription logic shared by both front-ends. Never prints — stdout belongs to the terminal meter in one case and the JSON-RPC transport in the other |
| `transcribe.py` | The interactive CLI |
| `mcp_server.py` | The MCP server |
| `check_audio.py` | Per-channel level probe (CLI, and backs `check_audio_setup`) |
| `tests/harness.py` | Runs `transcribe.py` against a stubbed device and a stubbed Whisper. Guards the segmentation boundaries — run it after any change to the live path |
| `tests/smoke_mcp.py` | Drives `mcp_server.py` over real stdio JSON-RPC: handshake, all six tools, a recording cycle, and that stdout stays clean |
| `tests/make_fixture.sh` | Generates the speech fixture `smoke_mcp.py` needs |

### Tests

```bash
python tests/harness.py 16000 1 2   # live, device accepts 16 kHz
python tests/harness.py 44100 1 2   # live, exercises the resampler
python tests/harness.py 44100 2 1   # classic record-then-transcribe
```

Expected boundaries in live mode: `2.40 / 4.25 / 8.01 / 8.01 / 8.01 / 2.97 / 2.75` s,
totalling 36.40 s. These run on either venv and need no microphone or model download —
both `sounddevice` and `whisper` are stubbed.

```bash
./tests/make_fixture.sh                     # once
.venv-mcp/bin/python tests/smoke_mcp.py     # needs the 3.12 venv
```

The smoke test uses real models and records briefly from BlackHole, which is silent unless
audio is routed into it — so it needs no microphone either. Its most important assertion is
that **every byte the server writes to stdout parses as JSON-RPC**: stdout is the transport,
so one stray `print` anywhere in the dependency tree breaks the server for every client.
That is also why `audio_core.py` never prints and every Whisper call passes `verbose=None` —
`verbose=False` *enables* Whisper's progress bar, and any non-`None` value enables its
language print.

## License

[Add your license information here]

