"""Per-channel level probe: shows which channels of an input device receive audio.

Records nothing to disk. Useful for verifying a BlackHole + Aggregate Device setup,
where the far end arrives on one pair of channels and your microphone on another —
the main meter in transcribe.py only shows the first two, so a dead mic channel is
otherwise invisible.

usage:  python check_audio.py                # pick a device interactively
        python check_audio.py "Meeting In"   # match a device by name
        python check_audio.py "Meeting In" 8 # ...for 8 seconds
"""
import sys
import time

import numpy as np
import sounddevice as sd

import audio_core as core


def probe(device_spec=None, seconds=5.0, on_tick=None):
    """Listen for `seconds` and report which channels carried signal.

    on_tick, if given, is called repeatedly with the current per-channel RMS so a
    CLI can draw meters. Returns a plain dict suitable for JSON.
    """
    device = core.resolve_device(device_spec)
    channels = device["channels"]
    rate = device["default_samplerate"]

    peak = np.zeros(channels)
    live = np.zeros(channels, dtype=bool)
    level = np.zeros(channels)

    def callback(indata, frames, time_info, status):
        nonlocal level
        rms = np.sqrt(np.mean(indata.astype(np.float64) ** 2, axis=0))
        level = rms
        np.maximum(peak, rms, out=peak)
        live[rms > core.SILENCE_THRESHOLD] = True

    with sd.InputStream(device=device["index"], channels=channels,
                        dtype="float32", samplerate=rate, callback=callback):
        deadline = time.time() + seconds
        while time.time() < deadline:
            if on_tick:
                on_tick(level)
            time.sleep(0.05)

    detail = [{"channel": c + 1, "peak": round(float(peak[c]), 5),
               "signal": bool(live[c])} for c in range(channels)]
    result = {
        "device": device["name"],
        "channels": channels,
        "sample_rate": rate,
        "seconds": seconds,
        "channel_detail": detail,
    }

    # A 3+ channel device is almost certainly an Aggregate: BlackHole's stereo pair
    # carrying the far end, plus a mono microphone.
    if channels >= 3:
        far = bool(live[0] or live[1])
        near = bool(live[2:].any())
        result["far_end_captured"] = far
        result["mic_captured"] = near
        result["verdict"] = (
            "both sides captured — the aggregate is wired correctly" if far and near
            else "far end missing — check the app's output is the Multi-Output device"
            if near else
            "microphone missing — check the Aggregate Device membership" if far else
            "no signal at all — wrong device, or nothing is playing")
    else:
        result["verdict"] = ("signal detected" if live.any()
                             else "no signal — nothing reached this device")
    return result


def main():
    spec = sys.argv[1] if len(sys.argv) > 1 else None
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

    if spec is None:
        devices = core.list_input_devices()
        print("🎙️  Input devices:\n")
        for n, d in enumerate(devices):
            print(f"[{n}] {d['name']} ({d['channels']} ch)")
        sel = input("\nSelect device [0]: ").strip()
        spec = devices[int(sel or 0)]["index"]

    def draw(level):
        bars = []
        for c, v in enumerate(level):
            n = int(min(v * 20, 1.0) * 20)
            bars.append(f"ch{c+1}:[{'█' * n}{'-' * (20 - n)}]")
        sys.stdout.write("\r" + "  ".join(bars) + " ")
        sys.stdout.flush()

    result = probe(spec, seconds, on_tick=draw)

    print(f"\n\n🎚  {result['device']} — {result['channels']} ch @ "
          f"{result['sample_rate']} Hz\n")
    for ch in result["channel_detail"]:
        state = "signal seen" if ch["signal"] else "SILENT — nothing ever arrived"
        print(f"  ch{ch['channel']}: peak {ch['peak']:.4f}  {state}")
    print(f"\n{result['verdict']}")


if __name__ == "__main__":
    main()
