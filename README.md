# Audio Transcription Tool

A Python script for recording audio from your Mac's microphone and transcribing it using faster-whisper (a faster implementation of OpenAI's Whisper model) with MPS (Metal Performance Shaders) acceleration.

## Features

- Real-time audio recording with visual level meters
- Automatic silence detection (stops after 60 seconds of silence)
- Multi-channel audio support (mono and stereo)
- Language auto-detection or manual language selection
- faster-whisper transcription with MPS acceleration for Apple Silicon Macs (up to 4x faster than standard Whisper)
- Interactive device selection

## Prerequisites

- macOS (tested on macOS with Apple Silicon, but should work on Intel Macs too)
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

7. **Install faster-whisper:**
   ```bash
   pip install faster-whisper
   ```

   Note: `faster-whisper` is a faster, more memory-efficient implementation of OpenAI's Whisper model. It's installed separately via pip and provides significant performance improvements over the standard Whisper package.

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
   - Select your audio input device from the list
     - For microphone input: Select your built-in microphone or external mic
     - For application audio (Teams, Zoom, etc.): Select "BlackHole 2ch" (requires BlackHole installation - see Installation step 5)
   - Start speaking or play audio - the script will record audio
   - Press `Ctrl+C` to stop recording manually, or wait for 60 seconds of silence

4. **Output:**
   - Audio is saved as `recording_YYYYMMDD_HHMMSS.wav`
   - Transcript is saved as `recording_YYYYMMDD_HHMMSS.txt`
   - A preview of the transcript is displayed in the terminal

## Capturing Audio from Applications

To transcribe audio from applications like Microsoft Teams, Zoom, or any other app:

1. **Install BlackHole** (see Installation step 5 above)

2. **Set up audio routing:**
   - Open **System Preferences > Sound > Output**
   - Select **"BlackHole 2ch"** as your output device
   - In your application (e.g., Microsoft Teams), ensure audio output is set to "BlackHole 2ch"
   - **Note:** You won't hear audio through your speakers when BlackHole is selected. To hear audio while transcribing, you can use a multi-output device or use BlackHole 16ch with more routing options.

3. **Run the transcription script:**
   - When prompted to select an input device, choose **"BlackHole 2ch"**
   - The script will now capture all audio routed to BlackHole

4. **To restore normal audio:**
   - Change System Preferences > Sound > Output back to your speakers or headphones

## Configuration

You can modify these constants in `transcribe.py` to customize behavior:

- `SAMPLE_RATE = 44100` - Audio sample rate (Hz)
- `REFRESH_HZ = 20` - Visual update frequency
- `SILENCE_TIMEOUT = 60` - Seconds of silence before auto-stop
- `SILENCE_THRESHOLD = 0.005` - Silence detection sensitivity (lower = more sensitive)

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

### faster-whisper Model Download
The Whisper model ("small") will be downloaded automatically on first run when using faster-whisper. This may take a few minutes and requires an internet connection. The model is cached for future use. faster-whisper uses CTranslate2 for optimized inference, providing faster transcription speeds.

### MPS (Metal Performance Shaders) Issues
If you encounter MPS-related errors on Apple Silicon Macs, the script will fall back to CPU processing, which is slower but still functional.

### Permission Denied Errors
If you see permission errors when writing files, ensure you have write permissions in the current directory.

## System Requirements

- macOS 10.15 or later (for MPS support on Apple Silicon)
- Python 3.8+
- At least 2GB free disk space (for Whisper models)
- Microphone access permissions

## Notes

- Recordings are saved in the same directory as the script
- The script uses the "small" Whisper model by default (good balance of speed and accuracy)
- For better accuracy, you can modify the script to use "medium" or "large" models (slower but more accurate)
- faster-whisper provides significant performance improvements while maintaining the same accuracy as OpenAI's Whisper
- Audio files are saved as 16-bit WAV files

## License

[Add your license information here]

