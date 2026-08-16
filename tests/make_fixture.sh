#!/bin/bash
# Generate the speech fixture used by tests/smoke_mcp.py.
#
# Uses macOS `say` with an explicit English voice — the system default voice is
# whatever the user's locale is set to, and a Danish voice reading English text
# produces convincing-looking nonsense that makes the test useless.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p fixtures
say -v Samantha -o fixtures/speech.aiff \
  "The quick brown fox jumps over the lazy dog. This is a test of live transcription running on a Mac. We are measuring how long the model takes to process a segment of speech compared to the duration of the audio itself."
ffmpeg -y -loglevel error -i fixtures/speech.aiff -ar 16000 -ac 1 fixtures/speech.wav
rm -f fixtures/speech.aiff
echo "wrote $(pwd)/fixtures/speech.wav"
