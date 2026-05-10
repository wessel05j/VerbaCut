# AI Auto Clipper

Local-first video clipping powered by Whisper, Ollama, yt-dlp, and FFmpeg.

AI Auto Clipper takes long videos, transcribes them, asks a local AI model to find useful moments, and exports those moments as separate clips.

## Features

- Local transcription with Whisper.
- Local AI clip selection through Ollama.
- YouTube link and channel queue support through yt-dlp.
- Resumable runs for interrupted processing.
- FFmpeg-based clip extraction with MP4 output and MKV fallback.
- Terminal setup wizard and dashboard.

## Requirements

- Python 3.10 or newer
- Git
- FFmpeg available in `PATH`
- Ollama running locally

## Quick Start

```powershell
git clone https://github.com/wessel05j/AI_Auto_clipper.git
cd AI_Auto_clipper
.\run.bat
```

Alternative Windows launcher:

```powershell
.\run.ps1
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
python setup_env.py --torch auto
python main.py
```

## Usage

1. Run the app.
2. Complete the setup wizard if no config exists.
3. Add videos to `input/`, or add YouTube links/channels in the dashboard.
4. Start clipping from the dashboard.
5. Collect exported clips from `output/`.

Processed source videos move to `temp/`. Runtime config lives in `config/config.json`.

## Project Map

```text
core/       clipping engine, AI pipeline, YouTube handling, FFmpeg extraction
ui/         setup wizard and terminal dashboard
utils/      config validation, logging, hardware/model helpers
input/      source videos
output/     exported clips
temp/       processed source archive
```

## Verify

```powershell
python -m compileall -q main.py core ui utils setup_env.py tests
python -m unittest discover -s tests
```

## Credits

Created by Erich Johannes Wessel.

## License

Apache License 2.0. See `LICENSE`.
