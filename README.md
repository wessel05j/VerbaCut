# VerbaCut

Local-first transcript clipping for long-form video.

VerbaCut downloads or reads source videos, transcribes speech with Whisper, asks a local Ollama model to find strong moments, and exports those moments as ready-to-review clips with FFmpeg.

## Why It Exists

Manual clipping is slow because you have to watch, mark, trim, and export by hand. VerbaCut turns that into a local pipeline:

```text
video -> transcript -> AI moment selection -> clip export -> review
```

Your videos and prompts stay on your machine. Ollama runs the AI model locally.

## Highlights

- Local Whisper transcription.
- Local AI clip selection through Ollama.
- YouTube links and channel queues through yt-dlp.
- Resumable runs for interrupted processing.
- FFmpeg MP4 export with MKV fallback.
- Terminal setup wizard and dashboard.
- Automatic Ollama startup when Ollama is installed but not running.
- Windows `verbacut` terminal shortcut installed on launch.

## Requirements

Install these first:

- Python 3.10 or newer
- Git

VerbaCut checks the rest during setup:

- FFmpeg and FFprobe
- Ollama
- Python packages from `requirements.txt`
- Torch CPU/CUDA mode based on your system

When possible, the setup helper offers to install missing FFmpeg/Ollama tooling with your platform package manager. On Windows that means `winget`; on macOS that means Homebrew; on Linux that means the detected package manager.

## One-Shot Setup

Windows PowerShell:

```powershell
git clone https://github.com/wessel05j/VerbaCut.git; cd VerbaCut; .\run.bat
```

Windows PowerShell alternative:

```powershell
git clone https://github.com/wessel05j/VerbaCut.git; cd VerbaCut; .\run.ps1
```

macOS/Linux Bash:

```bash
git clone https://github.com/wessel05j/VerbaCut.git && cd VerbaCut && bash run.sh
```

Those commands create a virtual environment, prepare dependencies, check external tools, start the app, and launch the setup wizard if no config exists.

## Launch Later

From the project folder:

```powershell
.\run.bat
```

or:

```bash
bash run.sh
```

On Windows, after the first launch opens a new terminal, you can start VerbaCut from anywhere:

```powershell
verbacut
```

If automatic shortcut setup fails, run:

```powershell
.\launcher\install-command.ps1
```

## First Run

1. Start VerbaCut.
2. Complete the setup wizard.
3. Choose the local Ollama model.
4. Set your clipping goal, output folder, and processing preferences.
5. Add videos to `input/` or add YouTube links/channels in the dashboard.
6. Start clipping.
7. Review exported clips in `output/`.

Processed source videos move to `temp/`. Runtime config lives in `config/config.json`.

## Folder Map

```text
core/       clipping engine, AI pipeline, YouTube handling, FFmpeg extraction
ui/         setup wizard and terminal dashboard
utils/      config validation, logging, hardware/model helpers
launcher/   platform launchers and terminal command wrappers
input/      source videos
output/     exported clips
temp/       processed source archive
config/     runtime configuration
logs/       application logs
```

## Troubleshooting

If Ollama is installed but not running, VerbaCut tries to start it before setup or clipping fails.

If setup repeats dependency installation every launch, run the launcher again after this version. The setup state now records the actual Torch mode that was installed, including CPU fallback.

If FFmpeg or Ollama is missing, run the launcher from an interactive terminal so the setup helper can show an install plan and ask before installing tools.

If the `verbacut` command is not found after first launch, open a new terminal. Windows only reads user PATH changes when a new terminal starts.

## Verify

```powershell
.\venv\Scripts\python.exe -m compileall -q main.py core ui utils setup_env.py tests
.\venv\Scripts\python.exe -m unittest discover -s tests
```

macOS/Linux:

```bash
venv/bin/python -m compileall -q main.py core ui utils setup_env.py tests
venv/bin/python -m unittest discover -s tests
```

## License

Apache License 2.0. See `LICENSE`.
