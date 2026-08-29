# SILIC 2

Sound identification and labeling pipeline for audio and video recordings.

## Setup

Use Python 3.10 or newer and install the Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`pydub` also requires FFmpeg to read formats such as MP3 and MP4. Put each model in
`model/<version>/` with both `best.pt` and `soundclass.csv`.

## Run

Command line, with either a recording or a one-level directory as input:

```powershell
python silic2.py --source sample --model v2026.1 --savepath result_silic
```

Add `--workers N` for multiprocessing and `--zip` to create a ZIP archive. Start the
desktop interface with:

```powershell
python silic2-ui.py
```

## Test

```powershell
python -m unittest discover -s tests -v
```

