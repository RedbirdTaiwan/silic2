# SILIC 2

Sound identification and labeling pipeline for audio and video recordings.

## Species & Sound class Coverage
- [./model/v2026.1](./model/v2026.1) , including 398 sound classes of 279 species, updated on Aug. 2026
  - 特別感謝吳昭頤、陳惇聿及楊懿如老師和海蟾蜍監測團隊提供蛙類聲音資料，以及翁國精老師及鄭佳馨同學提供珍貴的小鼯鼠及其他飛鼠的聲音資料。

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


