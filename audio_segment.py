# audio_segment.py
import subprocess, tempfile, os
from pathlib import Path
import yt_dlp

from settings import CLIP_START, CLIP_DURATION, SAMPLE_RATE
from local_cache import paths_for_key, move_into_cache

def _download_to_temp(query: str) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="yt_"))
    outtmpl = str(tmpdir / "audio.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "default_search": "ytsearch1",
        "outtmpl": outtmpl,
        "quiet": True,
        "prefer_ffmpeg": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "m4a", "preferredquality": "0"}
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(query, download=True)
        m4a = tmpdir / "audio.m4a"
        return m4a if m4a.exists() else Path(ydl.prepare_filename(info))

def ensure_audio(query: str, key: str) -> Path:
    """Return cached audio path; download if missing."""
    p = paths_for_key(key)
    if p["audio"].exists():
        return p["audio"]
    src = _download_to_temp(query)
    return move_into_cache(src, p["audio"])

def ensure_clip(audio_path: Path, key: str) -> Path:
    """Return cached 45s clip; trim if missing."""
    p = paths_for_key(key)
    if p["clip"].exists():
        return p["clip"]
    clip_path = p["clip"]
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", str(CLIP_START),
        "-t", str(CLIP_DURATION),
        "-i", str(audio_path),
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-y", str(clip_path),
    ]
    subprocess.run(cmd, check=True)
    return clip_path
