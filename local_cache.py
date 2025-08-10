
from __future__ import annotations
import json, re, shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
from settings import CACHE_DIR
from pathlib import Path

# Ensure CACHE_DIR is a directory (fixes cases where a file named ".cache" exists)
if CACHE_DIR.exists() and not CACHE_DIR.is_dir():
    CACHE_DIR.unlink()  # remove the file
CACHE_DIR.mkdir(parents=True, exist_ok=True)




def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return re.sub(r"-{2,}", "-", text)

def key_for_spotify(track_id: str) -> str:
    return f"spotify_{track_id}"

def pretty_spotify_key(title: str, artist: str, track_id: str) -> str:
    human = slugify(f"{artist}-{title}")
    return f"spotify_{human}-{track_id[:8]}"  # keeps uniqueness


def key_for_query(title: str, artist: str) -> str:
    return f"ytq_{slugify(f'{title} {artist}')}"

def dir_for_key(key: str) -> Path:
    d = CACHE_DIR / key
    _ensure_dir(d)
    return d

def paths_for_key(key: str) -> Dict[str, Path]:
    root = dir_for_key(key)
    return {
        "root": root,
        "audio": root / "audio.m4a",      # raw download (or whatever yt-dlp outputs we move here)
        "clip":  root / "clip.wav",       # trimmed mono clip
        "vec":   root / "features.npy",   # numpy vector
        "meta":  root / "meta.json",      # metadata
    }

def exists(key: str) -> bool:
    return paths_for_key(key)["vec"].exists()

def load(key: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    p = paths_for_key(key)
    vec = np.load(p["vec"])
    meta = {}
    if p["meta"].exists():
        meta = json.loads(p["meta"].read_text())
    return vec, meta

def save(key: str, vec: np.ndarray, meta: Dict[str, Any]) -> None:
    p = paths_for_key(key)
    np.save(p["vec"], vec)
    p["meta"].write_text(json.dumps(meta, indent=2))

def move_into_cache(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dest.resolve():
        return dest
    if dest.exists():
        dest.unlink()
    shutil.move(str(src), str(dest))
    return dest
