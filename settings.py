# settings.py
from pathlib import Path

# Base cache folder
CACHE_DIR = Path(".cache")

# Per-track folder layout: .cache/<key>/{audio.m4a, clip.wav, features.npy, meta.json}
CLIP_START = 30          # seconds
CLIP_DURATION = 45       # seconds
SAMPLE_RATE = 44100      # Hz
