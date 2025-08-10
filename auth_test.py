# ===== Standard Library =====
import os
import json
import tempfile
import subprocess
import urllib.request

# ===== Third-Party Libraries =====
import numpy as np
import requests
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import yt_dlp
import essentia
import essentia.standard as es

# ===== Local Modules =====
from local_cache import exists, load, save, key_for_query, paths_for_key
from audio_segment import ensure_audio, ensure_clip
from pathlib import Path
from spotipy.oauth2 import SpotifyOAuth, CacheFileHandler  # CacheFileHandler is in recent Spotipy

import local_cache as cache


# =====================
# CONFIG & AUTH
# =====================
load_dotenv()



# Put Spotipy's token file somewhere separate from our .cache folder
SPOTIFY_CACHE_DIR = Path(".spotify_oauth")
SPOTIFY_CACHE_DIR.mkdir(exist_ok=True)
SPOTIFY_TOKEN_FILE = SPOTIFY_CACHE_DIR / "token.json"



sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
    redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI", "http://localhost:8888/callback"),
    scope="playlist-modify-public",
    cache_path=str(SPOTIFY_TOKEN_FILE),   # <- keeps Spotipy off .cache/
    open_browser=False                    # <- avoids auto-opening from WSL
))
user_id = sp.current_user()["id"]

# =====================
# YOUTUBE HELPERS
# =====================


def download_youtube_audio(query: str) -> str:
    """
    Downloads best audio for the first YouTube search result as m4a.
    Returns the path to the downloaded file.
    """
    tmpdir = tempfile.mkdtemp(prefix="yt_")
    outtmpl = os.path.join(tmpdir, "audio.%(ext)s")
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
        # After postprocess the file should be audio.m4a
        m4a_path = os.path.join(tmpdir, "audio.m4a")
        if os.path.exists(m4a_path):
            return m4a_path
        # fallback: whatever filename yt-dlp produced
        return ydl.prepare_filename(info)


def make_clip(src_path: str, start_sec: int = 30, duration_sec: int = 45, sr: int = 44100) -> str:
    """
    Creates a trimmed mono WAV clip for fast analysis and returns its path.
    """
    fd, clip_path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", str(start_sec),           # seek quickly to ~30s
        "-t", str(duration_sec),         # keep 45s
        "-i", src_path,
        "-ac", "1",                      # mono
        "-ar", str(sr),                  # 44.1kHz
        "-y", clip_path
    ]
    subprocess.run(cmd, check=True)
    return clip_path

def analyze_from_youtube(title_and_artist: str, *, key=None, title=None, artist=None):
    """
    Returns a vector for 'title_and_artist', using cache if available.
    If `key` is provided (e.g., spotify_<track_id>), that key is used for caching.
    Otherwise we fall back to a slug from the title/artist/query.
    """
    # Choose cache key
    import essentia
    import essentia.standard as es
    if key:
        cache_key = key
    elif title is not None and artist is not None:
        cache_key = cache.key_for_query(title, artist)
    else:
        cache_key = cache.key_for_query(title_and_artist, "")  # slug from whole query

    # 1) Cached vector?
    if cache.exists(cache_key):
        vec, _meta = cache.load(cache_key)
        print("✅ Cached vector hit")
        return vec

    # 2) Ensure audio + clip in cache
    print(f"🔎 Downloading (or using cached) audio… for {title_and_artist}")
    audio_path = ensure_audio(title_and_artist, cache_key)
    print("✂️  Trimming (or using cached) clip…")
    clip_path = ensure_clip(audio_path, cache_key)

    # 3) Run Essentia
    extractor = es.MusicExtractor(
        lowlevelStats=['mean', 'stdev'],
        rhythmStats=['mean'],
        tonalStats=['mean'],
    )
    features, _ = extractor(str(clip_path))
    vec = build_feature_vector(features)

    # Optional: quick log
    names = set(features.descriptorNames())
    bpm = float(features['rhythm.bpm']) if 'rhythm.bpm' in names else 0.0
    print(f"BPM: {bpm:.1f} | Vector length: {len(vec)}")

    # 4) Save vector + meta
    meta = {
        "title_and_artist": title_and_artist,
        "bpm": bpm,
        "clip": str(cache.paths_for_key(cache_key)["clip"]),
        "audio": str(cache.paths_for_key(cache_key)["audio"]),
        "dims": int(vec.shape[0]),
    }
    cache.save(cache_key, vec, meta)
    print("💾 Cached vector saved")
    return vec


# =====================
# ESSENTIA HELPERS
# =====================

def analyze_track(track_id):
    # Get Spotify preview URL
    track = sp.track(track_id)
    preview_url = track.get("preview_url")
    if not preview_url:
        print("⚠️ No preview available for this track.")
        return

    # Download preview to temp file
    fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    urllib.request.urlretrieve(preview_url, tmp_path)

    # Run Essentia's MusicExtractor
    features, _ = es.MusicExtractor()(tmp_path)

    # Clean up file
    os.remove(tmp_path)

    # Print a couple of interesting features
    print("BPM:", features['rhythm.bpm'])
    print("Spectral Centroid Mean:", features['lowlevel.spectral_centroid.mean'])


# =====================
# COSINE SIMILARITY
# =====================

def build_feature_vector(features) -> np.ndarray:
    names = set(features.descriptorNames())
    vec = []

    # 1) MFCC means (13)
    if 'lowlevel.mfcc.mean' in names:
        vec.extend(list(map(float, features['lowlevel.mfcc.mean'])))
    else:
        vec.extend([0.0] * 13)

    # 2) Spectral centroid (mean)
    vec.append(float(features['lowlevel.spectral_centroid.mean'])
               if 'lowlevel.spectral_centroid.mean' in names else 0.0)

    # 3) Spectral flux (mean)
    vec.append(float(features['lowlevel.spectral_flux.mean'])
               if 'lowlevel.spectral_flux.mean' in names else 0.0)

    # 4) BPM
    vec.append(float(features['rhythm.bpm']) if 'rhythm.bpm' in names else 0.0)

    # 5) NEW: HPCP (12-D chroma) means
    if 'tonal.hpcp.mean' in names:
        vec.extend(list(map(float, features['tonal.hpcp.mean'])))
    else:
        vec.extend([0.0] * 12)

    # 6) NEW: Onset rate (mean)
    # MusicExtractor with rhythmStats=['mean'] exposes 'rhythm.onset_rate.mean'
    # If absent, fall back to scalar 'rhythm.onset_rate'
    if 'rhythm.onset_rate.mean' in names:
        vec.append(float(features['rhythm.onset_rate.mean']))
    elif 'rhythm.onset_rate' in names:
        vec.append(float(features['rhythm.onset_rate']))
    else:
        vec.append(0.0)

    return np.asarray(vec, dtype=float)

def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# =====================
# SPOTIFY HELPERS
# =====================
def get_user_input_song():
    song_title = input("Enter a song: ")
    results = sp.search(q=song_title, type="track", limit=1)

    if not results["tracks"]["items"]:
        print("No song found. Try again.")
        return None

    track = results["tracks"]["items"][0]
    track_id = track["id"]
    track_name = track["name"]
    artist_name = track["artists"][0]["name"]
    artist_id = track["artists"][0]["id"]

    return track_id, track_name, artist_name, artist_id

def get_artist_genres(artist_id):
    artist = sp.artist(artist_id)
    genres = artist.get("genres", [])
    return genres[:2] if genres else []



def get_spotify_track_metadata(track_id):
    track = sp.track(track_id)
    name = track["name"]
    artist = track["artists"][0]["name"]
    return name, artist

def get_track_pool(genre, limit_per_playlist=30):
    track_pool = []
    playlists = sp.search(q=genre, type="playlist", limit=3)

    for p in playlists["playlists"]["items"]:
        if not p or "id" not in p:  # Skip if playlist is None or missing ID
            continue

        playlist_id = p["id"]
        playlist_data = sp.playlist_tracks(playlist_id, limit=limit_per_playlist)
        for item in playlist_data["items"]:
            track = item.get("track")
            if track and track.get("id"):
                track_pool.append((track["id"], track["name"], track["artists"][0]["name"]))
    return track_pool

def get_genre_playlists(genres, per_genre=2):
    """Return a list of playlist IDs for the given genres."""
    pids = []
    for g in genres:
        res = sp.search(q=g, type="playlist", limit=per_genre)
        pids.extend([p["id"] for p in res["playlists"]["items"] if p and p.get("id")])
    return pids

def get_playlist_tracks(playlist_id, limit=20):
    """Return track dicts: {id, name, artist, preview_url} from one playlist."""
    tracks = []
    data = sp.playlist_tracks(playlist_id, limit=limit)
    for item in data["items"]:
        t = item.get("track")
        if not t or not t.get("id"):
            continue
        tracks.append({
            "id": t["id"],
            "name": t["name"],
            "artist": t["artists"][0]["name"],
            "preview_url": t.get("preview_url"),
        })
    return tracks

def build_candidate_pool(genres, per_genre=2, per_playlist=20, seed_id=None, cap=60):
    """
    Get up to `cap` unique candidate tracks from a few genre playlists.
    Skips the seed track id.
    """
    pids = get_genre_playlists(genres, per_genre=per_genre)
    pool, seen = [], set([seed_id] if seed_id else [])
    for pid in pids:
        for t in get_playlist_tracks(pid, limit=per_playlist):
            tid = t["id"]
            if tid and tid not in seen:
                seen.add(tid)
                pool.append(t)
                if len(pool) >= cap:
                    return pool
    return pool

def create_playlist(top_scored, name, public=True, description="This playlist was made using a python script"):
    """
    Creates a Spotify playlist and adds tracks.
    Accepts either:
      - list of tuples: (track_id, name, artist, score)
      - list of dicts:  {"id": ..., "name": ..., "artist": ..., "score": ...}
    Returns a dict with playlist metadata for the API to return.
    """
    # 1) Create the playlist
    user_id = sp.current_user()["id"]
    playlist = sp.user_playlist_create(
        user=user_id,
        name=name,
        public=public,
        description=description
    )

    # 2) Normalize inputs and dedupe while preserving order
    seen = set()
    track_ids = []
    for item in top_scored:
        if isinstance(item, dict):
            tid = item.get("id")
        else:
            tid = item[0] if item else None
        if tid and tid not in seen:
            seen.add(tid)
            track_ids.append(tid)

    # 3) Add items in chunks of 100 (API limit)
    for i in range(0, len(track_ids), 100):
        sp.playlist_add_items(playlist_id=playlist["id"], items=track_ids[i:i+100])

    url = playlist["external_urls"]["spotify"]
    count = len(track_ids)
    print(f"✅ Playlist '{name}' created with {count} tracks. {url}")

    # 4) Return metadata so the API can include it in JSON
    return {"id": playlist["id"], "url": url, "count": count, "name": name, "public": public}
    
    
# =====================
# FAST API HELPERS
# =====================


from typing import Optional

def _parse_track_id(seed: str) -> Optional[str]:
    if not isinstance(seed, str):
        return None
    if seed.startswith("spotify:track:"):
        return seed.split(":")[-1]
    if "open.spotify.com/track/" in seed:
        return seed.split("track/")[1].split("?")[0]
    if seed.isalnum() and 10 <= len(seed) <= 40:
        return seed
    return None

def _resolve_seed(seed_input: str):
    """Return (seed_id, seed_name, seed_artist, seed_artist_id) from name/URL/URI/ID."""
    tid = _parse_track_id(seed_input)
    if tid:
        t = sp.track(tid)
        return t["id"], t["name"], t["artists"][0]["name"], t["artists"][0]["id"]

    res = sp.search(q=seed_input, type="track", limit=1, market="GB")
    items = res.get("tracks", {}).get("items", [])
    if not items:
        raise ValueError(f"No Spotify match for: {seed_input!r}")
    t = items[0]
    return t["id"], t["name"], t["artists"][0]["name"], t["artists"][0]["id"]

def recommend_from_seed(seed_input: str, top_k: int = 20):
    # 0) Resolve seed
    seed_id, seed_name, seed_artist, seed_artist_id = _resolve_seed(seed_input)
    print(f"[seed] {seed_name} — {seed_artist}  (id={seed_id}, artist_id={seed_artist_id})")

    # 1) Seed features
    seed_key = cache.key_for_spotify(seed_id)
    seed_vec = analyze_from_youtube(f"{seed_name} {seed_artist}",
                                    key=seed_key, title=seed_name, artist=seed_artist)
    if seed_vec is None:
        raise RuntimeError("Could not extract features for the seed track.")
    print(f"[seed] features OK: len={len(seed_vec)}")

    # 2) Genres
    genres = get_artist_genres(seed_artist_id)
    print(f"[genres] {genres}")
    if not genres:
        raise RuntimeError("No genres found for this artist.")

    # 3) Candidates
    candidates = build_candidate_pool(genres, per_genre=2, per_playlist=20,
                                      seed_id=seed_id, cap=60)
    print(f"[candidates] gathered={len(candidates)}")
    if not candidates:
        raise RuntimeError("No candidates found.")

    # 4) Score candidates
    scored = []
    attempted = 0
    succeeded = 0
    last_err = None

    for t in candidates:
        attempted += 1
        try:
            key = cache.key_for_spotify(t["id"])
            vec = analyze_from_youtube(f"{t['name']} {t['artist']}",
                                       key=key, title=t["name"], artist=t["artist"])
            if vec is None:
                last_err = "feature_extractor returned None"
                continue
            sim = cosine_similarity(seed_vec, vec)
            scored.append({"id": t["id"], "name": t["name"], "artist": t["artist"], "score": float(sim)})
            succeeded += 1
        except Exception as e:
            last_err = str(e)
            continue

    print(f"[score] attempted={attempted} succeeded={succeeded} kept={len(scored)} last_err={last_err}")

    if not scored:
        raise RuntimeError(f"No features extracted for candidates. attempted={attempted} last_err={last_err}")

    # 5) Sort + top K
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def make_playlist_from_results(seed_title: str, scored_top):
    """
    scored_top: list of tuples (track_id, name, artist, score) OR list of dicts with those fields.
    Returns the playlist URL (string).
    """
    # normalize to a list of tuples
    normalized = []
    for item in scored_top:
        if isinstance(item, dict):
            normalized.append((item["id"], item["name"], item["artist"], item["score"]))
        else:
            normalized.append(item)
    name = f"PYTHON MADE - {seed_title}"
    # This function already creates the playlist and adds tracks:
    create_playlist(normalized, name)
    # If your create_playlist() prints the URL but doesn't return it,
    # you can make it return playlist['external_urls']['spotify'] instead.
    # For now, just return the name to confirm.
    return name  # replace with the actual URL if you modify create_playlist to return it

# =====================
# MAIN PROGRAM
# =====================


def main():
    # 1) Pick seed
    result = get_user_input_song()
    if result is None:
        return
    seed_id, seed_name, seed_artist, seed_artist_id = result
    print(f"🎵 Song: {seed_name}")
    print(f"🎤 Artist: {seed_artist}")

    # 2) Seed vector (cache key = spotify_<track_id>)
    print("\nAnalyzing seed audio (cached if available)…")
    seed_vec = analyze_from_youtube(
        f"{seed_name} {seed_artist}",
        key=cache.key_for_spotify(seed_id),
        title=seed_name,
        artist=seed_artist,
    )
    if seed_vec is None:
        print("⚠️ Could not extract features for the seed track.")
        return

    # 3) Genres → candidate pool
    genres = get_artist_genres(seed_artist_id)
    if not genres:
        print("⚠️ No genres found for this artist.")
        return
    print("🏷️ Genres:", ", ".join(genres))

    candidates = build_candidate_pool(
        genres,
        per_genre=2,
        per_playlist=20,
        seed_id=seed_id,
        cap=60,
    )
    print(f"📦 Candidates gathered: {len(candidates)}")

    if not candidates:
        print("⚠️ No candidates found.")
        return

    # 4) Score candidates
    scored = []
    for t in candidates:
        key = cache.key_for_spotify(t["id"])
        vec = analyze_from_youtube(f"{t['name']} {t['artist']}",
                                   key=key, title=t["name"], artist=t["artist"])
        if vec is None:
            continue
        sim = cosine_similarity(seed_vec, vec)
        scored.append((t["id"], t["name"], t["artist"], sim))

    if not scored:
        print("⚠️ No features extracted for candidates.")
        return

    # 5) Top 20
    scored.sort(key=lambda x: x[3], reverse=True)
    top = scored[:20]

    print("\n🏆 Top 20 similar tracks:")
    for tid, name, artist, sim in top:
        print(f"{sim:.3f}  {name} — {artist}")

    
    create_playlist(top, f"PYTHON MADE - {seed_name}")


if __name__ == "__main__":
    main()
