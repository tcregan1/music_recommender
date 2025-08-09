import requests
import json
import os

CACHE_FILE = "track_features_cache.json"
RAPIDAPI_KEY = "9a5e1eadacmsh4a889a1d586dc78p1914abjsn6dd814188f03"
# Load existing cache if it exists
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        feature_cache = json.load(f)
else:
    feature_cache = {}

def get_features_soundnet(song_name, artist_name):
    """Get track features from SoundNet API, with caching."""
    cache_key = f"{song_name.lower()}-{artist_name.lower()}"
    if cache_key in feature_cache:
        return feature_cache[cache_key]

    url = "https://track-analysis.p.rapidapi.com/pktx/analysis"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,  # your key here
        "X-RapidAPI-Host": "track-analysis.p.rapidapi.com"
    }
    params = {"song": song_name, "artist": artist_name}

    r = requests.get(url, headers=headers, params=params)
    if r.status_code != 200:
        print(f"❌ SoundNet error for {song_name} - {artist_name}: {r.status_code}, {r.text}")
        return None

    data = r.json()
    feature_cache[cache_key] = data
    with open(CACHE_FILE, "w") as f:
        json.dump(feature_cache, f, indent=2)

    return data

# Example usage
features = get_features_soundnet("Man in the box", "Alice in chains")
print(features)
