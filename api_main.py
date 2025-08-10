# api_main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Your existing helpers
from auth_test import recommend_from_seed, create_playlist  # both must be importable

app = FastAPI(title="Music Recommender API")

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- Request & response models ----------

class RecommendRequest(BaseModel):
    seed: str                 # name / URL / URI / ID
    top: int = 20             # how many recommendations to return
    create_playlist: bool = False  # if true, also create a Spotify playlist

class TrackOut(BaseModel):
    id: str
    name: str
    artist: str
    score: float

class RecommendResponse(BaseModel):
    seed: str
    top: int
    results: List[TrackOut]
    playlist_url: Optional[str] = None   # optional field, present only if created

# ---------- Endpoint ----------

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    try:
        # 1) Get recommendations via your existing pipeline
        results = recommend_from_seed(req.seed, top_k=req.top)  # List[dict]: {"id","name","artist","score"}

        # 2) Optionally create a playlist and return its URL
        playlist_url = None
        if req.create_playlist:
            # create_playlist can accept the list of dicts directly (per our updated function)
            meta = create_playlist(results, f"PYTHON MADE - {req.seed}")
            # meta is {"id": ..., "url": ..., "count": ..., "name": ..., "public": ...}
            playlist_url = meta.get("url")

        # 3) Return JSON
        return {
            "seed": req.seed,
            "top": req.top,
            "results": results,
            "playlist_url": playlist_url,
        }

    except RuntimeError as e:
        # e.g., feature extraction unavailable, no genres, etc.
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        # any other error becomes a 400
        raise HTTPException(status_code=400, detail=str(e))
