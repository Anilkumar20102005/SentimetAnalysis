"""
Product Sentiment Analysis — FastAPI Backend
Serves the frozen TF-IDF + Logistic Regression model trained on CyranoB/polarity.
"""

import os
import re
import json
import time
import logging
import joblib
import nltk
import numpy as np
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ─── NLTK Setup ─────────────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("omw-1.4",   quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_pipeline.joblib"
META_PATH  = BASE_DIR / "models" / "model_metadata.json"
DIST_DIR   = BASE_DIR.parent / "frontend" / "dist"

# ─── Global model state ──────────────────────────────────────────────────────
model_state: dict = {}

STOP_WORDS = set(stopwords.words("english"))
STOP_WORDS.update({"br", "would", "one", "also", "get", "got", "use", "used", "using", "like", "make", "made"})
lemmatizer = WordNetLemmatizer()


def preprocess(text: str) -> str:
    """Mirror of notebook preprocessing — must stay identical."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split()
              if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model
    logger.info("Loading sentiment model...")
    if not MODEL_PATH.exists():
        logger.error(f"Model not found at {MODEL_PATH}. Run the notebook first!")
        model_state["pipeline"] = None
        model_state["metadata"] = {}
    else:
        model_state["pipeline"] = joblib.load(MODEL_PATH)
        if META_PATH.exists():
            with open(META_PATH) as f:
                model_state["metadata"] = json.load(f)
        logger.info("Model loaded successfully.")
    yield
    model_state.clear()


# ─── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Product Sentiment Analysis API",
    description="Analyzes product review sentiment using a TF-IDF + Logistic Regression model trained on Amazon Review Polarity (CyranoB/polarity).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ─────────────────────────────────────────────────────────────────
class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=5000, description="Review text to analyze")
    title: Optional[str] = Field(None, max_length=500, description="Optional review title")

    model_config = {"json_schema_extra": {
        "example": {"title": "Great product!", "text": "Absolutely loved it. Works perfectly and arrived fast!"}
    }}


class SentimentResult(BaseModel):
    sentiment: str            # "positive" or "negative"
    confidence: float         # 0.0 – 1.0
    positive_prob: float
    negative_prob: float
    processed_text: str
    inference_ms: float


class BatchRequest(BaseModel):
    reviews: list[ReviewRequest] = Field(..., min_length=1, max_length=50)


class ModelInfoResponse(BaseModel):
    model_type: str
    dataset: str
    test_accuracy: Optional[float]
    test_roc_auc: Optional[float]
    train_samples: Optional[int]
    model_loaded: bool


# ─── API Router ──────────────────────────────────────────────────────────────
api_router = APIRouter()

@api_router.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "model_loaded": model_state.get("pipeline") is not None}


@api_router.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    meta = model_state.get("metadata", {})
    return ModelInfoResponse(
        model_type=meta.get("model_type", "Unknown"),
        dataset=meta.get("dataset", "Unknown"),
        test_accuracy=meta.get("test_accuracy"),
        test_roc_auc=meta.get("test_roc_auc"),
        train_samples=meta.get("train_samples"),
        model_loaded=model_state.get("pipeline") is not None,
    )


@api_router.post("/predict", response_model=SentimentResult, tags=["Prediction"])
async def predict(req: ReviewRequest):
    pipeline = model_state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run the notebook first!")

    combined = (req.title or "") + " " + req.text
    cleaned  = preprocess(combined)

    if not cleaned.strip():
        raise HTTPException(status_code=422, detail="Text is empty after preprocessing.")

    t0    = time.perf_counter()
    proba = pipeline.predict_proba([cleaned])[0]
    pred  = int(np.argmax(proba))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return SentimentResult(
        sentiment="positive" if pred == 1 else "negative",
        confidence=round(float(max(proba)), 4),
        positive_prob=round(float(proba[1]), 4),
        negative_prob=round(float(proba[0]), 4),
        processed_text=cleaned[:300],
        inference_ms=round(elapsed_ms, 2),
    )


@api_router.post("/predict/batch", tags=["Prediction"])
async def predict_batch(req: BatchRequest):
    pipeline = model_state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    results = []
    for review in req.reviews:
        combined = (review.title or "") + " " + review.text
        cleaned  = preprocess(combined)
        if not cleaned.strip():
            results.append({"error": "empty_after_preprocessing", "text": review.text[:50]})
            continue
        t0    = time.perf_counter()
        proba = pipeline.predict_proba([cleaned])[0]
        pred  = int(np.argmax(proba))
        elapsed_ms = (time.perf_counter() - t0) * 1000
        results.append({
            "sentiment": "positive" if pred == 1 else "negative",
            "confidence": round(float(max(proba)), 4),
            "positive_prob": round(float(proba[1]), 4),
            "negative_prob": round(float(proba[0]), 4),
            "inference_ms": round(elapsed_ms, 2),
            "text_preview": review.text[:80],
        })
    return {"count": len(results), "results": results}


@api_router.get("/examples", tags=["Prediction"])
async def get_examples():
    """Return example review texts for UI demo."""
    return {
        "examples": [
            {"label": "Strong Positive", "text": "This is absolutely the best product I've ever purchased! Works perfectly, arrived fast, and exceeded every expectation. Highly recommend to everyone!"},
            {"label": "Strong Negative", "text": "Terrible quality. The item broke after just two days of normal use. Complete waste of money. Customer service was unhelpful. Do not buy!"},
            {"label": "Mild Positive",   "text": "Pretty good product overall. Does what it's supposed to do and the price is reasonable. Minor packaging issues but nothing major."},
            {"label": "Mild Negative",   "text": "Not what I expected. The product looks different from the pictures and doesn't work as advertised. Somewhat disappointed."},
            {"label": "Neutral/Mixed",   "text": "Average product. Has some good features but also some drawbacks. Shipping was fast. Would consider repurchasing if improved."},
        ]
    }

# Register API routes both with /api prefix and at root
app.include_router(api_router, prefix="/api")
app.include_router(api_router)

# Mount frontend assets if built
if DIST_DIR.exists() and (DIST_DIR / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")

    @app.get("/", tags=["Frontend"])
    async def serve_index():
        return FileResponse(DIST_DIR / "index.html")

    @app.get("/{catchall:path}", tags=["Frontend"])
    async def serve_catchall(catchall: str):
        file_path = DIST_DIR / catchall
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(DIST_DIR / "index.html")
else:
    @app.get("/", tags=["Health"])
    async def root():
        return {
            "service": "Product Sentiment Analysis API",
            "version": "1.0.0",
            "status": "ready" if model_state.get("pipeline") else "model_not_loaded",
            "docs": "/docs",
        }
