from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, StringConstraints

from src.data.prepare_data import clean_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "best_pipeline.joblib"

SENTIMENTS = {-1: "negatif", 0: "neutre", 1: "positif"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODEL_PATH)
    yield
    app.state.model = None


app = FastAPI(
    title="API Sentiment YouTube",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"chrome-extension://[a-p]{32}",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

Comment = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=20000,
    ),
]


class BatchRequest(BaseModel):
    comments: list[Comment] = Field(min_length=1, max_length=50)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": app.state.model is not None,
    }


@app.post("/predict_batch")
def predict_batch(body: BatchRequest):
    cleaned = [clean_text(text) for text in body.comments]

    invalid_indices = [
        index for index, text in enumerate(cleaned) if not text
    ]
    if invalid_indices:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Commentaires vides après nettoyage.",
                "indices": invalid_indices,
            },
        )

    model = app.state.model
    probabilities = model.predict_proba(cleaned)
    classes = model.classes_

    results = []
    counts = {"negatif": 0, "neutre": 0, "positif": 0}

    for original, scores in zip(body.comments, probabilities):
        best_index = int(scores.argmax())
        label = int(classes[best_index])
        sentiment = SENTIMENTS[label]

        counts[sentiment] += 1
        results.append({
            "text": original,
            "label": label,
            "sentiment": sentiment,
            "confidence": round(float(scores[best_index]), 4),
        })

    total = len(results)

    return {
        "results": results,
        "statistics": {
            "total": total,
            "counts": counts,
            "percentages": {
                name: round(count / total * 100, 2)
                for name, count in counts.items()
            },
        },
    }