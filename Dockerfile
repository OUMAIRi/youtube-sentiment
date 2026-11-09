FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/user

WORKDIR /app

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

RUN useradd --create-home --uid 1000 user

COPY --chown=user:user app_api.py .
COPY --chown=user:user src/api/ ./src/api/
COPY --chown=user:user src/data/prepare_data.py ./src/data/prepare_data.py
COPY --chown=user:user models/best_pipeline.joblib ./models/best_pipeline.joblib

USER user

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "app_api:app", "--host", "0.0.0.0", "--port", "7860"]