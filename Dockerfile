# ---- UI build ----
FROM node:22-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- runtime ----
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NUMBA_CACHE_DIR=/tmp/numba-cache \
    CATFORGE_DATA_DIR=/data CATFORGE_UI_DIST=/app/frontend/dist
WORKDIR /app
COPY pyproject.toml README.md ./
COPY catforge/ catforge/
RUN pip install --no-cache-dir .
COPY --from=ui /ui/dist frontend/dist
VOLUME ["/data"]
EXPOSE 8000
CMD ["catforge", "serve", "--host", "0.0.0.0", "--port", "8000", "--data-dir", "/data"]
