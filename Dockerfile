# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.lock ./
RUN --mount=type=cache,target=/root/.cache/pip pip wheel --wheel-dir=/wheels -r requirements.lock
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --no-deps --wheel-dir=/wheels .

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /app
RUN groupadd --system app && useradd --system --gid app --create-home app
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels pip install --no-index --find-links=/wheels recotrail
COPY alembic.ini ./
COPY migrations ./migrations
COPY scripts ./scripts
COPY models ./models
USER app
EXPOSE 8000
CMD ["uvicorn", "recotrail.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS test
USER root
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels pip install --no-index --find-links=/wheels "recotrail[dev]"
COPY pyproject.toml ./
COPY tests ./tests
COPY data ./data
USER app
CMD ["sh", "-c", "alembic upgrade head && pytest"]

FROM runtime AS final
