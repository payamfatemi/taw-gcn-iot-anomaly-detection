FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir --upgrade pip &&     python -m pip install --no-cache-dir .
COPY configs ./configs
COPY scripts ./scripts
ENTRYPOINT ["python", "scripts/run_pipeline.py"]
