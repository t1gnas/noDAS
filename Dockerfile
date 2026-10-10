FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements-torch.txt .
RUN pip install -r requirements-torch.txt

COPY pyproject.toml .
RUN mkdir das api && touch das/__init__.py api/__init__.py && pip install -e .

COPY das das
COPY api api
COPY config.yaml .

ENTRYPOINT ["python", "-m", "das"]
CMD ["--help"]
