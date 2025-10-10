FROM python:3.10-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libsnappy-dev \
    zlib1g-dev \
    libbz2-dev \
    liblz4-dev \
    libzstd-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt indexing_and_retrieval/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r indexing_and_retrieval/requirements.txt

RUN python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True)"

COPY . indexing_and_retrieval/

RUN mkdir -p indexing_and_retrieval/plots indexing_and_retrieval/metrics indexing_and_retrieval/indices

ENV PYTHONPATH=/workspace

CMD ["python", "indexing_and_retrieval/scripts/main.py"]
