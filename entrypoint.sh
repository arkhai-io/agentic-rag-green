#!/bin/bash
set -e

# Download benchmark data if URLs are provided
BENCHMARK_DIR="/home/agent/data/benchmarks/female_longevity"
mkdir -p "$BENCHMARK_DIR/papers"

# Download papers if URL provided
if [ -n "$PAPERS_URL" ]; then
    echo "Downloading papers from $PAPERS_URL..."
    curl -fsSL "$PAPERS_URL" -o /tmp/papers.zip
    unzip -o /tmp/papers.zip -d "$BENCHMARK_DIR/papers"
    rm /tmp/papers.zip
    echo "Papers downloaded: $(ls -1 $BENCHMARK_DIR/papers | wc -l) files"
fi

# Download QA pairs if URL provided
if [ -n "$QA_PAIRS_URL" ]; then
    echo "Downloading QA pairs from $QA_PAIRS_URL..."
    curl -fsSL "$QA_PAIRS_URL" -o /tmp/qa.zip
    unzip -o /tmp/qa.zip -d /tmp/qa_extract
    # Find and move any .json file to benchmark dir (handles nested paths)
    find /tmp/qa_extract -name "*.json" -exec mv {} "$BENCHMARK_DIR/" \;
    rm -rf /tmp/qa.zip /tmp/qa_extract
    echo "QA pairs downloaded: $(ls -1 $BENCHMARK_DIR/*.json 2>/dev/null | wc -l) json files"
fi

# Start the server
exec uv run python -m src.server "$@"
