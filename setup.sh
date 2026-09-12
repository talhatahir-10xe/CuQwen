#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"

echo "========================================================"
echo "  Building image: cuqwen"
echo "========================================================"
docker build -f "$PROJECT_DIR/Dockerfile" -t cuqwen:latest "$PROJECT_DIR"
echo "[✔] Image 'cuqwen:latest' built successfully."

echo "========================================================"
echo "  Starting container: cuqwen_container"
echo "========================================================"
docker rm -f cuqwen_container 2>/dev/null || true
docker run \
    --gpus all \
    -it \
    --name cuqwen_container \
    -v "$PROJECT_DIR:/workspace/$PROJECT_NAME" \
    -w "/workspace/$PROJECT_NAME" \
    cuqwen:latest