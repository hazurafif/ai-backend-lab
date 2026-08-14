#!/usr/bin/env bash
# Serve Qwen3-Embedding-0.6B locally on Apple Silicon via MLX (mlx-lm server +
# mlx-embeddings). Exposes an OpenAI-compatible POST /v1/embeddings endpoint.
#
# Requires: macOS with Apple Silicon, uv. No Docker needed; MLX runs on Metal.
#
#   ./scripts/mlx_embeddings.sh                 # default model + port 8080
#   ./scripts/mlx_embeddings.sh <MODEL> <PORT>  # e.g. Qwen/Qwen3-Embedding-0.6B (auto-converted)
#
# Point the app at the server (or save an 'embeddings' connection with this
# base_url):
#
#   EMBEDDINGS_MLX_URL=http://127.0.0.1:8080/v1 uv run uvicorn app.main:app ...
#
# To also serve a chat LLM from the same process, add --model <repo> (chat +
# embeddings share one OpenAI-compatible port).
set -euo pipefail

MODEL="${1:-mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ}"
PORT="${2:-8080}"

exec uv run --with mlx-lm --with mlx-embeddings \
  mlx_lm.server --embedding-model "$MODEL" --port "$PORT"
