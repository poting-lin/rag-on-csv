#!/bin/bash
# Entrypoint for the Ollama Docker service.
# Starts the server, pulls the configured model, then keeps serving.

set -e

MODEL_NAME="${OLLAMA_MODEL:-llama3.2:1b}"

# Start Ollama serve in the background
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama API to be reachable
echo "Waiting for Ollama server to start..."
until ollama list > /dev/null 2>&1; do
    sleep 1
done
echo "Ollama server is running."

# Pull the model (no-op if already downloaded)
echo "Pulling model: ${MODEL_NAME}..."
ollama pull "${MODEL_NAME}"
echo "Model ${MODEL_NAME} is ready."

# Keep the container alive by waiting on the server process
wait $OLLAMA_PID
