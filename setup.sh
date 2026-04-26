#!/usr/bin/env bash
set -e

# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment variables
cp .env.example .env
echo "Fill in your .env values before continuing"

# 4. Build and start everything
docker compose -f infra/docker-compose.yml up -d --build

# 5. Wait for Ollama
sleep 10

# 6. Pull models
echo "Pulling models..."
docker exec ollama ollama pull mistral
docker exec ollama ollama pull nomic-embed-text
docker exec ollama ollama pull llama3.1

echo ""
echo "Setup complete. Now run:"
echo "uvicorn api.main:app --reload"

# 7. Ingest sample documents
python scripts/ingest_all.py

echo "Setup complete!"
echo "MCP server: python mcp_server/server.py"
