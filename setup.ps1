# 1. Create virtual environment (still needed for MCP server and scripts)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment variables
Copy-Item .env.example .env
Write-Host "Fill in your .env values before continuing"

# 4. Build and start everything
docker compose -f infra/docker-compose.yml up -d --build

# 5. Wait for Ollama
Start-Sleep -Seconds 10

# 6. Pull models
Write-Host "Pulling models..."
docker exec ollama ollama pull mistral
docker exec ollama ollama pull nomic-embed-text
docker exec ollama ollama pull llama3.1

Write-Host ""
Write-Host "Setup complete. Now run:"
Write-Host "uvicorn api.main:app --reload"
# 7. Ingest sample documents
python scripts/ingest_all.py

Write-Host "Setup complete!"
Write-Host "MCP server: python mcp_server/server.py"
