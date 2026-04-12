# setup.ps1

# 1. Create and activate a virtual environment
python -m venv .venv

.\.venv\Scripts\Activate.ps1

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Copy environment variables template
Copy-Item .env.example .env
Write-Host ".env created - fill in your values"

# 4. Start infrastructure
Write-Host "Starting Docker containers..."
docker compose -f infra/docker-compose.yml up -d

# 5. Wait a moment for Ollama to be ready
Start-Sleep -Seconds 5

# 6. Pull Ollama models
Write-Host "Pulling Mistral model (this will take a few minutes)..."
docker exec ollama ollama pull mistral

Write-Host "Pulling embedding model..."
docker exec ollama ollama pull nomic-embed-text

Write-Host "Pulling agent model..."
docker exec ollama ollama pull llama3.1

Write-Host ""
Write-Host "Setup complete. Now run:"
Write-Host "uvicorn api.main:app --reload"