# setup.ps1

# 1. Create and activate a virtual environment
python -m venv .venv

.\.venv\Scripts\Activate.ps1

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Copy environment variables template
Copy-Item .env.example .env
Write-Host ".env created - fill in your values"

# 4. Pull Ollama models (after docker compose is up)
# Run this after: docker compose up -d
# ollama pull mistral
# ollama pull nomic-embed-text

Write-Host "Setup complete. Next: docker compose up -d"