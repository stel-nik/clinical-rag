@echo off

cd /d C:\Users\stela\Coding\projects\clinical-rag

echo Starting Docker Desktop...
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"

echo Waiting for Docker to be ready...
timeout /t 10 >nul

echo Starting containers...
docker compose -f infra\docker-compose.yml up -d

echo Activating virtual environment...
call .\.venv\Scripts\activate

echo Running agent...
python agent\agent.py