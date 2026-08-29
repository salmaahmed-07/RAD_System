@echo off
REM deploy.bat - Deployment script for Windows

echo 🚀 Starting Telecom Egypt RAG System Deployment...

REM Check if .env file exists
if not exist .env (
    echo ⚠️  .env file not found. Creating from .env.example...
    copy .env.example .env
    echo 📝 Please edit .env file and add your OPENROUTER_API_KEY
    exit /b 1
)

REM Build Docker image
echo 📦 Building Docker image...
docker build -t telecom-rag .

REM Run with Docker Compose
echo 🚀 Starting containers...
docker-compose up -d

echo ✅ Deployment complete!
echo 🌐 Application running at: http://localhost:8501