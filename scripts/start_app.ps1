# Final RAG Chatbot System - PowerShell Launcher
# This script launches the Streamlit web interface with enhanced error handling

Write-Host ""

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "   Final RAG Chatbot System - Web Interface" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python version: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ and try again" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if we're in the right directory
if (-not (Test-Path "src\ui\streamlit_app.py")) {
    Write-Host "ERROR: src\ui\streamlit_app.py not found!" -ForegroundColor Red
    Write-Host "Please run this script from the 'final chatbot' directory" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if config/config.json exists
if (-not (Test-Path "config\config.json")) {
    Write-Host "WARNING: config\config.json not found!" -ForegroundColor Yellow
    Write-Host "The application may not work correctly without configuration" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Starting the application..." -ForegroundColor Green
Write-Host ""
Write-Host "The web interface will open at: http://localhost:8501" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the application" -ForegroundColor Yellow
Write-Host ""
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host ""

# Run the Python launcher
try {
    python scripts\launch.py
} catch {
    Write-Host ""
    Write-Host "ERROR: Failed to start the application" -ForegroundColor Red
    Write-Host "Error details: $_" -ForegroundColor Red
} finally {
    Write-Host ""
    Write-Host "Application stopped." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
}
