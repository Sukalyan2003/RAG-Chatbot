@echo off
:: Final RAG Chatbot System - Windows Launcher
:: This batch file launches the Streamlit web interface

echo.

cd /d "%~dp0.."
echo ===================================================
echo    Final RAG Chatbot System - Web Interface
echo ===================================================
echo.

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

:: Display Python version
echo Python version:
python --version
echo.

:: Check if we're in the right directory
if not exist "src\ui\streamlit_app.py" (
    echo ERROR: src\ui\streamlit_app.py not found!
    echo Please run this script from the 'final chatbot' directory
    pause
    exit /b 1
)

:: Run the Python launcher
echo Starting the application...
echo.
echo The web interface will open at: http://localhost:8501
echo Press Ctrl+C to stop the application
echo.
echo ===================================================
echo.

python scripts\launch.py

echo.
echo Application stopped.
pause
