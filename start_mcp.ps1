# PowerShell script to launch the MCP-enabled RAG Chatbot
# This script starts the MCP server and optionally the Streamlit interface

param(
    [string]$Config = "config\config.json",
    [string]$LogLevel = "INFO",
    [switch]$ServerOnly,
    [switch]$StreamlitOnly,
    [switch]$Help
)

if ($Help) {
    Write-Host "Final RAG Chatbot MCP Launcher"
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  .\start_mcp.ps1 [options]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Config <path>      Configuration file path (default: config\config.json)"
    Write-Host "  -LogLevel <level>   Logging level (default: INFO)"
    Write-Host "  -ServerOnly         Start only the MCP server"
    Write-Host "  -StreamlitOnly      Start only the Streamlit interface"
    Write-Host "  -Help               Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\start_mcp.ps1                    # Start both server and interface"
    Write-Host "  .\start_mcp.ps1 -ServerOnly        # Start only MCP server"
    Write-Host "  .\start_mcp.ps1 -StreamlitOnly     # Start only Streamlit interface"
    exit 0
}

# Check if Python is available
try {
    $pythonVersion = python --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Python is not installed or not in PATH"
        exit 1
    }
    Write-Host "Using Python: $pythonVersion"
} catch {
    Write-Error "Error checking Python installation: $_"
    exit 1
}

# Check if required files exist
$requiredFiles = @(
    "src\mcp\mcp_server.py",
    "src\ui\streamlit_app_mcp.py",
    $Config
)

foreach ($file in $requiredFiles) {
    if (!(Test-Path $file)) {
        Write-Error "Required file not found: $file"
        exit 1
    }
}

Write-Host "Final RAG Chatbot MCP System Starting..." -ForegroundColor Green
Write-Host "Configuration: $Config" -ForegroundColor Yellow
Write-Host "Log Level: $LogLevel" -ForegroundColor Yellow

if ($ServerOnly) {
    Write-Host "Starting MCP Server only..." -ForegroundColor Cyan
    python src\mcp\mcp_server.py --config $Config --log-level $LogLevel
} elseif ($StreamlitOnly) {
    Write-Host "Starting Streamlit interface only..." -ForegroundColor Cyan
    streamlit run src\ui\streamlit_app_mcp.py --server.port 8501
} else {
    Write-Host "Starting both MCP Server and Streamlit interface..." -ForegroundColor Cyan
    
    # Start MCP server in background
    $serverJob = Start-Job -ScriptBlock {
        param($config, $logLevel)
        python src\mcp\mcp_server.py --config $config --log-level $logLevel
    } -ArgumentList $Config, $LogLevel
    
    Write-Host "MCP Server started in background (Job ID: $($serverJob.Id))" -ForegroundColor Green
    
    # Wait a moment for server to start
    Start-Sleep -Seconds 2
    
    # Start Streamlit interface
    Write-Host "Starting Streamlit interface..." -ForegroundColor Cyan
    try {
        streamlit run src\ui\streamlit_app_mcp.py --server.port 8501
    } finally {
        # Clean up background job when Streamlit stops
        Write-Host "Stopping MCP Server..." -ForegroundColor Yellow
        Stop-Job $serverJob -PassThru | Remove-Job
    }
}

Write-Host "Final RAG Chatbot MCP System stopped." -ForegroundColor Red
