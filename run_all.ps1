# run_all.ps1 - Launch MLflow UI and Streamlit in two new PowerShell windows.
# Usage: run this from project root:  .\run_all.ps1

# Allow script execution for this session (temporary)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

# Resolve project root
$proj = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$venvPython = Join-Path $proj "venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Error "venv python not found at $venvPython. Create/repair venv first."
    exit 1
}

# MLflow in new window
Start-Process powershell -ArgumentList "-NoExit","-Command","cd `"$proj`"; `"$venvPython`" -m mlflow ui --port 5000"

# Streamlit in new window, logs to streamlit_background.log
$streamlitLog = Join-Path $proj "streamlit_background.log"
$streamlitCommand = "cd `"$proj`"; `"$venvPython`" -u -m streamlit run `"$proj\app.py`" --server.port 8501 --server.address 127.0.0.1 2>&1 | Out-File -FilePath `"$streamlitLog`" -Encoding utf8"
Start-Process powershell -ArgumentList "-NoExit","-Command",$streamlitCommand

Write-Host "Started MLflow and Streamlit in new windows."
Write-Host "MLflow UI: http://127.0.0.1:5000"
Write-Host "Streamlit logs: $streamlitLog"
