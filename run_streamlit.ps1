# --- Launch Streamlit for EMIPredict_AI ---
Set-Location "C:\Users\appu0\EMIPredict_AI"

# 1️⃣ Activate venv
& .\venv\Scripts\Activate.ps1

# 2️⃣ Free port 8501 if an old process is stuck
$net = netstat -aon | findstr ":8501"
if ($net) {
    $pid = ($net -split '\s+')[-1]
    if ($pid -ne $PID) {
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        Write-Host "✅ Old Streamlit process (PID $pid) stopped."
    }
} else {
    Write-Host "✅ Port 8501 free."
}

# 3️⃣ Start Streamlit cleanly with polling watcher and log to file
Start-Process `
  -FilePath ".\venv\Scripts\python.exe" `
  -ArgumentList "-u -m streamlit run .\app.py --server.port 8501 --server.address 127.0.0.1 --server.fileWatcherType=poll" `
  -RedirectStandardOutput "streamlit.log" `
  -RedirectStandardError "streamlit.log" `
  -NoNewWindow

# 4️⃣ Wait a few seconds, then auto-open browser
Start-Sleep -Seconds 4
Start-Process "http://127.0.0.1:8501"

Write-Host "`n🚀 Streamlit app launching... Logs saved to streamlit.log"
