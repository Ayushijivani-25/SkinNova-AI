$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Backend = Join-Path $ProjectRoot "backend"
$Frontend = Join-Path $ProjectRoot "frontend"

Start-Process -WindowStyle Hidden -FilePath python -ArgumentList "run_backend_5004.py" -WorkingDirectory $Backend
Start-Process -WindowStyle Hidden -FilePath python -ArgumentList "-m http.server 5173" -WorkingDirectory $Frontend

Write-Host "SkinNova AI is running."
Write-Host "Website: http://127.0.0.1:5173"
Write-Host "Backend: http://127.0.0.1:5004/api/health"
