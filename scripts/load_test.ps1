param(
  [string]$ApiUrl = "http://localhost:8000/predict",
  [string]$InputPath = ".\input.json",
  [int]$Rps = 1
)

$body = Get-Content $InputPath -Raw

while ($true) {
  1..$Rps | ForEach-Object {
    Invoke-RestMethod -Method Post -Uri $ApiUrl -ContentType "application/json" -Body $body | Out-Null
  }
  Start-Sleep -Seconds 1
}
