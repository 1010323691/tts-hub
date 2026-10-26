# start.ps1 — start the TTS Hub service (Breeze TTS 2 + Qwen3-TTS).
#   .\start.ps1                                   # 127.0.0.1:7861
#   .\start.ps1 -BindHost 0.0.0.0 -ApiKey xxx     # LAN access, bearer-token protected
#   .\start.ps1 -IdleUnloadMinutes 0 -IdleShutdownMinutes 0   # stay loaded, never exit
# Every TTS_* environment variable (see .env.example) is honoured as well; old BREEZE_* names still work.
param(
    [string]$BindHost = "",
    [int]$Port = 0,
    [string]$ApiKey = "",
    [string]$Model = "",
    [ValidateSet("", "breeze", "qwen3")]
    [string]$Engine = "",
    [int]$MaxBatch = 0,
    [double]$IdleUnloadMinutes = -1,
    [double]$IdleShutdownMinutes = -1
)
$root = $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $py)) { throw "未找到 .venv，请先运行 install.ps1" }

# Load .env (KEY=VALUE lines) without overriding variables already set.
$envFile = Join-Path $root ".env"
if (Test-Path -LiteralPath $envFile) {
    Get-Content -LiteralPath $envFile | ForEach-Object {
        if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$' -and -not [Environment]::GetEnvironmentVariable($Matches[1])) {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2].Trim('"'))
        }
    }
}
$env:PYTHONUTF8 = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$cliArgs = @("-m", "tts_hub")
if ($BindHost) { $cliArgs += @("--host", $BindHost) }
if ($Port) { $cliArgs += @("--port", $Port) }
if ($ApiKey) { $cliArgs += @("--api-key", $ApiKey) }
if ($Model) { $cliArgs += @("--model", $Model) }
if ($Engine) { $cliArgs += @("--engine", $Engine) }
if ($MaxBatch) { $cliArgs += @("--max-batch", $MaxBatch) }
if ($IdleUnloadMinutes -ge 0) { $cliArgs += @("--idle-unload-minutes", $IdleUnloadMinutes) }
if ($IdleShutdownMinutes -ge 0) { $cliArgs += @("--idle-shutdown-minutes", $IdleShutdownMinutes) }
Set-Location -LiteralPath $root
& $py @cliArgs
