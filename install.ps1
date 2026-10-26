# install.ps1 — set up the standalone TTS Hub service (Breeze TTS 2 + Qwen3-TTS) in .\.venv
#   powershell -ExecutionPolicy Bypass -File .\install.ps1
#   powershell -ExecutionPolicy Bypass -File .\install.ps1 -PythonVersion 3.12 -Recreate
#
# Steps: uv -> .venv -> pip deps -> CUDA torch -> pinned Breeze inference code -> GPU check.
# Model weights (BreezeBlue/breeze-tts-2, ~7.7 GB) download into the HuggingFace cache on the
# first service start; an existing cache (e.g. from NarrifyAudio) is reused.

param(
    [string]$PythonVersion = "3.14",
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu128",
    [switch]$Recreate
)

$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
$venv = Join-Path $root ".venv"
$py = Join-Path $venv "Scripts\python.exe"

Write-Output "===== [1/6] ensure uv ====="
py -m uv --version
if ($LASTEXITCODE -ne 0) {
    py -m pip install --quiet --upgrade uv
    if ($LASTEXITCODE -ne 0) { throw "pip install uv failed (exit $LASTEXITCODE)" }
}

Write-Output "===== [2/6] ensure .venv (Python $PythonVersion) ====="
py -m uv python install $PythonVersion
if ($LASTEXITCODE -ne 0) { throw "uv python install $PythonVersion failed" }
if ((Test-Path -LiteralPath $py) -and $Recreate) {
    Remove-Item -LiteralPath $venv -Recurse -Force
}
if (-not (Test-Path -LiteralPath $py)) {
    py -m uv venv $venv --python $PythonVersion
    if ($LASTEXITCODE -ne 0) { throw "uv venv failed" }
}
Write-Output ("venv python: " + (& $py --version))

Write-Output "===== [3/6] install dependencies ====="
py -m uv pip install -p $py -r (Join-Path $root "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }

Write-Output "===== [4/6] install CUDA torch last ====="
py -m uv pip install -p $py "torch==2.11.0" "torchaudio==2.11.0" --index-url $TorchIndex --force-reinstall --no-deps
if ($LASTEXITCODE -ne 0) { throw "CUDA torch install failed" }

Write-Output "===== [5/6] Breeze inference code (pinned) ====="
# Not a pip package: vendor/breeze-tts is a git checkout of the pinned upstream commit.
$breezeCommit = "6eba901d1845353802652c708a4740fdfc5baa75"
$breezeDir = Join-Path $root "vendor\breeze-tts"
if (-not (Test-Path -LiteralPath (Join-Path $breezeDir "breeze_infer\runtime.py"))) {
    git clone --quiet https://github.com/breezeblue-ai/breeze-tts.git $breezeDir
    if ($LASTEXITCODE -ne 0) { throw "git clone breeze-tts failed" }
}
if (Test-Path -LiteralPath (Join-Path $breezeDir ".git")) {
    git -C $breezeDir fetch --quiet origin $breezeCommit 2>$null
    git -C $breezeDir checkout --quiet $breezeCommit
    if ($LASTEXITCODE -ne 0) { Write-Output "WARNING: checkout of $breezeCommit failed" }
    else { Write-Output "Breeze code at $breezeCommit" }
}

Write-Output "===== [6/6] verify ====="
& $py -c "import sys,torch;print('python='+sys.version.split()[0]);print('torch='+torch.__version__);print('cuda_available='+str(torch.cuda.is_available()));print('device='+(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'))"
& $py -c "import qwen_tts, soundfile, fastapi, mcp; print('imports OK')"
if ($LASTEXITCODE -ne 0) { throw "import check failed" }
Write-Output "done. Start with: .\start.ps1"
