Param(
  [string]$RepoId = "alexwengg/parakeet-tdt-0.6b-v2-ov",
  [string]$BuildType = "Release"
)

$ErrorActionPreference = "Stop"

Write-Host "[Eddy] Configuring build..."
if (-not (Test-Path -LiteralPath "build")) {
  cmake -S . -B build -DCMAKE_BUILD_TYPE=$BuildType -DEDDY_ENABLE_OPENVINO=ON | Out-Host
}

Write-Host "[Eddy] Building hf_fetch_models..."
cmake --build build --config $BuildType --target hf_fetch_models | Out-Host

$exe = Join-Path -Path (Resolve-Path "build\examples\cpp\$BuildType").Path -ChildPath "hf_fetch_models.exe"
if (-not (Test-Path -LiteralPath $exe)) {
  throw "hf_fetch_models.exe not found at $exe"
}

# Resolve Eddy Parakeet models target
$local = $env:LOCALAPPDATA
if (-not $local) { throw "LOCALAPPDATA not set; cannot resolve cache directory" }
$target = Join-Path $local "eddy\models\parakeet-v2\files"
New-Item -ItemType Directory -Force -Path $target | Out-Null

Write-Host "[Eddy] Downloading Parakeet OV artifacts to: $target"
& $exe --repo $RepoId --target $target | Out-Host

Write-Host "[Eddy] Done. Files are in: $target"
