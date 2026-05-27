param(
    [string]$InputPath = "data/raw/docfinqa_train.json"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot + "\.."

if (-not (Test-Path ".env")) {
    throw ".env file is missing. Copy .env.example to .env and set OPENAI_API_KEY."
}

docker compose --profile ingest run --rm pipeline ingest --input-path $InputPath
