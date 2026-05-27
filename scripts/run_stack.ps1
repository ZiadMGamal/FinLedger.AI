param(
    [switch]$Detached
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot + "\.."

if (-not (Test-Path ".env")) {
    throw ".env file is missing. Copy .env.example to .env and set OPENAI_API_KEY."
}

if ($Detached) {
    docker compose up --build -d
} else {
    docker compose up --build
}
