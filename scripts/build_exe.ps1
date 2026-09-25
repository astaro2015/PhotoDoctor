$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$Log = Join-Path $Root 'build.log'
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$Dist = Join-Path $Root 'dist'
$Work = Join-Path $Root 'build'
$Spec = Join-Path $Root 'PhotoDoctor.spec'
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Initialize-Log {
    [System.IO.File]::WriteAllText($Log, "Photo Doctor build log`r`n", $Utf8NoBom)
}

function Write-Log([string]$Text) {
    $Line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | $Text"
    [System.IO.File]::AppendAllText($Log, $Line + "`r`n", $Utf8NoBom)
    Write-Host $Line
}

function Write-CommandOutput([string]$Text) {
    if ($null -eq $Text) { return }
    [System.IO.File]::AppendAllText($Log, $Text + "`r`n", $Utf8NoBom)
    Write-Host $Text
}

function Format-ArgumentForLog([string]$Value) {
    if ($Value -match '[\s"]') {
        return '"' + ($Value -replace '"', '\"') + '"'
    }
    return $Value
}

function Invoke-Checked([string]$Exe, [string[]]$Arguments, [string]$ErrorMessage) {
    $ShownArgs = ($Arguments | ForEach-Object { Format-ArgumentForLog ([string]$_) }) -join ' '
    Write-Log "RUN: $Exe $ShownArgs"

    # Windows PowerShell 5.1 turns stderr from native programs into ErrorRecord
    # objects. PyInstaller writes normal INFO/WARNING progress messages to stderr,
    # so the script-wide ErrorActionPreference='Stop' must not be allowed to abort
    # a healthy build on the first informational line. A native process is judged
    # only by its exit code; stdout/stderr are still captured in full in build.log.
    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Exe @Arguments 2>&1 | ForEach-Object {
            Write-CommandOutput ([string]$_)
        }
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }

    if ($ExitCode -ne 0) {
        throw "$ErrorMessage Exit code: $ExitCode"
    }
}

try {
    Initialize-Log

    if (-not (Test-Path -LiteralPath $Python)) {
        throw 'Run START_PhotoDoctor.bat once before building the EXE.'
    }
    if (-not (Test-Path -LiteralPath $Spec)) {
        throw 'PhotoDoctor.spec is missing.'
    }

    Write-Log 'Installing pinned build dependencies in local virtual environment.'
    Push-Location $Root
    try {
        Invoke-Checked $Python @('-m', 'pip', 'install', '--disable-pip-version-check', '-r', 'requirements-build.txt') 'Could not install build dependencies.'
        Invoke-Checked $Python @('-m', 'PyInstaller', '--version') 'Could not query PyInstaller version.'
    }
    finally {
        Pop-Location
    }

    Remove-Item -LiteralPath $Dist -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $Work -Recurse -Force -ErrorAction SilentlyContinue

    Write-Log 'Building standalone PhotoDoctor.exe from PhotoDoctor.spec.'
    Push-Location $Root
    try {
        Invoke-Checked $Python @('-m', 'PyInstaller', '--noconfirm', '--clean', $Spec) 'PyInstaller failed.'
    }
    finally {
        Pop-Location
    }

    $Exe = Join-Path $Dist 'PhotoDoctor.exe'
    if (-not (Test-Path -LiteralPath $Exe)) {
        throw 'PyInstaller did not create dist\PhotoDoctor.exe.'
    }

    $SizeMb = [Math]::Round((Get-Item -LiteralPath $Exe).Length / 1MB, 1)
    Write-Log "Done: $Exe ($SizeMb MB)"
    exit 0
}
catch {
    try {
        Write-Log "ERROR: $($_.Exception.Message)"
    }
    catch {
    }
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host "Details: $Log"
    exit 1
}
