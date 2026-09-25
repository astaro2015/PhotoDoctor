$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$Log = Join-Path $Root 'setup.log'
$VenvDir = Join-Path $Root '.venv'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
$VenvPythonw = Join-Path $VenvDir 'Scripts\pythonw.exe'
$PythonVersion = '3.12.10'
$PythonInstallerUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe"
$TempDir = Join-Path $env:TEMP 'PhotoDoctorSetup'
$Installer = Join-Path $TempDir "python-$PythonVersion-amd64.exe"

function Write-Log([string]$Text) {
    $Line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | $Text"
    $Line | Tee-Object -FilePath $Log -Append
}

function Test-Python312([string]$Exe, [string[]]$PrefixArgs = @()) {
    try {
        $Output = & $Exe @PrefixArgs --version 2>&1
        if ($LASTEXITCODE -ne 0) { return $false }
        return ([string]$Output -match '^Python 3\.12\.')
    }
    catch {
        return $false
    }
}

function Find-Python312 {
    if ((Test-Path -LiteralPath $VenvPython) -and (Test-Python312 $VenvPython)) {
        return @{ Exe = $VenvPython; Args = @(); IsVenv = $true }
    }

    $Py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($Py -and (Test-Python312 $Py.Source @('-3.12'))) {
        return @{ Exe = $Py.Source; Args = @('-3.12'); IsVenv = $false }
    }

    $Candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'),
        (Join-Path $env:ProgramFiles 'Python312\python.exe')
    )

    if (${env:ProgramFiles(x86)}) {
        $Candidates += (Join-Path ${env:ProgramFiles(x86)} 'Python312\python.exe')
    }

    foreach ($Candidate in $Candidates) {
        if ((Test-Path -LiteralPath $Candidate) -and (Test-Python312 $Candidate)) {
            return @{ Exe = $Candidate; Args = @(); IsVenv = $false }
        }
    }

    $Python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($Python -and (Test-Python312 $Python.Source)) {
        return @{ Exe = $Python.Source; Args = @(); IsVenv = $false }
    }

    return $null
}

function Invoke-Checked([string]$Exe, [string[]]$Arguments, [string]$ErrorMessage) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$ErrorMessage Exit code: $LASTEXITCODE"
    }
}

try {
    Set-Content -LiteralPath $Log -Value 'Photo Doctor setup log' -Encoding UTF8
    Write-Log "Project root: $Root"

    $Python = Find-Python312
    if (-not $Python) {
        Write-Log "Python 3.12 not found. Downloading official Python $PythonVersion for current user."
        New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
        Invoke-WebRequest -Uri $PythonInstallerUrl -OutFile $Installer -UseBasicParsing

        $Signature = Get-AuthenticodeSignature -FilePath $Installer
        $Signer = ''
        if ($Signature.SignerCertificate) {
            $Signer = [string]$Signature.SignerCertificate.Subject
        }
        if ($Signature.Status -ne 'Valid' -or $Signer -notmatch 'Python Software Foundation') {
            throw "Python installer signature validation failed. Status=$($Signature.Status), Signer=$Signer"
        }
        Write-Log 'Python installer signature is valid.'

        $InstallerArgs = @(
            '/quiet',
            'InstallAllUsers=0',
            'PrependPath=0',
            'Include_launcher=1',
            'InstallLauncherAllUsers=0',
            'Include_pip=1',
            'Include_test=0',
            'Include_doc=0',
            'Shortcuts=0'
        )
        $Process = Start-Process -FilePath $Installer -ArgumentList $InstallerArgs -Wait -PassThru
        if ($Process.ExitCode -ne 0) {
            throw "Python installer failed. Exit code: $($Process.ExitCode)"
        }

        $Python = Find-Python312
        if (-not $Python) {
            throw 'Python 3.12 was installed but could not be found.'
        }
        Write-Log 'Python 3.12 installed for current user.'
    }
    else {
        Write-Log 'Python 3.12 found.'
    }

    if (Test-Path -LiteralPath $VenvDir) {
        if ((-not (Test-Path -LiteralPath $VenvPython)) -or (-not (Test-Python312 $VenvPython))) {
            Write-Log 'Existing virtual environment is invalid. Recreating it.'
            Remove-Item -LiteralPath $VenvDir -Recurse -Force
        }
    }

    if (-not (Test-Path -LiteralPath $VenvPython)) {
        Write-Log 'Creating local virtual environment.'
        $VenvArgs = @($Python.Args) + @('-m', 'venv', $VenvDir)
        Invoke-Checked $Python.Exe $VenvArgs 'Could not create virtual environment.'
    }

    if (-not (Test-Path -LiteralPath $VenvPython)) {
        throw 'Virtual environment was created but python.exe is missing.'
    }

    Write-Log 'Updating pip.'
    Invoke-Checked $VenvPython @('-m', 'pip', 'install', '--disable-pip-version-check', '-U', 'pip') 'Could not update pip.'

    Write-Log 'Installing Photo Doctor dependencies.'
    Push-Location $Root
    try {
        Invoke-Checked $VenvPython @('-m', 'pip', 'install', '--disable-pip-version-check', '-r', 'requirements-user.txt') 'Could not install Photo Doctor dependencies.'
    }
    finally {
        Pop-Location
    }

    Write-Log 'Verifying runtime environment.'
    $VerifyScript = Join-Path $Root 'scripts\verify_env.py'
    Invoke-Checked $VenvPython @($VerifyScript) 'Runtime verification failed.'

    if (-not (Test-Path -LiteralPath $VenvPythonw)) {
        throw 'pythonw.exe is missing from virtual environment.'
    }

    Write-Log 'Setup completed. Starting Photo Doctor.'
    Start-Process -FilePath $VenvPythonw -ArgumentList @('-m', 'photodoctor') -WorkingDirectory $Root
    exit 0
}
catch {
    try {
        Write-Log "ERROR: $($_.Exception.Message)"
    }
    catch {
    }
    Write-Host ''
    Write-Host 'Photo Doctor setup failed.' -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host "Details: $Log"
    exit 1
}
