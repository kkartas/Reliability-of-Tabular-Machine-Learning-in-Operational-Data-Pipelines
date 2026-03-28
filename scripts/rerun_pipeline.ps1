[CmdletBinding()]
param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [ValidateSet("refresh", "full")][string]$Mode = "refresh",
    [string]$DatasetSet = "extended",
    [string]$DatasetSourcePolicy = "openml_only",
    [string]$C1AnchorMode = "label_agnostic",
    [switch]$RunC1Sensitivity,
    [switch]$StrictPaperVerify,
    [switch]$SkipCompile,
    [string]$CompileLogPath = "docs\compile_check_latest.txt"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Resolve-PathOrRooted {
    param(
        [Parameter(Mandatory = $true)][string]$PathValue,
        [Parameter(Mandatory = $true)][string]$RootDir
    )
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return $PathValue
    }
    return (Join-Path $RootDir $PathValue)
}

function Invoke-PythonScript {
    param(
        [Parameter(Mandatory = $true)][string]$PythonPath,
        [Parameter(Mandatory = $true)][string]$ScriptPath,
        [Parameter(Mandatory = $false)][string[]]$Arguments = @()
    )
    & $PythonPath $ScriptPath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $ScriptPath $($Arguments -join ' ') (exit=$LASTEXITCODE)"
    }
}

function Invoke-CompileScript {
    param(
        [Parameter(Mandatory = $true)][string]$RepositoryRoot
    )
    $psExe = (Get-Process -Id $PID).Path
    & $psExe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepositoryRoot "scripts\compile_paper.ps1")
    if ($LASTEXITCODE -ne 0) {
        throw "Compile script failed with exit code $LASTEXITCODE"
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonPath = Resolve-PathOrRooted -PathValue $PythonExe -RootDir $repoRoot

if (-not (Test-Path $pythonPath)) {
    throw "Python executable not found: $pythonPath"
}

Push-Location $repoRoot
try {
    Write-Host "[pipeline] repo root: $repoRoot"
    Write-Host "[pipeline] python: $pythonPath"
    Write-Host "[pipeline] mode: $Mode"

    if ($Mode -eq "full") {
        Write-Host "[pipeline] running full experiment pipeline..."
        $env:PYTHONHASHSEED = "0"
        $env:DATASET_SET = $DatasetSet
        $env:DATASET_SOURCE_POLICY = $DatasetSourcePolicy
        $env:C1_ANCHOR_MODE = $C1AnchorMode
        Write-Host "[pipeline] PYTHONHASHSEED=$($env:PYTHONHASHSEED)"
        Write-Host "[pipeline] DATASET_SET=$($env:DATASET_SET)"
        Write-Host "[pipeline] DATASET_SOURCE_POLICY=$($env:DATASET_SOURCE_POLICY)"
        Write-Host "[pipeline] C1_ANCHOR_MODE=$($env:C1_ANCHOR_MODE)"
        Invoke-PythonScript -PythonPath $pythonPath -ScriptPath "run_experiments.py"
    }
    else {
        Write-Host "[pipeline] refresh mode: reusing existing raw experiment artifacts."
        $requiredArtifact = Join-Path $repoRoot "results\raw\metrics_long.csv"
        if (-not (Test-Path $requiredArtifact)) {
            throw "Missing required artifact: $requiredArtifact. Run with -Mode full, or generate artifacts first."
        }
    }

    if ($RunC1Sensitivity) {
        Write-Host "[pipeline] running C1 anchor sensitivity (label-informed vs label-agnostic)..."
        Invoke-PythonScript -PythonPath $pythonPath -ScriptPath "scripts\c1_anchor_sensitivity.py"
    }

    Write-Host "[pipeline] regenerating derived tables..."
    Invoke-PythonScript -PythonPath $pythonPath -ScriptPath "scripts\regenerate_derived_tables.py"

    Write-Host "[pipeline] building LaTeX tables..."
    Invoke-PythonScript -PythonPath $pythonPath -ScriptPath "scripts\build_latex_tables.py"

    Write-Host "[pipeline] extracting paper numbers (text/json)..."
    Invoke-PythonScript -PythonPath $pythonPath -ScriptPath "scripts\extract_paper_numbers.py" -Arguments @("--format", "text", "--out", "docs\paper_numbers_latest.txt")
    Invoke-PythonScript -PythonPath $pythonPath -ScriptPath "scripts\extract_paper_numbers.py" -Arguments @("--format", "json", "--out", "docs\paper_numbers_latest.json")

    Write-Host "[pipeline] verifying paper numbers against artifacts..."
    & $pythonPath "scripts\extract_paper_numbers.py" "--verify-paper" "--paper-path" "paper\paper.tex" "--format" "json" "--out" "docs\paper_numbers_validation.json"
    if ($LASTEXITCODE -ne 0) {
        $msg = "Paper-number verification reported mismatches. Review docs\\paper_numbers_validation.json."
        if ($StrictPaperVerify) {
            throw $msg
        }
        Write-Warning "[pipeline] $msg Continuing (use -StrictPaperVerify to fail on mismatch)."
    }

    if (-not $SkipCompile) {
        Write-Host "[pipeline] compiling manuscript..."
        Invoke-CompileScript -RepositoryRoot $repoRoot

        $paperLogPrimary = Join-Path $repoRoot "paper\paper.log"
        $paperLogLegacy = Join-Path $repoRoot "paper\paper_buildcheck.log"
        $paperLog = $paperLogPrimary
        if (-not (Test-Path $paperLog) -and (Test-Path $paperLogLegacy)) {
            $paperLog = $paperLogLegacy
        }
        $compileLogTarget = Resolve-PathOrRooted -PathValue $CompileLogPath -RootDir $repoRoot
        if (Test-Path $paperLog) {
            $targetDir = Split-Path -Parent $compileLogTarget
            if ($targetDir) {
                New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
            }
            Copy-Item -Path $paperLog -Destination $compileLogTarget -Force
            Write-Host "[pipeline] compile log copied to: $compileLogTarget"

            $undefinedPattern = "LaTeX Warning: (Reference|Citation).*undefined|There were undefined"
            if (Select-String -Path $compileLogTarget -Pattern $undefinedPattern -Quiet) {
                throw "Undefined references/citations found in compile log: $compileLogTarget"
            }
        }
        else {
            Write-Warning "[pipeline] expected compile log not found: $paperLog"
        }
    }
    else {
        Write-Host "[pipeline] skipping compile (--SkipCompile)."
    }

    Write-Host "[pipeline] completed successfully."
}
finally {
    Pop-Location
}
