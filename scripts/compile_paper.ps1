Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$candidateBins = @(
    (Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64"),
    (Join-Path ${env:ProgramFiles} "MiKTeX\miktex\bin\x64")
)
foreach ($candidate in $candidateBins) {
    if ($candidate -and (Test-Path $candidate)) {
        $pathParts = $env:PATH -split ';'
        if ($pathParts -notcontains $candidate) {
            $env:PATH = "$candidate;$env:PATH"
            Write-Host "[build] Added TeX bin to PATH: $candidate"
        }
    }
}

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $false)][string[]]$Arguments = @()
    )
    $process = Start-Process -FilePath $FilePath -ArgumentList $Arguments -NoNewWindow -Wait -PassThru
    return [int]$process.ExitCode
}

$paperDir = Join-Path $PSScriptRoot "..\paper"
$paperDir = (Resolve-Path $paperDir).Path

function Remove-IfExists {
    param(
        [Parameter(Mandatory = $true)][string]$PathValue
    )
    if (Test-Path $PathValue) {
        try {
            Remove-Item -Path $PathValue -Force -ErrorAction Stop
        }
        catch {
            Write-Warning "[build] Could not remove stale artifact (locked?): $PathValue"
        }
    }
}

Push-Location $paperDir
try {
    $mainJobName = "paper"
    $legacyJobName = "paper_buildcheck"

    # Remove stale aux artifacts from both canonical and legacy buildcheck names.
    # This prevents old macros (for example \newmarginnote from legacy builds)
    # from breaking current compiles when class/package sets changed.
    $staleExtensions = @("aux", "bbl", "blg", "out", "toc", "lof", "lot")
    foreach ($ext in $staleExtensions) {
        Remove-IfExists -PathValue (Join-Path $paperDir ($mainJobName + "." + $ext))
        Remove-IfExists -PathValue (Join-Path $paperDir ($legacyJobName + "." + $ext))
    }

    if (Get-Command latexmk -ErrorAction SilentlyContinue) {
        if (Get-Command perl -ErrorAction SilentlyContinue) {
            Write-Host "[build] Using latexmk from PATH"
            $latexmkExit = Invoke-NativeCommand -FilePath "latexmk" -Arguments @("-pdf", "-interaction=nonstopmode", "-halt-on-error", "paper.tex")
            if ($latexmkExit -eq 0) {
                exit 0
            }
            Write-Warning "[build] latexmk failed (exit=$latexmkExit); falling back to pdflatex+bibtex."
        }
        else {
            Write-Warning "[build] latexmk found but perl is missing; falling back to pdflatex+bibtex."
        }
    }

    if (Get-Command pdflatex -ErrorAction SilentlyContinue) {
        Write-Host "[build] Using pdflatex+bibtex fallback from PATH"
        $versionExit = Invoke-NativeCommand -FilePath "pdflatex" -Arguments @("--version")
        if ($versionExit -ne 0) {
            Write-Warning "[build] pdflatex --version returned exit=$versionExit; continuing with compile attempt."
        }
        $pass1 = Invoke-NativeCommand -FilePath "pdflatex" -Arguments @("-interaction=nonstopmode", "-halt-on-error", "paper.tex")
        if ($pass1 -ne 0) { exit $pass1 }
        $bib = Invoke-NativeCommand -FilePath "bibtex" -Arguments @($mainJobName)
        if ($bib -ne 0) { exit $bib }
        $pass2 = Invoke-NativeCommand -FilePath "pdflatex" -Arguments @("-interaction=nonstopmode", "-halt-on-error", "paper.tex")
        if ($pass2 -ne 0) { exit $pass2 }
        $pass3 = Invoke-NativeCommand -FilePath "pdflatex" -Arguments @("-interaction=nonstopmode", "-halt-on-error", "paper.tex")
        if ($pass3 -ne 0) { exit $pass3 }
        $pass4 = Invoke-NativeCommand -FilePath "pdflatex" -Arguments @("-interaction=nonstopmode", "-halt-on-error", "paper.tex")
        if ($pass4 -ne 0) { exit $pass4 }
        $pass5 = Invoke-NativeCommand -FilePath "pdflatex" -Arguments @("-interaction=nonstopmode", "-halt-on-error", "paper.tex")
        exit $pass5
    }

    Write-Error @"
No TeX compiler found.
Install one of:
  1) MiKTeX (Windows) and ensure latexmk/pdflatex are in PATH
  2) TeX Live with latexmk
Then rerun: .\scripts\compile_paper.ps1
"@
}
finally {
    Pop-Location
}
