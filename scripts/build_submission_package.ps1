[CmdletBinding()]
param(
    [string]$OutDir = "submission",
    [string]$ReproZipName = "MLPaper_reproducibility_package_20260314.zip",
    [string]$PaperZipName = "MLPaper_paper_sources_20260314.zip"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Copy-RelativePath {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot,
        [Parameter(Mandatory = $true)][string]$StageDir,
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [switch]$Required
    )
    $src = Join-Path $RepoRoot $RelativePath
    if (-not (Test-Path $src)) {
        if ($Required) {
            throw "[package] missing required path: $RelativePath"
        }
        Write-Warning "[package] skip missing: $RelativePath"
        return
    }
    $dest = Join-Path $StageDir $RelativePath
    $destParent = Split-Path -Parent $dest
    if ($destParent) {
        New-Item -ItemType Directory -Force -Path $destParent | Out-Null
    }
    Copy-Item -Path $src -Destination $dest -Recurse -Force
}

function Write-Manifest {
    param(
        [Parameter(Mandatory = $true)][string]$StageDir,
        [Parameter(Mandatory = $true)][string]$ManifestPath
    )
    Get-ChildItem -Path $StageDir -Recurse -File |
        ForEach-Object {
            $_.FullName.Substring($StageDir.Length + 1).Replace("\", "/")
        } |
        Sort-Object |
        Set-Content -Path $ManifestPath -Encoding UTF8
}

function Write-ZipHash {
    param(
        [Parameter(Mandatory = $true)][string]$ZipPath,
        [Parameter(Mandatory = $true)][string]$HashPath
    )
    $zipHash = (Get-FileHash -Algorithm SHA256 -Path $ZipPath).Hash
    $zipName = Split-Path -Leaf $ZipPath
    Set-Content -Path $HashPath -Value "$zipHash  $zipName" -Encoding UTF8
    return $zipHash
}

function New-ZipFromStage {
    param(
        [Parameter(Mandatory = $true)][string]$StageDir,
        [Parameter(Mandatory = $true)][string]$ZipPath
    )
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::CreateFromDirectory(
        $StageDir,
        $ZipPath,
        [System.IO.Compression.CompressionLevel]::Optimal,
        $false
    )
}

function Set-JsonProperty {
    param(
        [Parameter(Mandatory = $true)]$Object,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)]$Value
    )
    if ($Object.PSObject.Properties.Name -contains $Name) {
        $Object.$Name = $Value
    }
    else {
        $Object | Add-Member -NotePropertyName $Name -NotePropertyValue $Value
    }
}

function Get-Utf8Sha256 {
    param(
        [Parameter(Mandatory = $true)][string]$Text
    )
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
        return (($sha.ComputeHash($bytes) | ForEach-Object { $_.ToString("x2") }) -join "")
    }
    finally {
        $sha.Dispose()
    }
}

function Get-ManifestPayloadSha256 {
    param(
        [Parameter(Mandatory = $true)][string]$PathValue
    )
    $lines = Get-Content -Path $PathValue
    if (-not $lines -or $lines.Count -le 1) {
        return Get-Utf8Sha256 -Text ""
    }
    $payload = (($lines | Select-Object -Skip 1) | ForEach-Object { "$_`n" }) -join ""
    return Get-Utf8Sha256 -Text $payload
}

function Update-StagedRunMetadata {
    param(
        [Parameter(Mandatory = $true)][string]$StageDir
    )
    $runMetadataPath = Join-Path $StageDir "results\run_metadata.json"
    if (-not (Test-Path $runMetadataPath)) {
        return
    }

    $metadata = Get-Content -Path $runMetadataPath -Raw | ConvertFrom-Json
    if (-not $metadata.code_fingerprint) {
        $metadata | Add-Member -NotePropertyName code_fingerprint -NotePropertyValue ([pscustomobject]@{})
    }
    if (-not $metadata.configuration) {
        $metadata | Add-Member -NotePropertyName configuration -NotePropertyValue ([pscustomobject]@{})
    }
    if (-not $metadata.artifact_rows) {
        $metadata | Add-Member -NotePropertyName artifact_rows -NotePropertyValue ([pscustomobject]@{})
    }

    $snapshotManifestRel = "results/code_snapshot_manifest.csv"
    $snapshotManifestPath = Join-Path $StageDir $snapshotManifestRel
    if (Test-Path $snapshotManifestPath) {
        $snapshotManifestSha = (Get-FileHash -Algorithm SHA256 -Path $snapshotManifestPath).Hash.ToLower()
        $snapshotPayloadSha = Get-ManifestPayloadSha256 -PathValue $snapshotManifestPath
        $snapshotCount = (Import-Csv $snapshotManifestPath | Measure-Object).Count

        Set-JsonProperty -Object $metadata.code_fingerprint -Name "code_snapshot_manifest_path" -Value $snapshotManifestRel
        Set-JsonProperty -Object $metadata.code_fingerprint -Name "code_snapshot_manifest_sha256" -Value $snapshotManifestSha
        Set-JsonProperty -Object $metadata.code_fingerprint -Name "code_snapshot_manifest_payload_sha256" -Value $snapshotPayloadSha
        Set-JsonProperty -Object $metadata.code_fingerprint -Name "code_snapshot_file_count" -Value ([string]$snapshotCount)
        Set-JsonProperty -Object $metadata.code_fingerprint -Name "immutable_snapshot_sha256" -Value $snapshotPayloadSha
        Set-JsonProperty -Object $metadata.artifact_rows -Name "code_snapshot_manifest" -Value $snapshotCount
    }

    $datasetManifestRel = "results/dataset_hash_manifest.csv"
    $datasetManifestPath = Join-Path $StageDir $datasetManifestRel
    if (Test-Path $datasetManifestPath) {
        Set-JsonProperty -Object $metadata.configuration -Name "dataset_hash_manifest_path" -Value $datasetManifestRel
        Set-JsonProperty -Object $metadata.configuration -Name "dataset_hash_manifest_sha256" -Value ((Get-FileHash -Algorithm SHA256 -Path $datasetManifestPath).Hash.ToLower())
        Set-JsonProperty -Object $metadata.artifact_rows -Name "dataset_hash_manifest" -Value ((Import-Csv $datasetManifestPath | Measure-Object).Count)
    }

    $metadata | ConvertTo-Json -Depth 100 | Set-Content -Path $runMetadataPath -Encoding UTF8
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$outPath = Join-Path $repoRoot $OutDir
$buildRoot = Join-Path $repoRoot ".build\submission"
$reproStageDir = Join-Path $buildRoot "repro_stage"
$paperStageDir = Join-Path $buildRoot "paper_stage"
$reproZipPath = Join-Path $outPath $ReproZipName
$paperZipPath = Join-Path $outPath $PaperZipName
$reproManifestPath = Join-Path $outPath "package_manifest.txt"
$paperManifestPath = Join-Path $outPath "paper_source_manifest.txt"

New-Item -ItemType Directory -Force -Path $outPath | Out-Null
if (Test-Path $buildRoot) {
    Remove-Item -Path $buildRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $reproStageDir | Out-Null
New-Item -ItemType Directory -Force -Path $paperStageDir | Out-Null

$reproPaths = @(
    "README.md",
    "requirements.txt",
    "requirements-lock.txt",
    "run_experiments.py",
    "datasets.py",
    "preprocess.py",
    "corruptions.py",
    "models.py",
    "metrics.py",
    "calibration.py",
    "plotting.py",
    "paper/paper.pdf",
    "paper/paper.tex",
    "paper/references.bib",
    "paper/AuthorGuide/einformatica.cls",
    "paper/AuthorGuide/IEEEtran_for_EI.bst",
    "paper/AuthorGuide/EISEJ_logo.png",
    "paper/AuthorGuide/ORCID.pdf",
    "results/run_metadata.json",
    "results/code_snapshot_manifest.csv",
    "results/dataset_hash_manifest.csv",
    "results/metrics_summary.csv",
    "results/raw/metrics_long.csv",
    "results/raw/policy_metrics_long.csv",
    "results/raw/calibration_metrics_long.csv",
    "results/raw/corruption_diagnostics_long.csv",
    "results/raw/effects_seed_paired.csv",
    "results/raw/table_calibration_pairwise_tests.csv",
    "results/raw/table_cross_dataset_effects.csv",
    "results/raw/table_primary_dataset_inference_summary.csv",
    "results/raw/table_primary_hypothesis_tests.csv",
    "results/raw/table_threshold_policy_pairwise_tests.csv",
    "results/tables",
    "docs/compile_check_latest.txt",
    "docs/paper_numbers_latest.txt",
    "docs/paper_numbers_latest.json",
    "docs/paper_numbers_validation.json",
    "docs/statistical_analysis_plan.md",
    "docs/dataset_protocol.md",
    "scripts/build_latex_tables.py",
    "scripts/regenerate_derived_tables.py",
    "scripts/extract_paper_numbers.py",
    "scripts/generate_dataset_hash_manifest.py",
    "scripts/compile_paper.ps1",
    "scripts/rerun_pipeline.ps1",
    "scripts/c1_anchor_sensitivity.py",
    "scripts/build_submission_package.ps1"
)

foreach ($rel in $reproPaths) {
    Copy-RelativePath -RepoRoot $repoRoot -StageDir $reproStageDir -RelativePath $rel -Required
}

$optionalReproPaths = @(
    "docs/paper_numbers_validation_20260314.json"
)
foreach ($rel in $optionalReproPaths) {
    Copy-RelativePath -RepoRoot $repoRoot -StageDir $reproStageDir -RelativePath $rel
}

$figurePaths = @(
    "results/plots/paper__cross_dataset__severe_deltas.png",
    "results/plots/paper__calibration__delta_vs_uncal.png",
    "results/plots/paper__policy__summary.png",
    "results/plots/credit_default__ignore_unknown__hist_gb__C2__auc.png",
    "results/plots/credit_default__ignore_unknown__hist_gb__C2__f1.png",
    "results/plots/credit_default__ignore_unknown__hist_gb__C2plusC4__uncal__auc_heatmap.png",
    "results/plots/electricity__ignore_unknown__hist_gb__C2plusC4__uncal__ece_heatmap.png",
    "results/plots/plot_index.csv",
    "results/plots/figure_qc_checklist.txt"
)
foreach ($rel in $figurePaths) {
    Copy-RelativePath -RepoRoot $repoRoot -StageDir $reproStageDir -RelativePath $rel -Required
}

$paperSourcePaths = @(
    "paper/paper.tex",
    "paper/references.bib",
    "paper/AuthorGuide/einformatica.cls",
    "paper/AuthorGuide/IEEEtran_for_EI.bst",
    "paper/AuthorGuide/EISEJ_logo.png",
    "paper/AuthorGuide/ORCID.pdf"
)

foreach ($rel in $paperSourcePaths) {
    Copy-RelativePath -RepoRoot $repoRoot -StageDir $paperStageDir -RelativePath $rel -Required
}

Update-StagedRunMetadata -StageDir $reproStageDir
Write-Manifest -StageDir $reproStageDir -ManifestPath $reproManifestPath
Write-Manifest -StageDir $paperStageDir -ManifestPath $paperManifestPath

if (Test-Path $reproZipPath) {
    Remove-Item -Path $reproZipPath -Force
}
if (Test-Path $paperZipPath) {
    Remove-Item -Path $paperZipPath -Force
}
New-ZipFromStage -StageDir $reproStageDir -ZipPath $reproZipPath
New-ZipFromStage -StageDir $paperStageDir -ZipPath $paperZipPath

$reproHashPath = Join-Path $outPath ($ReproZipName + ".sha256.txt")
$paperHashPath = Join-Path $outPath ($PaperZipName + ".sha256.txt")
$reproHash = Write-ZipHash -ZipPath $reproZipPath -HashPath $reproHashPath
$paperHash = Write-ZipHash -ZipPath $paperZipPath -HashPath $paperHashPath

if (Test-Path $buildRoot) {
    Remove-Item -Path $buildRoot -Recurse -Force
}

Write-Host "[package] reproducibility zip: $reproZipPath"
Write-Host "[package] reproducibility sha256: $reproHash"
Write-Host "[package] paper source zip: $paperZipPath"
Write-Host "[package] paper source sha256: $paperHash"
