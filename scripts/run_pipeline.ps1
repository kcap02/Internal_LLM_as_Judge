<#
.SYNOPSIS
  Canonical pipeline runner. Encodes which stage runs in which interpreter.

.DESCRIPTION
  CPU stages (00/01/02/20) need a working `datasets`; GPU stages (10/11/12)
  need `spectral_trust` and CUDA. Those live in different environments here —
  `datasets` cannot even be imported in the GPU env — so the split is not a
  preference, it is a requirement. This script is the single place that
  knows it.

  Stage 02 is a gate: a FAIL there means a bank would produce artefacts, so
  the run stops rather than spending GPU time.

.PARAMETER Mode
  pilot : <4B panel, capped item count. Validates the pipeline end to end.
  main  : 7-27B panel, full banks, distractors from the disjoint pilot panel.

.PARAMETER Datasets
  Which banks to process. Default: llmbar (shortest prompts, adversarial).

.EXAMPLE
  .\scripts\run_pipeline.ps1 -Mode pilot -Datasets llmbar
  .\scripts\run_pipeline.ps1 -Mode main -Datasets llmbar judgebench -SkipSpectral
#>
[CmdletBinding()]
param(
    [ValidateSet('pilot', 'main')] [string] $Mode = 'pilot',
    [string[]] $Datasets = @('llmbar'),
    [int] $Limit = 400,
    [switch] $SkipSpectral,
    [switch] $SkipSolver
)

$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'

$repo = Split-Path -Parent $PSScriptRoot
$cpu  = 'python'
$gpu  = 'C:\Users\valno\anaconda3\envs\gemma_spectral\python.exe'
if (-not (Test-Path $gpu)) { throw "GPU interpreter not found: $gpu" }
Set-Location $repo

$cfg = if ($Mode -eq 'main') { @('--config', 'configs/main.json') } else { @() }
$panel = if ($Mode -eq 'pilot') { @('--pilot') } else { @() }

function Step($name, $exe, $argv) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    Write-Host "$exe $($argv -join ' ')" -ForegroundColor DarkGray
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { throw "$name failed (exit $LASTEXITCODE)" }
}

# ── CPU: datasets and banks ──────────────────────────────────────────────────
Step '00 download datasets' $cpu (@('scripts/00_download_datasets.py') + $cfg)
Step '01 build judge banks' $cpu (@('scripts/01_build_judge_banks.py') + $cfg)

# ── CPU: confound gate ───────────────────────────────────────────────────────
Write-Host "`n=== 02 confound audit (gate) ===" -ForegroundColor Cyan
$audit = & $cpu scripts/02_audit_confounds.py @cfg @panel --only @Datasets 2>&1
$audit | ForEach-Object { Write-Host $_ }
if ($audit | Select-String -Quiet '\[FAIL\]') {
    throw 'Bank audit reported FAIL — fix it before spending GPU time.'
}

# ── GPU: solver, then rebuild MCQ banks with panel-derived distractors ───────
$mcq = @($Datasets | Where-Object { $_ -in @('mmlu', 'mmlu_pro') })
if ($mcq -and -not $SkipSolver) {
    # In main mode the solver must run over the DISTRACTOR panel (the pilot
    # models), not the judges: that disjointness is what removes C-SELF.
    $solverPanel = if ($Mode -eq 'main') { @('--pilot') } else { @('--pilot') }
    Step '10 solver (MCQ)' $gpu (@('scripts/10_run_solver.py') + $cfg + $solverPanel + @('--only') + $mcq)
    Step '01 rebuild MCQ banks' $cpu (@('scripts/01_build_judge_banks.py') + $cfg + @('--force', '--only') + $mcq)
}

# ── GPU: verdicts + activations (fast), then spectral (slow) ─────────────────
$lim = if ($Mode -eq 'pilot') { @('--limit', "$Limit") } else { @() }
Step '11 judge + activations' $gpu (@('scripts/11_run_judge.py') + $cfg + $panel + $lim + @('--only') + $Datasets)

if (-not $SkipSpectral) {
    $dry = if ($Mode -eq 'pilot') { @('--dry-run', "$Limit") } else { @() }
    Step '12 judge + spectral' $gpu (@('scripts/12_run_judge_spectral.py') + $cfg + $panel + $dry + @('--only') + $Datasets)
}

# ── CPU: analysis ────────────────────────────────────────────────────────────
Step '20 analyse' $cpu (@('scripts/20_analyse.py') + $cfg + @('--only') + $Datasets)

Write-Host "`nDone. Results in results/, provenance logs in logs/." -ForegroundColor Green
Write-Host "Read the CONDITIONAL AUROC and the BH-FDR survivor list; treat" -ForegroundColor Green
Write-Host "pooled AUROC and any SUPPRESSED slice as non-results." -ForegroundColor Green
