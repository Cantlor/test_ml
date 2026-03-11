param(
    [string]$Config = "$(Join-Path $PSScriptRoot '..\configs\train_config.yaml')",
    [string]$Hardware = "$(Join-Path $PSScriptRoot '..\configs\hardware_config.yaml')",
    [string]$RunId = "$(Get-Date -AsUTC -Format 'yyyyMMdd_HHmmss')",
    [string]$LogLevel = 'INFO',
    [switch]$SkipCheck,
    [switch]$SkipPredict,
    [switch]$SkipEval,
    [switch]$TrainInfer
)

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$modRoot = (Resolve-Path (Join-Path $scriptDir '..')).Path
$projRoot = (Resolve-Path (Join-Path $modRoot '..')).Path

function Resolve-OptionalPath([string]$PathValue, [string]$Name) {
    try {
        return (Resolve-Path $PathValue).Path
    }
    catch {
        throw "$Name not found: $PathValue"
    }
}

$Config = Resolve-OptionalPath -PathValue $Config -Name 'Config'
$Hardware = Resolve-OptionalPath -PathValue $Hardware -Name 'Hardware config'

$venvPython = Join-Path $projRoot '.venv\Scripts\python.exe'
$pythonExe = $null
$pythonBaseArgs = @()

if (Test-Path $venvPython) {
    $pythonExe = $venvPython
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonExe = 'python'
}
elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonExe = 'py'
    $pythonBaseArgs = @('-3.12')
}
else {
    throw 'Python was not found. Install Python 3.12 or create .venv.'
}

function Invoke-Python {
    param([string[]]$Args)
    & $pythonExe @pythonBaseArgs @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with code $LASTEXITCODE"
    }
}

Push-Location $modRoot
try {
    $env:CONFIG_PATH = $Config
    $env:RUN_ID_VALUE = $RunId

    $runDir = (& $pythonExe @pythonBaseArgs -c "import os; from pathlib import Path; from net_train.config import load_train_config; cfg = load_train_config(os.environ['CONFIG_PATH']); runs_root = cfg.paths.get('runs_root', (cfg.project_root / 'output_data/module_net_train/runs').resolve()); print((Path(runs_root) / os.environ['RUN_ID_VALUE']).resolve())").Trim()

    if (Test-Path $runDir) {
        throw "Run directory already exists: $runDir (use another -RunId)"
    }

    Write-Host ''
    Write-Host '==> Environment' -ForegroundColor Cyan
    Write-Host "module_net_train: $modRoot"
    Write-Host "project_root:      $projRoot"
    Write-Host "python:            $pythonExe $($pythonBaseArgs -join ' ')"
    Write-Host "config:            $Config"
    Write-Host "hardware:          $Hardware"
    Write-Host "run_id:            $RunId"
    Write-Host "run_dir:           $runDir"

    if (-not $SkipCheck) {
        Write-Host ''
        Write-Host '==> 01_check_prep_data.py' -ForegroundColor Cyan
        Invoke-Python -Args @(
            'scripts/01_check_prep_data.py',
            '--config', $Config,
            '--out_json', (Join-Path $runDir 'prep_data_summary.json'),
            '--log_level', $LogLevel
        )
        Write-Host '[OK] 01_check_prep_data finished' -ForegroundColor Green
    }
    else {
        Write-Host '[WARN] Skipping 01_check_prep_data' -ForegroundColor Yellow
    }

    Write-Host ''
    Write-Host '==> 02_train.py' -ForegroundColor Cyan
    $trainArgs = @(
        'scripts/02_train.py',
        '--config', $Config,
        '--hardware', $Hardware,
        '--run_id', $RunId,
        '--log_level', $LogLevel
    )
    if (-not $TrainInfer) {
        $trainArgs += '--no_infer'
    }
    Invoke-Python -Args $trainArgs
    Write-Host '[OK] 02_train finished' -ForegroundColor Green

    if (-not $SkipPredict) {
        Write-Host ''
        Write-Host '==> 03_predict_aoi.py' -ForegroundColor Cyan
        Invoke-Python -Args @(
            'scripts/03_predict_aoi.py',
            '--config', $Config,
            '--hardware', $Hardware,
            '--run_dir', $runDir,
            '--log_level', $LogLevel
        )
        Write-Host '[OK] 03_predict_aoi finished' -ForegroundColor Green
    }
    else {
        Write-Host '[WARN] Skipping 03_predict_aoi' -ForegroundColor Yellow
    }

    if (-not $SkipEval) {
        Write-Host ''
        Write-Host '==> 04_eval.py' -ForegroundColor Cyan
        Invoke-Python -Args @(
            'scripts/04_eval.py',
            '--config', $Config,
            '--hardware', $Hardware,
            '--run_dir', $runDir,
            '--log_level', $LogLevel
        )
        Write-Host '[OK] 04_eval finished' -ForegroundColor Green
    }
    else {
        Write-Host '[WARN] Skipping 04_eval' -ForegroundColor Yellow
    }

    if ((-not $TrainInfer) -and $SkipPredict) {
        Write-Host '[WARN] Inference was not run (02_train with --no_infer and 03_predict skipped).' -ForegroundColor Yellow
    }

    Write-Host ''
    Write-Host '==> DONE' -ForegroundColor Cyan
    Write-Host "[OK] module_net_train pipeline completed: $runDir" -ForegroundColor Green
}
finally {
    Pop-Location
}
