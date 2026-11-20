param([switch]$Smoke)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
function Run-Step([string[]]$Arguments) {
    & $pythonPath @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $Arguments" }
}
Run-Step -Arguments @('-m','healthcare_ml.cli','download','--all')
Run-Step -Arguments @('-m','healthcare_ml.cli','validate-data','--all')
Run-Step -Arguments @('-m','healthcare_ml.cli','prepare','--all')
Run-Step -Arguments @('-m','healthcare_ml.cli','smoke','--all')
if ($Smoke) { Run-Step -Arguments @('-m','pytest','-q'); exit }
Run-Step -Arguments @('-m','healthcare_ml.cli','train','--config','configs/heart.yaml')
Run-Step -Arguments @('-m','healthcare_ml.cli','train','--config','configs/readmission.yaml')
Run-Step -Arguments @('-m','healthcare_ml.cli','eda','--all')
Run-Step -Arguments @('-m','healthcare_ml.cli','ablations','--all')
Run-Step -Arguments @('-m','healthcare_ml.cli','optimization')
Run-Step -Arguments @('-m','healthcare_ml.cli','freeze','--all')
Run-Step -Arguments @('-m','healthcare_ml.cli','evaluate','--all','--split','test')
Run-Step -Arguments @('-m','healthcare_ml.cli','explain','--all')
Run-Step -Arguments @('-m','pytest','-q')
Run-Step -Arguments @('-m','healthcare_ml.cli','report','--all')

