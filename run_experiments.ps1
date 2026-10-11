param([string]$Python = 'C:\Users\28405\anaconda3\envs\env\python.exe', [int]$Epochs = 1500, [switch]$Overwrite)
$ErrorActionPreference = 'Stop'
$extraArgs = @()
if ($Overwrite) { $extraArgs = @('--overwrite') }
foreach ($split in @('official','grouped')) {
    foreach ($model in @('baseline','pooled')) {
        & $Python -u "$PSScriptRoot/experiment.py" --epochs $Epochs --split $split --model $model --fast @extraArgs
        if ($LASTEXITCODE -ne 0) { throw "Experiment failed: $split $model" }
    }
}
