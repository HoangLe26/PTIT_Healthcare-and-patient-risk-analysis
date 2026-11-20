$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$deckPath = Join-Path $projectRoot 'reports/presentation.pptx'
$qaPath = Join-Path $projectRoot 'tmp/qa/slides'
New-Item -ItemType Directory -Path $qaPath -Force | Out-Null
$alreadyRunning = @(Get-Process POWERPNT -ErrorAction SilentlyContinue).Count -gt 0
$powerPointApp = New-Object -ComObject PowerPoint.Application
$deck = $null
try {
    # Read-only and WithWindow=false. Do not display or alter an existing user deck.
    $deck = $powerPointApp.Presentations.Open($deckPath, -1, 0, 0)
    for ($i = 1; $i -le $deck.Slides.Count; $i++) {
        $outputFile = Join-Path $qaPath ('slide_{0:d2}.png' -f $i)
        $deck.Slides.Item($i).Export($outputFile, 'PNG', 1600, 900)
    }
    Write-Output ('Rendered {0} slides using PowerPoint' -f $deck.Slides.Count)
} finally {
    if ($null -ne $deck) { $deck.Close() }
    if (-not $alreadyRunning) { $powerPointApp.Quit() }
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($powerPointApp) | Out-Null
}

