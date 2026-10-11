param([string]$Gh = 'gh', [string]$Name = 'polycrystal-gnn-reproduction')
$ErrorActionPreference = 'Stop'
& $Gh auth status
if ($LASTEXITCODE -ne 0) { throw 'Run gh auth login first, then retry.' }
# Refuse to overwrite an existing repository or silently change its visibility.
$login = & $Gh api user --jq .login
if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve authenticated GitHub account.' }
& $Gh repo view "$login/$Name" --json name 2>$null
if ($LASTEXITCODE -eq 0) { throw 'Repository already exists. Inspect it before configuring an origin or pushing.' }
& $Gh repo create $Name --private --description 'Windows Anaconda reproduction and grouped evaluation of PolycrystalGraph' --source $PSScriptRoot --remote origin --push
if ($LASTEXITCODE -ne 0) { throw 'Repository creation or push failed; inspect the output before retrying.' }
& $Gh repo view "$login/$Name" --json url,isPrivate
if ($LASTEXITCODE -ne 0) { throw 'Cannot verify created repository.' }
