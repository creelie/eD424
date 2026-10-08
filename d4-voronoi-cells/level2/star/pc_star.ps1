# Run the star programme (build_part_star.jl, solved by ../fast/solve_fast.jl) on a Windows
# machine: B = 127/250 (24 centres within 2.0161 of c), BL = 0.6141 (a further centre within
# sqrt 6), at the degrees (8, 10) and then (10, 12).  A value below 25 would say, in floating
# point, that the 24 leave no room for the further centre.  Exploration, not a proof.
#
# It installs what it needs on first use and skips what is already there:
#   Julia 1.10.9 (portable zip) in %USERPROFILE%\julia-1.10.9,
#   the package of de Laat, Leijenhorst and de Muinck Keizer (sources only, from
#   third_party\llm24-certificate) in %USERPROFILE%\lsc\LasserreSphericalCodes, with the
#   zonal cache of level2\zonalstore_4.tar.gz and its Julia dependencies.
# Chunk files go to %USERPROFILE%\star_runs; logs to level2\star\pcruns.
#   powershell -ExecutionPolicy Bypass -File level2\star\pc_star.ps1 [-Threads 8] [-Degrees "8,10;10,12"]
param([int]$Threads = 8, [string]$Degrees = "8,10;10,12")

$ErrorActionPreference = 'Continue'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$lvl2 = Split-Path -Parent $here
$repo = Split-Path -Parent $lvl2
$logs = Join-Path $here 'pcruns'
New-Item -ItemType Directory -Force $logs | Out-Null
$log = Join-Path $logs 'pc_star.log'
function Say($m) {
    $t = (Get-Date).ToUniversalTime().ToString('yyyy-MM-dd HH:mm:ss')
    "$t UTC  $m" | Out-File -FilePath $log -Append -Encoding ascii
}
function Fetch($url, $out) {
    for ($i = 1; $i -le 60; $i++) {
        & curl.exe -sS -L -C - --retry 5 --retry-delay 10 -o $out $url
        if ($LASTEXITCODE -eq 0) { return $true }
        Say "download of $url interrupted (curl exit $LASTEXITCODE), retry $i"
        Start-Sleep -Seconds 20
    }
    return $false
}

function RunJulia($argList, $outFile) {
    $errFile = $outFile -replace '\.log$', '.err'
    $p = Start-Process -FilePath $julia -ArgumentList $argList -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $outFile -RedirectStandardError $errFile
    return $p.ExitCode
}

Say "start: threads $Threads, degrees $Degrees, repo $repo"
$mem = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1)
Say "free memory $mem GB, $([Environment]::ProcessorCount) logical cores"

# 1. Julia
$jdir = Join-Path $env:USERPROFILE 'julia-1.10.9'
$julia = Join-Path $jdir 'bin\julia.exe'
if (-not (Test-Path $julia)) {
    $zip = Join-Path $env:USERPROFILE 'julia-1.10.9-win64.zip'
    Say 'downloading Julia 1.10.9'
    if (-not (Fetch 'https://julialang-s3.julialang.org/bin/winnt/x64/1.10/julia-1.10.9-win64.zip' $zip)) { Say 'FAILED: Julia download'; exit 1 }
    Say 'unpacking Julia'
    & tar.exe -xf $zip -C $env:USERPROFILE
    if (-not (Test-Path $julia)) { Say "FAILED: $julia not found after unpacking"; exit 1 }
}
Say ("julia: " + (& $julia --version))

# 2. The package sources and the zonal cache
$lscroot = Join-Path $env:USERPROFILE 'lsc'
$lsc = Join-Path $lscroot 'LasserreSphericalCodes'
if (-not (Test-Path (Join-Path $lsc 'Project.toml'))) {
    New-Item -ItemType Directory -Force $lscroot | Out-Null
    $parts = Join-Path $repo 'third_party\llm24-certificate'
    $zip = Join-Path $lscroot 'LasserreSphericalCodes.zip'
    Say 'joining the archive of the authors'
    $fs = [IO.File]::Create($zip)
    foreach ($part in 'part-00', 'part-01') {
        $in = [IO.File]::OpenRead((Join-Path $parts "LasserreSphericalCodes.zip.$part")); $in.CopyTo($fs); $in.Close()
    }
    $fs.Close()
    $md5 = (Get-FileHash -Algorithm MD5 $zip).Hash.ToLower()
    Say "archive md5 $md5 (expected 02acd5270f7b3fa799abdeb5291706fd)"
    if ($md5 -ne '02acd5270f7b3fa799abdeb5291706fd') { Say 'FAILED: archive checksum'; exit 1 }
    & tar.exe -xf $zip -C $lscroot 'LasserreSphericalCodes/src/*' LasserreSphericalCodes/Project.toml LasserreSphericalCodes/Manifest.toml
    Remove-Item $zip
}
$zs = Join-Path $lsc 'cache\zonalstore'
if (-not (Test-Path (Join-Path $zs '4'))) {
    New-Item -ItemType Directory -Force $zs | Out-Null
    & tar.exe -xzf (Join-Path $lvl2 'zonalstore_4.tar.gz') -C $zs
    Get-ChildItem (Join-Path $zs '4') -Filter 'pol-4-2-*.txt' | ForEach-Object {
        New-Item -ItemType File -Force (Join-Path $zs ('pol-2-' + $_.Name.Substring(8))) | Out-Null
    }
    Say ("zonal entries installed: " + (Get-ChildItem (Join-Path $zs '4')).Count)
}

# 3. The Julia dependencies (retried, since the downloads may break off)
$ok = $false
for ($i = 1; $i -le 20 -and -not $ok; $i++) {
    Say "Pkg.instantiate, attempt $i"
    $code = RunJulia @("--project=$lsc", "$here\pc_pkgs.jl") "$logs\pkg_$i.log"
    if ($code -eq 0) { $ok = $true } else { Say "Pkg attempt $i exit $code"; Start-Sleep -Seconds 30 }
}
if (-not $ok) { Say 'FAILED: Pkg.instantiate (see pkg_*.log)'; exit 1 }
Say 'packages ready'

# 4. The star programme at each pair of degrees
foreach ($dd in $Degrees.Split(';')) {
    $d1, $dl = $dd.Split(',')
    $tag = "s$d1$dl"
    $out = Join-Path (Join-Path $env:USERPROFILE 'star_runs') $tag
    New-Item -ItemType Directory -Force $out | Out-Null
    foreach ($p in 0, 1) {
        if (Test-Path "$out\built$p.txt") { continue }
        Say "build ($d1, $dl) part $p"
        $code = RunJulia @("--project=$lsc", '-t', '1', "$here\pc_build_star.jl", $d1, $dl, '128', '20', '127//250', '6141//10000', "$p", '1', $out) "$logs\$tag.part$p.log"
        if ($code -ne 0) { Say "FAILED: build ($d1, $dl) part $p, exit $code"; exit 1 }
        'done' | Out-File "$out\built$p.txt"
    }
    Say "solve ($d1, $dl) on $Threads threads"
    $env:CKPT = "$out.ckpt.jls"; $env:CKPT_EVERY = '5'; $env:OZ_K = '7'; $env:CHOL_K = '8'
    $env:CHOL_L = '10'; $env:OMEGA_EXP = '3'
    $code = RunJulia @("--project=$lsc", '-t', "$Threads", "$lvl2\fast\pc_solve.jl", $out, '128', '200', 'fast', '7', '1e-6') "$logs\$tag.solve.log"
    Say "solve ($d1, $dl) ended, exit $code"
}
Say 'all done'
