<#
.SYNOPSIS
Creates the Start / Stop VoiceForge shortcuts on the desktop.

Run once:  powershell -ExecutionPolicy Bypass -File tools\make_desktop_shortcuts.ps1

Re-running overwrites the two shortcuts in place, so it is also the way to
repoint them after the repo moves.
#>
$ErrorActionPreference = 'Stop'

# The script lives in tools/, so the repo root is its parent. Resolved rather
# than hardcoded so a moved or cloned repo still makes correct shortcuts.
$RepoRoot = Split-Path -Parent $PSScriptRoot

# This account has both a local and a OneDrive Desktop; only GetFolderPath
# names the one Explorer is actually showing.
$Desktop = [Environment]::GetFolderPath('Desktop')

function New-VoiceForgeShortcut {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Target,
        [Parameter(Mandatory)][string]$Description,
        [Parameter(Mandatory)][string]$Icon
    )

    $shell = New-Object -ComObject WScript.Shell
    try {
        $lnkPath = Join-Path $Desktop "$Name.lnk"
        $shortcut = $shell.CreateShortcut($lnkPath)
        $shortcut.TargetPath = $Target
        $shortcut.WorkingDirectory = $RepoRoot
        $shortcut.Description = $Description

        # Green for start, red for stop: the two sit side by side on the
        # desktop and the shared logo alone does not say which is which.
        # A missing icon must not fail the run - the shortcut still works
        # with the default batch-file icon.
        $iconPath = Join-Path $RepoRoot $Icon
        if (Test-Path $iconPath) {
            $shortcut.IconLocation = $iconPath
        }

        $shortcut.Save()
        Write-Output "Wrote $lnkPath"
    }
    finally {
        [Runtime.InteropServices.Marshal]::ReleaseComObject($shell) | Out-Null
    }
}

New-VoiceForgeShortcut -Name 'Start VoiceForge' `
    -Target (Join-Path $RepoRoot 'Start_VoiceForge.bat') `
    -Description 'Start the AIO VoiceForge server on the RTX 5060 Ti' `
    -Icon 'assets\voiceforge_start.ico'

New-VoiceForgeShortcut -Name 'Stop VoiceForge' `
    -Target (Join-Path $RepoRoot 'Stop_VoiceForge.bat') `
    -Description 'Stop the AIO VoiceForge server' `
    -Icon 'assets\voiceforge_stop.ico'
