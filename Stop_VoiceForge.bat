@echo off
REM Stops the AIO VoiceForge server started by Start_VoiceForge.bat.
REM Takes an optional port as the first argument; 7860 is webui.py's default.
setlocal EnableDelayedExpansion

set PORT=%~1
if not defined PORT set PORT=7860

set FOUND=0
REM Start_VoiceForge watches for this flag: taskkill /F makes the app exit
REM nonzero, and without the flag its window would read that as a crash and
REM sit waiting for a keypress. Written before the kill so it cannot lose
REM the race against a fast shutdown.
set "STOPFLAG=%TEMP%\voiceforge_stop.flag"
echo stopped by Stop_VoiceForge.bat> "%STOPFLAG%"
REM The pipe is deliberately NOT caret-escaped. Inside the quoted -Command
REM string cmd leaves a caret literal, PowerShell then fails to parse it, and
REM the empty capture reads as "not running" against a live server.
for /f "delims=" %%P in ('powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique"') do (
    REM PID 0 is the system idle process - never a server, never killable.
    if not "%%P"=="0" (
        echo Stopping VoiceForge process tree at PID %%P ...
        REM /T matters: the TTS engine and cleanup sidecar are children of the
        REM webui process and would outlive it holding VRAM.
        taskkill /PID %%P /T /F
        set FOUND=1
    )
)

if !FOUND!==0 (
    echo Nothing is listening on port !PORT! - VoiceForge does not appear to be running.
    REM Nothing was killed, so the flag would only mislead the next start.
    del "%STOPFLAG%" >nul 2>&1
    call :LINGER
    exit /b 0
)

REM taskkill returns before a loaded model has unmapped, so the port can stay
REM held for seconds after the kill. Poll until it is actually free, or the
REM next start fails with a port already in use.
set POLL_COUNT=0
:POLL_LOOP
set /a POLL_COUNT+=1
set PORT_BUSY=0
for /f "delims=" %%C in ('powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique"') do (
    if not "%%C"=="0" set PORT_BUSY=1
)
if !PORT_BUSY!==0 goto :PORT_FREE
if !POLL_COUNT! GEQ 10 goto :PORT_STUCK
REM timeout /t fails when stdin is redirected, which it is under a shortcut.
ping -n 2 127.0.0.1 >nul
goto :POLL_LOOP

:PORT_FREE
echo VoiceForge stopped - port !PORT! is free.
call :LINGER
exit /b 0

:PORT_STUCK
REM The one outcome worth reading, so this window waits to be dismissed.
echo WARNING: port !PORT! is still held after 10 seconds. Check Task Manager.
pause
exit /b 1

REM Long enough to read the confirmation, short enough not to be in the way.
REM ping rather than timeout, which fails whenever stdin is not a console.
:LINGER
ping -n 3 127.0.0.1 >nul
goto :eof
