@echo off
REM ---------------------------------------------------------------------------
REM One-time setup for the "Clean Up Audio" feature on the Download & Extract
REM tab. Builds a SIDECAR virtualenv next to this repo and installs the audio
REM separation stack into it.
REM
REM It is deliberately a separate environment: that stack needs
REM huggingface-hub 1.x and numpy 2.x, both of which break this app's pinned
REM transformers 4.52.1 / gradio 6.17.3. Nothing installed here can reach the
REM app's venv.
REM
REM Roughly 4 GB of downloads. Separation models download later, on first use.
REM ---------------------------------------------------------------------------

setlocal
set "REPO_DIR=%~dp0"
set "ROOT_DIR=%REPO_DIR%.."
set "VENV_DIR=%ROOT_DIR%\venv_audioclean"

echo.
echo Building the audio cleanup environment in:
echo   %VENV_DIR%
echo.

REM Python 3.11 preferred; audio-separator supports 3.10 through 3.13.
set "BASE_PYTHON="
for %%V in (3.11 3.12 3.10 3.13) do (
    if not defined BASE_PYTHON (
        py -%%V -c "import sys" >nul 2>&1 && set "BASE_PYTHON=py -%%V"
    )
)

if not defined BASE_PYTHON (
    echo ERROR: No suitable Python found. Install Python 3.11 from python.org
    echo        and run this script again.
    exit /b 1
)

echo Using base interpreter: %BASE_PYTHON%
%BASE_PYTHON% -m venv "%VENV_DIR%" || exit /b 1

set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

"%VENV_PYTHON%" -m pip install --upgrade pip setuptools wheel || exit /b 1

REM CUDA build first, so audio-separator does not pull the CPU-only torch.
"%VENV_PYTHON%" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124 || exit /b 1

REM The [cpu] extra refers to ONNX runtime only. The torch-backed models this
REM app defaults to still run on the GPU through the CUDA torch installed
REM above; onnxruntime-gpu is skipped because on Windows it needs hand-matched
REM CUDA and cuDNN DLLs to load at all.
"%VENV_PYTHON%" -m pip install "audio-separator[cpu]" speechbrain silero-vad scikit-learn soundfile || exit /b 1

echo.
echo Verifying...
"%VENV_PYTHON%" -c "import torch, audio_separator, speechbrain, silero_vad, sklearn; print('cleanup env OK - CUDA:', torch.cuda.is_available())" || exit /b 1

echo.
echo Done. Restart the app and the Clean Up Audio panel will be enabled.
endlocal
