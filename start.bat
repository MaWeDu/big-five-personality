@echo off
REM Move to the src folder relative to this batch file.
cd /d "%~dp0src"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "run.py"
) else (
    python "run.py"
)

pause