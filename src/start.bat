@echo off
:: Wechselt in den Hauptordner (eine Ebene höher), damit die venv und Pfade greifen
cd /d %~dp0\..

if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe src\run.py
) else (
    python src\run.py
)
pause