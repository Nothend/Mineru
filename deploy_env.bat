@echo off
chcp 65001 >nul
set SCRIPT_DIR=%~dp0

REM Try Python first (more stable)
where python >nul 2>nul
if %errorlevel%==0 (
    echo [INFO] Python detected, starting unified deploy script...
    python "%SCRIPT_DIR%deploy_env.py"
    if %errorlevel% neq 0 pause
    exit /b
)

echo [ERROR] Python not found, cannot perform deployment.
echo Please install Python 3.10+ and add it to PATH.
pause
