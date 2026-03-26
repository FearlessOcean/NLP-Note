@echo off
setlocal

:: Get current Date and Time (Format: YYYY-MM-DD HH:MM)
set "datestr=%date:~0,10% %time:~0,5%"

echo --- 1. Pulling from remote ---
git pull
if %errorlevel% neq 0 (
    echo [ERROR] Pull failed. Please check for conflicts or network issues.
    pause
    exit /b %errorlevel%
)

echo.
echo --- 2. Adding changes ---
git add .

echo.
echo --- 3. Committing changes: %datestr% ---
git commit -m "%datestr%"
if %errorlevel% neq 0 (
    echo [INFO] No changes to commit.
)

echo.
echo --- 4. Pushing to remote ---
git push
if %errorlevel% neq 0 (
    echo [ERROR] Push failed.
    pause
    exit /b %errorlevel%
)

echo.
echo --- Success! ---
pause