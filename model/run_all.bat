@echo off
REM PIXEL-VerifyNet model showcase launcher (Option B).
REM Runs the untouched terminal presentation, then opens results.html at the end.
cd /d "%~dp0"
if not exist "run_models.py" (
  echo ERROR: run_models.py not found in %CD%
  pause
  exit /b 1
)
if not exist "results.html" (
  echo ERROR: results.html not found in %CD%
  pause
  exit /b 1
)
python run_models.py %*
echo.
echo Opening results page: %CD%\results.html
set PAGE=%CD%\results.html
REM Try installed browsers explicitly first (works even with no .html association).
if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
  start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" "%PAGE%"
  goto opened
)
if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" (
  start "" "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" "%PAGE%"
  goto opened
)
if exist "C:\Program Files\Microsoft\Edge\Application\msedge.exe" (
  start "" "C:\Program Files\Microsoft\Edge\Application\msedge.exe" "%PAGE%"
  goto opened
)
if exist "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" (
  start "" "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" "%PAGE%"
  goto opened
)
REM Fallback: let Windows resolve the .html association.
start "" "%PAGE%"
:opened
echo If the page did not open, open this file manually:
echo   %PAGE%

