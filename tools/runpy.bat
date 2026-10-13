@echo off
REM Background runner: python -u <args> >> logs\runner.log
cd /d "%~dp0.."
echo ==== START %DATE% %TIME% : python %* ==== >> "logs\runner.log"
python -u %* >> "logs\runner.log" 2>&1
echo ==== EXITCODE=%ERRORLEVEL% : python %* ==== >> "logs\runner.log"
