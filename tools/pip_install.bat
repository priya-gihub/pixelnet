@echo off
REM One-off dependency installer (separate log so it never contends with a
REM running download that holds logs\runner.log open).
cd /d "%~dp0.."
echo ==== PIP INSTALL START %DATE% %TIME% ==== > "logs\pip.log"
python -u -m pip install pytest seaborn plotly streamlit >> "logs\pip.log" 2>&1
echo ==== PIP EXITCODE=%ERRORLEVEL% ==== >> "logs\pip.log"
