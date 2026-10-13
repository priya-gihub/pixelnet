"""List running python processes with their command lines (Windows)."""
import subprocess

ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
      "ForEach-Object { $_.ProcessId.ToString() + ' | ' + $_.CommandLine }")
try:
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                         capture_output=True, text=True, timeout=25)
    lines = [l for l in out.stdout.splitlines() if l.strip()]
    print(f"python processes: {len(lines)}")
    for l in lines:
        print("  " + l[:160])
except Exception as e:
    print("error:", repr(e))