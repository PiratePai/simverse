@echo off
rem stop the simverse workbench server (window titled "simverse-server" + children)
setlocal
taskkill /FI "WINDOWTITLE eq simverse-server*" /T /F >nul 2>nul
if errorlevel 1 (
  echo [simverse] no running "simverse-server" window found.
) else (
  echo [simverse] server stopped.
)
ping -n 3 127.0.0.1 >nul
