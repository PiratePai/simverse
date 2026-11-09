@echo off
rem ------------------------------------------------------------------
rem simverse workbench one-click launcher (double-click to start)
rem   - starts local server (minimized window titled "simverse-server")
rem   - opens default browser when ready
rem   - stop: close that window, or run the stop-workbench .bat
rem   optional arg: port (default 8765), e.g.  launcher 9000
rem ------------------------------------------------------------------
setlocal
cd /d "%~dp0"

set PORT=8765
if not "%~1"=="" set PORT=%~1
set URL=http://127.0.0.1:%PORT%/

set PYRUN=py -3.12
where py >nul 2>nul || set PYRUN=python

rem already running? just open the browser
curl -s -o nul "%URL%"
if not errorlevel 1 goto already

echo [simverse] starting server at %URL% ...
start "simverse-server" /min %PYRUN% -m simverse.cli serve --host 127.0.0.1 --port %PORT%

set /a TRIES=0
:wait
ping -n 2 127.0.0.1 >nul
curl -s -o nul "%URL%"
if not errorlevel 1 goto ready
set /a TRIES+=1
if %TRIES% lss 30 goto wait
echo [simverse] ERROR: server not ready in ~30s. Check the minimized
echo [simverse] "simverse-server" window for the traceback.
pause
exit /b 1

:ready
echo [simverse] ready: %URL%
if defined SIMVERSE_NO_BROWSER (
  echo [simverse] browser open skipped ^(SIMVERSE_NO_BROWSER^).
) else (
  start "" "%URL%"
  echo [simverse] browser opened.
)
echo [simverse] stop: close the minimized "simverse-server" window,
echo [simverse] or run the stop-workbench .bat
ping -n 4 127.0.0.1 >nul
exit /b 0

:already
echo [simverse] server already running at %URL%
if not defined SIMVERSE_NO_BROWSER start "" "%URL%"
ping -n 3 127.0.0.1 >nul
exit /b 0
