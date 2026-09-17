@echo off
rem ---------------------------------------------------------------------------
rem  Video dubber launcher (Windows cmd)
rem
rem    dublar.cmd check
rem    dublar.cmd languages
rem    dublar.cmd input\movie.mp4 --target pt
rem ---------------------------------------------------------------------------
setlocal
set "ROOT=%~dp0"
set "PYTHONPATH=%ROOT%"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if not exist "%ROOT%.python\python.exe" (
  echo.
  echo   Portable Python not found at "%ROOT%.python\python.exe"
  echo   Run the bootstrap first:  python tools\bootstrap.py
  echo.
  exit /b 1
)

"%ROOT%.python\python.exe" -m dublador %*
exit /b %ERRORLEVEL%
