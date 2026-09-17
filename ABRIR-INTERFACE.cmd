@echo off
rem ---------------------------------------------------------------------------
rem  Abre a INTERFACE do Video Dublador no navegador.
rem
rem  Basta dar dois cliques neste arquivo. Nao e preciso digitar nada.
rem  Deixe esta janela aberta enquanto usa a interface; feche-a para sair.
rem ---------------------------------------------------------------------------
setlocal
set "ROOT=%~dp0"
set "PYTHONPATH=%ROOT%"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
title Video Dublador - Interface

if not exist "%ROOT%.python\python.exe" (
  echo.
  echo   ERRO: Python portatil nao encontrado em "%ROOT%.python\python.exe"
  echo.
  pause
  exit /b 1
)

echo.
echo   Abrindo a interface do Video Dublador...
echo   O navegador deve abrir sozinho. Se nao abrir, use o endereco mostrado abaixo.
echo.

"%ROOT%.python\python.exe" -m dublador gui %*

echo.
echo   Interface encerrada.
pause
