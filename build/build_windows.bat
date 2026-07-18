@echo off
REM Gera o executavel do EstatLab para Windows (rodar em uma maquina Windows).
cd /d "%~dp0\.."

.venv\Scripts\pip install --quiet pyinstaller
.venv\Scripts\pyinstaller ^
  --name EstatLab ^
  --windowed ^
  --noconfirm ^
  --exclude-module tkinter ^
  app\main.py

echo Pronto: dist\EstatLab\EstatLab.exe
