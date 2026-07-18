#!/bin/zsh
# Gera o executável do EstatLab para macOS (rodar em um Mac).
set -e
cd "$(dirname "$0")/.."

./.venv/bin/pip install --quiet pyinstaller
./.venv/bin/pyinstaller \
  --name EstatLab \
  --windowed \
  --noconfirm \
  --exclude-module tkinter \
  app/main.py

echo "Pronto: dist/EstatLab.app"
