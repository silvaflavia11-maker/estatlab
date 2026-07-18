"""Ponto de entrada do EstatLab: ``python -m app.main``."""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from app.ui.janela_principal import JanelaPrincipal


def principal() -> int:
    aplicacao = QApplication(sys.argv)
    aplicacao.setApplicationName("EstatLab")
    aplicacao.setOrganizationName("EstatLab")
    janela = JanelaPrincipal()
    janela.show()
    return aplicacao.exec()


if __name__ == "__main__":
    sys.exit(principal())
