"""Janela de gráfico com atualização automática e exportação."""
from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.resultados import ErroAnalise
from app.plots.graficos import FORMATOS_EXPORTACAO


class JanelaGrafico(QMainWindow):
    """Mostra uma Figure; ``gerador()`` recria o gráfico a partir da planilha
    atual, permitindo a atualização automática quando os dados mudam."""

    def __init__(self, titulo: str, gerador, modelo=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(titulo)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self._gerador = gerador
        self._modelo = modelo

        self._figura = gerador()
        self._canvas = FigureCanvasQTAgg(self._figura)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)

        self._auto = QCheckBox("Atualizar automaticamente quando os dados mudarem")
        self._auto.setChecked(True)
        botao_exportar = QPushButton("Exportar imagem…")
        botao_exportar.clicked.connect(self._exportar)

        centro = QWidget()
        leiaute = QVBoxLayout(centro)
        leiaute.addWidget(self._toolbar)
        leiaute.addWidget(self._canvas, stretch=1)
        leiaute.addWidget(self._auto)
        leiaute.addWidget(botao_exportar)
        self.setCentralWidget(centro)
        self.resize(760, 600)

        if modelo is not None:
            modelo.dataChanged.connect(self._dados_mudaram)
            modelo.modelReset.connect(self._dados_mudaram)

    def _dados_mudaram(self, *args) -> None:
        if self._auto.isChecked():
            self.atualizar()

    def atualizar(self) -> None:
        try:
            nova = self._gerador()
        except ErroAnalise:
            return  # dados momentaneamente inválidos durante a edição
        leiaute = self.centralWidget().layout()
        antiga = self._canvas
        self._figura = nova
        self._canvas = FigureCanvasQTAgg(nova)
        leiaute.replaceWidget(antiga, self._canvas)
        antiga.deleteLater()
        toolbar_antiga = self._toolbar
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        leiaute.replaceWidget(toolbar_antiga, self._toolbar)
        toolbar_antiga.deleteLater()

    def _exportar(self) -> None:
        caminho, _ = QFileDialog.getSaveFileName(
            self, "Exportar gráfico", "grafico.png", FORMATOS_EXPORTACAO
        )
        if not caminho:
            return
        try:
            self._figura.savefig(caminho, dpi=300, bbox_inches="tight")
        except Exception as erro:  # ex.: formato sem suporte do Pillow
            QMessageBox.warning(self, "Exportar gráfico",
                                f"Não foi possível exportar: {erro}")
