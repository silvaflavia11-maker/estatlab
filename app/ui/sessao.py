"""Janela de sessão: acumula a saída das análises, exportável HTML/PDF."""
from __future__ import annotations

from PySide6.QtGui import QTextDocument
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from app.reports.formatacao import CSS_SESSAO

_HTML_INICIAL = (
    "<p><i>Sessão iniciada. Os resultados das análises aparecerão aqui, "
    "com a interpretação de cada teste.</i></p><hr>"
)


class SessaoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.navegador = QTextBrowser()
        self.navegador.document().setDefaultStyleSheet(CSS_SESSAO)
        self._html = _HTML_INICIAL
        self.navegador.setHtml(self._html)

        botao_html = QPushButton("Exportar HTML…")
        botao_pdf = QPushButton("Exportar PDF…")
        botao_limpar = QPushButton("Limpar sessão")
        botao_html.clicked.connect(self._exportar_html)
        botao_pdf.clicked.connect(self._exportar_pdf)
        botao_limpar.clicked.connect(self._limpar)

        botoes = QHBoxLayout()
        for b in (botao_html, botao_pdf, botao_limpar):
            botoes.addWidget(b)
        botoes.addStretch()

        leiaute = QVBoxLayout(self)
        leiaute.addWidget(self.navegador)
        leiaute.addLayout(botoes)

    def acrescentar(self, html: str) -> None:
        self._html += html
        self.navegador.setHtml(self._html)
        barra = self.navegador.verticalScrollBar()
        barra.setValue(barra.maximum())

    def html_completo(self) -> str:
        return f"<html><head><meta charset='utf-8'><style>{CSS_SESSAO}</style></head><body>{self._html}</body></html>"

    def carregar_html(self, html: str) -> None:
        self._html = html or _HTML_INICIAL
        self.navegador.setHtml(self._html)

    def html_bruto(self) -> str:
        return self._html

    def _limpar(self) -> None:
        if QMessageBox.question(self, "Limpar sessão",
                                "Apagar todos os resultados da sessão?") == QMessageBox.Yes:
            self.carregar_html(_HTML_INICIAL)

    def _exportar_html(self) -> None:
        caminho, _ = QFileDialog.getSaveFileName(self, "Exportar sessão",
                                                 "sessao.html", "HTML (*.html)")
        if caminho:
            with open(caminho, "w", encoding="utf-8") as arquivo:
                arquivo.write(self.html_completo())

    def _exportar_pdf(self) -> None:
        caminho, _ = QFileDialog.getSaveFileName(self, "Exportar sessão",
                                                 "sessao.pdf", "PDF (*.pdf)")
        if not caminho:
            return
        from PySide6.QtGui import QPageLayout, QPageSize
        from PySide6.QtPrintSupport import QPrinter

        impressora = QPrinter(QPrinter.HighResolution)
        impressora.setOutputFormat(QPrinter.PdfFormat)
        impressora.setOutputFileName(caminho)
        impressora.setPageSize(QPageSize(QPageSize.A4))
        documento = QTextDocument()
        documento.setDefaultStyleSheet(CSS_SESSAO)
        documento.setHtml(self._html)
        documento.print_(impressora)
