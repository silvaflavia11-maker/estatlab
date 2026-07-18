"""Visão da planilha com copiar/colar estilo Excel."""
from __future__ import annotations

from PySide6.QtGui import QGuiApplication, QKeySequence
from PySide6.QtWidgets import QTableView


class TabelaView(QTableView):
    def keyPressEvent(self, evento):
        if evento.matches(QKeySequence.Copy):
            self._copiar()
        elif evento.matches(QKeySequence.Paste):
            self._colar()
        elif evento.matches(QKeySequence.Delete) or evento.key() == 0x01000003:  # Backspace
            self._apagar_selecao()
        else:
            super().keyPressEvent(evento)

    def _copiar(self):
        selecao = self.selectedIndexes()
        if not selecao:
            return
        selecao.sort(key=lambda i: (i.row(), i.column()))
        linhas: dict[int, list[str]] = {}
        for indice in selecao:
            linhas.setdefault(indice.row(), []).append(indice.data() or "")
        texto = "\n".join("\t".join(valores) for valores in linhas.values())
        QGuiApplication.clipboard().setText(texto)

    def _colar(self):
        texto = QGuiApplication.clipboard().text()
        if not texto:
            return
        atual = self.currentIndex()
        if not atual.isValid():
            return
        modelo = self.model()
        linhas = texto.rstrip("\n").split("\n")
        precisa_linhas = atual.row() + len(linhas) - modelo.rowCount()
        if precisa_linhas > 0:
            modelo.adicionar_linhas(precisa_linhas + 10)
        for dl, linha in enumerate(linhas):
            for dc, valor in enumerate(linha.split("\t")):
                lin, col = atual.row() + dl, atual.column() + dc
                if col < modelo.columnCount():
                    modelo.setData(modelo.index(lin, col), valor)

    def _apagar_selecao(self):
        modelo = self.model()
        for indice in self.selectedIndexes():
            modelo.setData(indice, "")
