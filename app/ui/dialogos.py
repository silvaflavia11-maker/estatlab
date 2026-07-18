"""Diálogo genérico de configuração de análises."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QSpinBox,
)

ALTERNATIVAS = [
    ("bilateral (≠)", "bilateral"),
    ("unilateral à esquerda (<)", "menor"),
    ("unilateral à direita (>)", "maior"),
]


class DialogoAnalise(QDialog):
    """Formulário montado por composição; ``executar()`` devolve os valores."""

    def __init__(self, titulo: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(titulo)
        self._form = QFormLayout(self)
        self._campos: dict[str, callable] = {}

        self._botoes = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._botoes.accepted.connect(self.accept)
        self._botoes.rejected.connect(self.reject)

    def combo_coluna(self, chave: str, rotulo: str, colunas: list[str]):
        combo = QComboBox()
        combo.addItems(colunas)
        self._form.addRow(rotulo, combo)
        self._campos[chave] = combo.currentText
        return self

    def lista_colunas(self, chave: str, rotulo: str, colunas: list[str],
                      minimo: int = 2):
        lista = QListWidget()
        lista.addItems(colunas)
        lista.setSelectionMode(QListWidget.MultiSelection)
        self._form.addRow(QLabel(rotulo))
        self._form.addRow(lista)
        self._campos[chave] = lambda: [item.text() for item in lista.selectedItems()]
        return self

    def numero(self, chave: str, rotulo: str, valor: float = 0.0,
               minimo: float = -1e12, maximo: float = 1e12, decimais: int = 4):
        campo = QDoubleSpinBox()
        campo.setRange(minimo, maximo)
        campo.setDecimals(decimais)
        campo.setValue(valor)
        self._form.addRow(rotulo, campo)
        self._campos[chave] = campo.value
        return self

    def inteiro(self, chave: str, rotulo: str, valor: int = 0,
                minimo: int = 0, maximo: int = 10**9):
        campo = QSpinBox()
        campo.setRange(minimo, maximo)
        campo.setValue(valor)
        self._form.addRow(rotulo, campo)
        self._campos[chave] = campo.value
        return self

    def escolha(self, chave: str, rotulo: str, opcoes: list[tuple[str, object]]):
        combo = QComboBox()
        for texto, _ in opcoes:
            combo.addItem(texto)
        self._form.addRow(rotulo, combo)
        self._campos[chave] = lambda: opcoes[combo.currentIndex()][1]
        return self

    def texto(self, chave: str, rotulo: str, padrao: str = ""):
        campo = QLineEdit(padrao)
        self._form.addRow(rotulo, campo)
        self._campos[chave] = campo.text
        return self

    def caixa(self, chave: str, rotulo: str, marcado: bool = False):
        caixa = QCheckBox(rotulo)
        caixa.setChecked(marcado)
        self._form.addRow(caixa)
        self._campos[chave] = caixa.isChecked
        return self

    def alternativa_e_alfa(self):
        self.escolha("alternativa", "Hipótese alternativa:", ALTERNATIVAS)
        self.numero("alfa", "Nível de significância (α):", 0.05, 0.001, 0.5, 3)
        return self

    def executar(self) -> dict | None:
        self._form.addRow(self._botoes)
        if self.exec() != QDialog.Accepted:
            return None
        return {chave: obter() for chave, obter in self._campos.items()}
