"""Planilha de dados e modelo Qt correspondente.

A planilha guarda os valores em um DataFrame de dtype ``object`` (como uma
grade livre); a tipagem efetiva é inferida por coluna. As análises recebem
as colunas via ``valores()`` e fazem a coerção numérica (NaN para células
vazias/não numéricas), então células fora do padrão nunca quebram cálculo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

LINHAS_INICIAIS = 200
COLUNAS_INICIAIS = 12


def _parse_celula(texto: str):
    """Converte entrada do usuário: número PT-BR/EN, vazio → NaN, senão texto."""
    texto = texto.strip()
    if texto == "":
        return np.nan
    normalizado = texto
    if "," in normalizado:
        normalizado = normalizado.replace(".", "").replace(",", ".")
    try:
        return float(normalizado)
    except ValueError:
        return texto


def _exibir_celula(valor) -> str:
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return ""
    if isinstance(valor, float):
        if valor.is_integer() and abs(valor) < 1e15:
            return str(int(valor))
        return f"{valor:g}".replace(".", ",")
    return str(valor)


class Planilha:
    def __init__(self, nome: str = "Planilha 1", df: pd.DataFrame | None = None):
        self.nome = nome
        if df is None:
            df = pd.DataFrame(
                np.full((LINHAS_INICIAIS, COLUNAS_INICIAIS), np.nan, dtype=object),
                columns=[f"C{i + 1}" for i in range(COLUNAS_INICIAIS)],
            )
        self.df = df.astype(object)

    # ------------------------------------------------------------ leitura
    def valores(self, coluna: str) -> pd.Series:
        return self.df[coluna]

    def colunas(self) -> list[str]:
        return list(self.df.columns)

    def coluna_e_numerica(self, coluna: str) -> bool:
        serie = self.df[coluna].dropna()
        serie = serie[serie.astype(str).str.strip() != ""]
        if serie.empty:
            return False
        return pd.to_numeric(serie, errors="coerce").notna().all()

    def colunas_numericas(self) -> list[str]:
        return [c for c in self.df.columns if self.coluna_e_numerica(c)]

    def tipo_coluna(self, coluna: str) -> str:
        serie = self.df[coluna].dropna()
        if serie.empty:
            return "vazia"
        return "numérica" if self.coluna_e_numerica(coluna) else "texto"

    # ------------------------------------------------------------ arquivo
    @staticmethod
    def de_arquivo(caminho: str, nome: str | None = None) -> "Planilha":
        if caminho.lower().endswith((".xlsx", ".xlsm")):
            df = pd.read_excel(caminho)
        else:
            # CSV: tenta detectar separador (vírgula, ponto-e-vírgula, tab)
            df = pd.read_csv(caminho, sep=None, engine="python", decimal=",")
            if df.select_dtypes("number").empty:
                df2 = pd.read_csv(caminho, sep=None, engine="python")
                if not df2.select_dtypes("number").empty:
                    df = df2
        import os

        return Planilha(nome or os.path.splitext(os.path.basename(caminho))[0], df)

    def salvar(self, caminho: str) -> None:
        df = self._df_compacto()
        if caminho.lower().endswith(".xlsx"):
            df.to_excel(caminho, index=False)
        else:
            df.to_csv(caminho, index=False, sep=";", decimal=",")

    def _df_compacto(self) -> pd.DataFrame:
        """Remove linhas/colunas totalmente vazias do final da grade."""
        df = self.df.copy()
        vazio = df.isna() | (df.astype(str).apply(lambda s: s.str.strip()) == "")
        linhas = ~vazio.all(axis=1)
        colunas = ~vazio.all(axis=0)
        if linhas.any():
            df = df.loc[: linhas[linhas].index[-1], colunas]
        else:
            df = df.loc[[], colunas]
        return df


class QtPlanilhaModel(QAbstractTableModel):
    def __init__(self, planilha: Planilha, parent=None):
        super().__init__(parent)
        self.planilha = planilha

    # ------------------------------------------------------------ Qt API
    def rowCount(self, parent=QModelIndex()):
        return len(self.planilha.df)

    def columnCount(self, parent=QModelIndex()):
        return len(self.planilha.df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role in (Qt.DisplayRole, Qt.EditRole):
            valor = self.planilha.df.iat[index.row(), index.column()]
            return _exibir_celula(valor)
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if role != Qt.EditRole or not index.isValid():
            return False
        self.planilha.df.iat[index.row(), index.column()] = _parse_celula(str(value))
        self.dataChanged.emit(index, index, [Qt.DisplayRole])
        return True

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            nome = self.planilha.df.columns[section]
            tipo = self.planilha.tipo_coluna(nome)
            sufixo = {"numérica": " (num)", "texto": " (txt)", "vazia": ""}[tipo]
            return f"{nome}{sufixo}"
        return str(section + 1)

    def flags(self, index):
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable

    # ------------------------------------------------------------ edição estrutural
    def definir_valor_bruto(self, linha: int, coluna: int, valor) -> None:
        self.planilha.df.iat[linha, coluna] = valor

    def notificar_tudo(self):
        self.beginResetModel()
        self.endResetModel()

    def adicionar_linhas(self, quantidade: int = 50):
        df = self.planilha.df
        novas = pd.DataFrame(
            np.full((quantidade, len(df.columns)), np.nan, dtype=object),
            columns=df.columns,
        )
        self.beginResetModel()
        self.planilha.df = pd.concat([df, novas], ignore_index=True)
        self.endResetModel()

    def adicionar_coluna(self, nome: str):
        if nome in self.planilha.df.columns:
            raise ValueError(f"Já existe uma coluna chamada '{nome}'.")
        self.beginResetModel()
        self.planilha.df[nome] = np.nan
        self.planilha.df[nome] = self.planilha.df[nome].astype(object)
        self.endResetModel()

    def renomear_coluna(self, atual: str, novo: str):
        if novo in self.planilha.df.columns and novo != atual:
            raise ValueError(f"Já existe uma coluna chamada '{novo}'.")
        self.beginResetModel()
        self.planilha.df = self.planilha.df.rename(columns={atual: novo})
        self.endResetModel()

    def remover_coluna(self, nome: str):
        self.beginResetModel()
        self.planilha.df = self.planilha.df.drop(columns=[nome])
        self.endResetModel()
