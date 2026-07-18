"""Correlação (Pearson e Spearman) e covariância."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .resultados import ErroAnalise, ResultadoTabela
from .util import ALT_SCIPY


def _df_numerico(dados: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    if len(colunas) < 2:
        raise ErroAnalise("Selecione pelo menos 2 colunas numéricas.")
    df = dados[colunas].apply(pd.to_numeric, errors="coerce")
    return df


def _classificar_r(r: float) -> str:
    forca = abs(r)
    if forca < 0.3:
        rotulo = "fraca"
    elif forca < 0.7:
        rotulo = "moderada"
    else:
        rotulo = "forte"
    sentido = "positiva" if r > 0 else "negativa"
    return f"correlação {sentido} {rotulo}"


def correlacao(dados: pd.DataFrame, colunas: list[str], metodo: str = "pearson",
               alfa: float = 0.05) -> ResultadoTabela:
    """Correlação par a par com p-valores (H₀: ρ = 0 para cada par)."""
    if metodo not in ("pearson", "spearman"):
        raise ErroAnalise("Método de correlação inválido: use 'pearson' ou 'spearman'.")
    df = _df_numerico(dados, colunas)

    from app.reports.formatacao import fmt, fmt_p  # formatação PT-BR

    linhas: list[list[str]] = []
    notas: list[str] = [
        "Para cada par testa-se H₀: ρ = 0 (não há correlação) contra H₁: ρ ≠ 0.",
    ]
    func = stats.pearsonr if metodo == "pearson" else stats.spearmanr
    for i in range(len(colunas)):
        for j in range(i + 1, len(colunas)):
            par = df[[colunas[i], colunas[j]]].dropna()
            if len(par) < 3:
                raise ErroAnalise(
                    f"O par '{colunas[i]}' × '{colunas[j]}' tem menos de 3 observações completas."
                )
            res = func(par[colunas[i]], par[colunas[j]],
                       alternative=ALT_SCIPY["bilateral"])
            r, p = float(res.statistic), float(res.pvalue)
            linhas.append([f"{colunas[i]} × {colunas[j]}", str(len(par)), fmt(r), fmt_p(p)])
            decisao = ("significativa" if p < alfa else "não significativa")
            notas.append(
                f"{colunas[i]} × {colunas[j]}: r = {fmt(r, 3)} ({_classificar_r(r)}), "
                f"p = {fmt_p(p)} → {decisao} ao nível α = {fmt(alfa, 2)}."
            )
    nome = "Pearson" if metodo == "pearson" else "Spearman (postos)"
    return ResultadoTabela(
        titulo=f"Correlação de {nome}",
        cabecalhos=["par de colunas", "n", "coeficiente r", "p-valor"],
        linhas=linhas,
        notas=notas,
    )


def covariancia(dados: pd.DataFrame, colunas: list[str]) -> ResultadoTabela:
    df = _df_numerico(dados, colunas).dropna()
    if len(df) < 2:
        raise ErroAnalise("São necessárias pelo menos 2 observações completas.")
    matriz = df.cov()

    from app.reports.formatacao import fmt

    linhas = [[c] + [fmt(matriz.loc[c, c2]) for c2 in colunas] for c in colunas]
    return ResultadoTabela(
        titulo="Matriz de covariâncias",
        cabecalhos=[""] + colunas,
        linhas=linhas,
        notas=[f"n = {len(df)} observações completas (linhas com valores ausentes foram excluídas).",
               "A diagonal contém as variâncias amostrais de cada coluna."],
    )
