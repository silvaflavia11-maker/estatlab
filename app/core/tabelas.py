"""Tabelas de contingência: tabulação cruzada, qui-quadrado, Fisher exato
e qui-quadrado de aderência."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .resultados import ErroAnalise, ResultadoComposto


def _categorias(df: pd.DataFrame, coluna: str) -> pd.Series:
    serie = df[coluna].astype(object)
    vazio = serie.isna() | (serie.astype(str).str.strip() == "")
    serie = serie.where(~vazio, np.nan).dropna().astype(str)
    if serie.empty:
        raise ErroAnalise(f"A coluna '{coluna}' está vazia.")
    return serie


def tabulacao_cruzada(df: pd.DataFrame, col_linhas: str, col_colunas: str,
                      alfa: float = 0.05) -> ResultadoComposto:
    """Tabela cruzada com teste qui-quadrado de independência e V de Cramér."""
    from app.reports.formatacao import fmt, fmt_p

    pares = pd.DataFrame({"l": _categorias(df, col_linhas),
                          "c": _categorias(df, col_colunas)}).dropna()
    if len(pares) < 2:
        raise ErroAnalise("Menos de 2 observações completas nas duas colunas.")
    tabela = pd.crosstab(pares["l"], pares["c"])
    if tabela.shape[0] < 2 or tabela.shape[1] < 2:
        raise ErroAnalise("As duas colunas precisam de pelo menos 2 categorias cada.")

    qui2, p, gl, esperados = stats.chi2_contingency(tabela)
    n = int(tabela.to_numpy().sum())
    k = min(tabela.shape) - 1
    cramer = float(np.sqrt(qui2 / (n * k))) if k > 0 else float("nan")

    linhas_tab = []
    for nome_linha in tabela.index:
        linhas_tab.append([nome_linha] + [int(v) for v in tabela.loc[nome_linha]]
                          + [int(tabela.loc[nome_linha].sum())])
    linhas_tab.append(["total"] + [int(v) for v in tabela.sum()] + [n])

    itens: list[tuple] = [
        ("subtitulo", f"Contagens observadas ({col_linhas} × {col_colunas})"),
        ("tabela", [col_linhas + " \\ " + col_colunas] + list(tabela.columns)
         + ["total"], linhas_tab),
        ("subtitulo", "Teste qui-quadrado de independência"),
        ("nota", f"H₀: '{col_linhas}' e '{col_colunas}' são independentes.  "
                 "H₁: existe associação entre elas."),
        ("tabela", ["estatística", "valor"],
         [["χ²", fmt(float(qui2))], ["graus de liberdade", int(gl)],
          ["p-valor", fmt_p(float(p))], ["n", n],
          ["V de Cramér (força da associação)", fmt(cramer, 3)]]),
    ]
    if p < alfa:
        itens.append(("interpretacao",
                      f"Como p = {fmt_p(float(p))} < α = {fmt(alfa, 2)}, rejeita-se "
                      f"H₀: há evidência de associação entre '{col_linhas}' e "
                      f"'{col_colunas}'. V de Cramér = {fmt(cramer, 2)} "
                      "(0 = nenhuma; 1 = perfeita)."))
    else:
        itens.append(("interpretacao",
                      f"Como p = {fmt_p(float(p))} ≥ α = {fmt(alfa, 2)}, não se "
                      f"rejeita H₀: não há evidência de associação entre "
                      f"'{col_linhas}' e '{col_colunas}'."))

    frac_baixa = float((esperados < 5).mean())
    if frac_baixa > 0.2:
        itens.append(("aviso", f"{fmt(100 * frac_baixa, 0)}% das caselas têm "
                               "frequência esperada < 5 — o qui-quadrado pode ser "
                               "impreciso. Para tabelas 2×2, use o teste exato de "
                               "Fisher."))
    return ResultadoComposto(
        titulo=f"Tabulação cruzada e qui-quadrado: {col_linhas} × {col_colunas}",
        itens=itens,
    )


def fisher_exato(df: pd.DataFrame, col_linhas: str, col_colunas: str,
                 alfa: float = 0.05) -> ResultadoComposto:
    from app.reports.formatacao import fmt, fmt_p

    pares = pd.DataFrame({"l": _categorias(df, col_linhas),
                          "c": _categorias(df, col_colunas)}).dropna()
    tabela = pd.crosstab(pares["l"], pares["c"])
    if tabela.shape != (2, 2):
        raise ErroAnalise("O teste exato de Fisher exige uma tabela 2×2 "
                          f"(obtida: {tabela.shape[0]}×{tabela.shape[1]}). "
                          "Use o qui-quadrado para tabelas maiores.")
    razao, p = stats.fisher_exact(tabela)

    linhas_tab = [[nome] + [int(v) for v in tabela.loc[nome]] for nome in tabela.index]
    decisao = (f"Como p = {fmt_p(float(p))} {'<' if p < alfa else '≥'} α = "
               f"{fmt(alfa, 2)}, " +
               ("rejeita-se H₀: há evidência de associação."
                if p < alfa else
                "não se rejeita H₀: não há evidência de associação."))
    return ResultadoComposto(
        titulo=f"Teste exato de Fisher: {col_linhas} × {col_colunas}",
        itens=[
            ("tabela", [col_linhas + " \\ " + col_colunas] + list(tabela.columns),
             linhas_tab),
            ("nota", "H₀: as duas variáveis são independentes."),
            ("tabela", ["estatística", "valor"],
             [["razão de chances", fmt(float(razao), 3)],
              ["p-valor (exato, bilateral)", fmt_p(float(p))]]),
            ("interpretacao", decisao + " O teste exato é válido mesmo com "
                              "contagens pequenas."),
        ],
    )


def aderencia(df: pd.DataFrame, coluna: str,
              proporcoes: dict[str, float] | None = None,
              alfa: float = 0.05) -> ResultadoComposto:
    """Qui-quadrado de aderência; ``proporcoes`` None = proporções iguais."""
    from app.reports.formatacao import fmt, fmt_p

    contagens = _categorias(df, coluna).value_counts().sort_index()
    n = int(contagens.sum())
    if len(contagens) < 2:
        raise ErroAnalise(f"A coluna '{coluna}' precisa de pelo menos 2 categorias.")

    if proporcoes:
        faltantes = set(contagens.index) - set(proporcoes)
        if faltantes:
            raise ErroAnalise("Informe a proporção esperada de todas as categorias "
                              f"(faltam: {', '.join(sorted(faltantes))}).")
        soma = sum(proporcoes[c] for c in contagens.index)
        if abs(soma - 1) > 1e-6:
            raise ErroAnalise(f"As proporções esperadas devem somar 1 (soma: {soma:g}).")
        esperados = np.array([n * proporcoes[c] for c in contagens.index])
        hipotese = "as proporções especificadas"
    else:
        esperados = np.full(len(contagens), n / len(contagens))
        hipotese = "proporções iguais em todas as categorias"

    qui2, p = stats.chisquare(contagens.to_numpy(), esperados)
    gl = len(contagens) - 1

    linhas = [[categoria, int(obs), fmt(float(esp)), fmt(float((obs - esp) ** 2 / esp))]
              for categoria, obs, esp in zip(contagens.index, contagens, esperados)]
    decisao = (f"Como p = {fmt_p(float(p))} {'<' if p < alfa else '≥'} α = "
               f"{fmt(alfa, 2)}, " +
               (f"rejeita-se H₀: as frequências de '{coluna}' não seguem {hipotese}."
                if p < alfa else
                f"não se rejeita H₀: as frequências de '{coluna}' são compatíveis "
                f"com {hipotese}."))
    itens: list[tuple] = [
        ("nota", f"H₀: as categorias de '{coluna}' seguem {hipotese}."),
        ("tabela", ["categoria", "observado", "esperado", "contribuição ao χ²"],
         linhas),
        ("tabela", ["estatística", "valor"],
         [["χ²", fmt(float(qui2))], ["graus de liberdade", gl],
          ["p-valor", fmt_p(float(p))], ["n", n]]),
        ("interpretacao", decisao),
    ]
    if (esperados < 5).any():
        itens.append(("aviso", "Há categorias com frequência esperada < 5; o "
                               "qui-quadrado pode ser impreciso."))
    return ResultadoComposto(titulo=f"Qui-quadrado de aderência: {coluna}",
                             itens=itens)
