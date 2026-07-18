"""Ferramentas da qualidade: run chart, Pareto, Multi-Vari e intervalos de
tolerância.

Intervalo de tolerância normal bilateral: aproximação de Howe (1969),
k = √[ (n−1)(1+1/n) z²_{(1+P)/2} / χ²_{α;n−1} ]. Unilateral: fator exato via
t não central. Não paramétrico (mín/máx): confiança atingida
1 − n·P^{n−1} + (n−1)·P^n (regra clássica: n = 93 para 95/95 bilateral).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .resultados import ErroAnalise, ResultadoComposto
from .util import limpar_numerica


def pareto_contagens(dados, coluna: str) -> pd.Series:
    serie = pd.Series(dados).dropna()
    serie = serie[serie.astype(str).str.strip() != ""]
    if serie.empty:
        raise ErroAnalise(f"A coluna '{coluna}' está vazia.")
    return serie.astype(str).value_counts()


def pareto_resumo(dados, coluna: str) -> ResultadoComposto:
    from app.reports.formatacao import fmt

    contagens = pareto_contagens(dados, coluna)
    total = int(contagens.sum())
    acumulado = 0
    linhas = []
    poucos_vitais = []
    for categoria, contagem in contagens.items():
        acumulado += int(contagem)
        pct_acum = 100 * acumulado / total
        linhas.append([categoria, int(contagem),
                       fmt(100 * contagem / total, 1) + "%",
                       fmt(pct_acum, 1) + "%"])
        if pct_acum <= 80 or not poucos_vitais:
            poucos_vitais.append(str(categoria))
    return ResultadoComposto(
        titulo=f"Análise de Pareto: {coluna}",
        itens=[
            ("tabela", ["categoria", "contagem", "%", "% acumulado"], linhas),
            ("interpretacao",
             "Princípio de Pareto: poucas categorias costumam concentrar a "
             "maior parte das ocorrências. Aqui, ~80% concentram-se em: "
             + ", ".join(poucos_vitais) + ". Priorize-as."),
        ],
    )


def multivari_dados(df: pd.DataFrame, resposta: str, fatores: list[str]):
    """Prepara médias para a carta Multi-Vari (2 ou 3 fatores)."""
    from .anova import _empilhar

    if not 2 <= len(fatores) <= 3:
        raise ErroAnalise("A carta Multi-Vari usa 2 ou 3 fatores.")
    dados = _empilhar(df, resposta, fatores)
    return dados  # colunas: y, f0, f1[, f2]


def intervalo_tolerancia(dados, coluna: str, cobertura: float = 0.95,
                         confianca: float = 0.95,
                         lado: str = "bilateral") -> ResultadoComposto:
    from app.reports.formatacao import fmt

    if not 0 < cobertura < 1 or not 0 < confianca < 1:
        raise ErroAnalise("Cobertura e confiança devem estar entre 0 e 1.")
    x = limpar_numerica(dados, coluna, n_minimo=3)
    n = x.size
    media = float(np.mean(x))
    s = float(np.std(x, ddof=1))

    if lado == "bilateral":
        z = stats.norm.ppf((1 + cobertura) / 2)
        qui2 = stats.chi2.ppf(1 - confianca, n - 1)
        k = float(np.sqrt((n - 1) * (1 + 1 / n) * z**2 / qui2))
        limites_normal = (media - k * s, media + k * s)
        descricao = (f"({fmt(limites_normal[0])}; {fmt(limites_normal[1])})")
    elif lado in ("superior", "inferior"):
        delta = stats.norm.ppf(cobertura) * np.sqrt(n)
        k = float(stats.nct.ppf(confianca, n - 1, delta) / np.sqrt(n))
        if lado == "superior":
            limites_normal = (-np.inf, media + k * s)
            descricao = f"(−∞; {fmt(limites_normal[1])})"
        else:
            limites_normal = (media - k * s, np.inf)
            descricao = f"({fmt(limites_normal[0])}; +∞)"
    else:
        raise ErroAnalise("Lado inválido: use 'bilateral', 'superior' ou 'inferior'.")

    # Não paramétrico (mín/máx): confiança atingida para a cobertura pedida
    if lado == "bilateral":
        conf_np = 1 - n * cobertura ** (n - 1) + (n - 1) * cobertura**n
        limites_np = f"({fmt(float(np.min(x)))}; {fmt(float(np.max(x)))})"
    else:
        conf_np = 1 - cobertura**n
        limites_np = (f"(−∞; {fmt(float(np.max(x)))})" if lado == "superior"
                      else f"({fmt(float(np.min(x)))}; +∞)")

    from statsmodels.stats.diagnostic import normal_ad

    _, p_normal = normal_ad(x)
    itens: list[tuple] = [
        ("tabela", ["item", "valor"],
         [["n", str(n)], ["média", fmt(media)], ["desvio-padrão", fmt(s)],
          ["cobertura desejada", fmt(100 * cobertura, 1) + "%"],
          ["confiança", fmt(100 * confianca, 1) + "%"]]),
        ("subtitulo", "Método normal"),
        ("tabela", ["item", "valor"],
         [["fator k", fmt(k, 4)], ["intervalo de tolerância", descricao]]),
        ("subtitulo", "Método não paramétrico (mínimo/máximo da amostra)"),
        ("tabela", ["item", "valor"],
         [["intervalo", limites_np],
          ["confiança atingida", fmt(100 * float(conf_np), 1) + "%"]]),
        ("interpretacao",
         f"Com {fmt(100 * confianca, 0)}% de confiança, pelo menos "
         f"{fmt(100 * cobertura, 0)}% da população de '{coluna}' está dentro do "
         "intervalo de tolerância (método normal). O método não paramétrico não "
         "supõe normalidade, mas exige n grande para alta confiança."),
    ]
    if p_normal < 0.05:
        itens.append(("aviso", "Os dados rejeitam normalidade (AD, p < 0,05): "
                               "prefira o intervalo não paramétrico."))
    return ResultadoComposto(titulo=f"Intervalo de tolerância: {coluna}",
                             itens=itens)
