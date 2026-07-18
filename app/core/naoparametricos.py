"""Testes não paramétricos.

O teste de sequências (runs) é implementado pela aproximação normal de
Wald-Wolfowitz (fórmula clássica), validado contra recomputação independente.
Os demais vêm de scipy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .resultados import ErroAnalise, ResultadoTeste
from .util import ALT_SCIPY, SIMBOLO_ALT, limpar_numerica, validar_alternativa
from .anova import _empilhar, _grupos


def _frase_mediana(coluna: str, alternativa: str, ref) -> str:
    verbo = {"bilateral": "difere de", "menor": "é menor que",
             "maior": "é maior que"}[alternativa]
    return f"a mediana de '{coluna}' {verbo} {ref}"


def teste_sinal(dados, coluna: str, mediana0: float,
                alternativa: str = "bilateral", alfa: float = 0.05) -> ResultadoTeste:
    validar_alternativa(alternativa)
    x = limpar_numerica(dados, coluna)
    acima = int(np.sum(x > mediana0))
    abaixo = int(np.sum(x < mediana0))
    empates = int(np.sum(x == mediana0))
    n_efetivo = acima + abaixo
    if n_efetivo == 0:
        raise ErroAnalise("Todos os valores são iguais à mediana hipotética.")
    res = stats.binomtest(acima, n_efetivo, 0.5, alternative=ALT_SCIPY[alternativa])

    avisos = []
    if empates:
        avisos.append(f"{empates} valor(es) igual(is) à mediana hipotética foram "
                      "descartados (procedimento padrão do teste do sinal).")
    return ResultadoTeste(
        titulo=f"Teste do sinal: {coluna}",
        h0=f"H₀: mediana = {mediana0}",
        h1=f"H₁: mediana {SIMBOLO_ALT[alternativa]} {mediana0}",
        nome_estatistica="X (valores acima)",
        estatistica=float(acima),
        p_valor=float(res.pvalue),
        alfa=alfa,
        conclusao_h1=_frase_mediana(coluna, alternativa, mediana0),
        amostras=[{"amostra": coluna, "n": int(x.size), "acima": acima,
                   "abaixo": abaixo, "empates": empates,
                   "mediana amostral": float(np.median(x))}],
        detalhes={"método": "binomial exato sobre os sinais"},
        avisos=avisos,
    )


def teste_wilcoxon(dados, coluna: str, mediana0: float = 0.0,
                   dados2=None, coluna2: str | None = None,
                   alternativa: str = "bilateral", alfa: float = 0.05) -> ResultadoTeste:
    """Wilcoxon de postos sinalizados: 1 amostra (vs mediana) ou pareado."""
    validar_alternativa(alternativa)
    if dados2 is not None:
        par = pd.DataFrame({"a": pd.to_numeric(pd.Series(dados), errors="coerce"),
                            "b": pd.to_numeric(pd.Series(dados2), errors="coerce")}).dropna()
        if len(par) < 3:
            raise ErroAnalise("São necessários pelo menos 3 pares completos.")
        diferencas = par["a"].to_numpy() - par["b"].to_numpy()
        titulo = f"Teste de Wilcoxon (pareado): {coluna} − {coluna2}"
        h0 = f"H₀: a mediana das diferenças '{coluna}' − '{coluna2}' é 0"
        conclusao = f"a mediana das diferenças '{coluna}' − '{coluna2}' " + \
            {"bilateral": "difere de 0", "menor": "é menor que 0",
             "maior": "é maior que 0"}[alternativa]
        alvo = 0.0
    else:
        x = limpar_numerica(dados, coluna, n_minimo=3)
        diferencas = x - mediana0
        titulo = f"Teste de Wilcoxon (1 amostra): {coluna}"
        h0 = f"H₀: mediana = {mediana0}"
        conclusao = _frase_mediana(coluna, alternativa, mediana0)
        alvo = mediana0

    diferencas = diferencas[diferencas != 0]
    if diferencas.size < 3:
        raise ErroAnalise("Menos de 3 diferenças não nulas para o teste de Wilcoxon.")
    res = stats.wilcoxon(diferencas, alternative=ALT_SCIPY[alternativa])

    return ResultadoTeste(
        titulo=titulo,
        h0=h0,
        h1=f"H₁: mediana {SIMBOLO_ALT[alternativa]} {alvo}",
        nome_estatistica="W",
        estatistica=float(res.statistic),
        p_valor=float(res.pvalue),
        alfa=alfa,
        conclusao_h1=conclusao,
        amostras=[{"amostra": "diferenças não nulas", "n": int(diferencas.size),
                   "mediana": float(np.median(diferencas))}],
        avisos=["O teste de Wilcoxon pressupõe distribuição simétrica das "
                "diferenças em torno da mediana."],
    )


def teste_mann_whitney(dados1, dados2, col1: str, col2: str,
                       alternativa: str = "bilateral",
                       alfa: float = 0.05) -> ResultadoTeste:
    validar_alternativa(alternativa)
    x1 = limpar_numerica(dados1, col1)
    x2 = limpar_numerica(dados2, col2)
    res = stats.mannwhitneyu(x1, x2, alternative=ALT_SCIPY[alternativa])

    return ResultadoTeste(
        titulo=f"Teste de Mann-Whitney: {col1} × {col2}",
        h0=f"H₀: as distribuições de '{col1}' e '{col2}' são iguais "
           "(medianas iguais)",
        h1=f"H₁: mediana de '{col1}' {SIMBOLO_ALT[alternativa]} mediana de '{col2}'",
        nome_estatistica="U",
        estatistica=float(res.statistic),
        p_valor=float(res.pvalue),
        alfa=alfa,
        conclusao_h1=f"a mediana de '{col1}' "
        + {"bilateral": "difere da", "menor": "é menor que a",
           "maior": "é maior que a"}[alternativa] + f" mediana de '{col2}'",
        amostras=[{"amostra": col1, "n": int(x1.size), "mediana": float(np.median(x1))},
                  {"amostra": col2, "n": int(x2.size), "mediana": float(np.median(x2))}],
        avisos=["Interpretação como diferença de medianas exige formas de "
                "distribuição semelhantes nos dois grupos."],
    )


def teste_kruskal(df: pd.DataFrame, resposta: str, fator: str,
                  alfa: float = 0.05) -> ResultadoTeste:
    dados = _empilhar(df, resposta, [fator])
    grupos = _grupos(dados)
    if len(grupos) < 2:
        raise ErroAnalise(f"O fator '{fator}' precisa de pelo menos 2 níveis.")
    estat, p = stats.kruskal(*grupos.values())

    return ResultadoTeste(
        titulo=f"Teste de Kruskal-Wallis: {resposta} × {fator}",
        h0=f"H₀: as medianas de '{resposta}' são iguais em todos os níveis de '{fator}'",
        h1="H₁: pelo menos um nível tem mediana diferente",
        nome_estatistica="H",
        estatistica=float(estat),
        p_valor=float(p),
        alfa=alfa,
        gl=float(len(grupos) - 1),
        conclusao_h1=f"pelo menos um nível de '{fator}' tem mediana de "
                     f"'{resposta}' diferente",
        amostras=[{"amostra": nivel, "n": len(x), "mediana": float(np.median(x))}
                  for nivel, x in grupos.items()],
    )


def teste_mood(df: pd.DataFrame, resposta: str, fator: str,
               alfa: float = 0.05) -> ResultadoTeste:
    dados = _empilhar(df, resposta, [fator])
    grupos = _grupos(dados)
    if len(grupos) < 2:
        raise ErroAnalise(f"O fator '{fator}' precisa de pelo menos 2 níveis.")
    estat, p, mediana_global, tabela = stats.median_test(*grupos.values())

    return ResultadoTeste(
        titulo=f"Teste da mediana de Mood: {resposta} × {fator}",
        h0=f"H₀: as medianas de '{resposta}' são iguais em todos os níveis de '{fator}'",
        h1="H₁: pelo menos um nível tem mediana diferente",
        nome_estatistica="χ²",
        estatistica=float(estat),
        p_valor=float(p),
        alfa=alfa,
        gl=float(len(grupos) - 1),
        conclusao_h1=f"pelo menos um nível de '{fator}' tem mediana de "
                     f"'{resposta}' diferente",
        amostras=[{"amostra": nivel, "n": len(x), "mediana": float(np.median(x)),
                   "acima da mediana global": int(tabela[0][k]),
                   "abaixo/igual": int(tabela[1][k])}
                  for k, (nivel, x) in enumerate(grupos.items())],
        detalhes={"mediana global": float(mediana_global)},
        avisos=["O teste de Mood é mais robusto a outliers que Kruskal-Wallis, "
                "porém tem menor poder."],
    )


def teste_friedman(df: pd.DataFrame, colunas: list[str],
                   alfa: float = 0.05) -> ResultadoTeste:
    if len(colunas) < 3:
        raise ErroAnalise("O teste de Friedman exige pelo menos 3 colunas "
                          "(tratamentos medidos nos mesmos blocos/sujeitos).")
    matriz = df[colunas].apply(pd.to_numeric, errors="coerce").dropna()
    if len(matriz) < 3:
        raise ErroAnalise("Menos de 3 linhas (blocos) completas nas colunas "
                          "selecionadas.")
    estat, p = stats.friedmanchisquare(*[matriz[c] for c in colunas])

    return ResultadoTeste(
        titulo="Teste de Friedman: " + ", ".join(colunas),
        h0="H₀: os tratamentos (colunas) têm o mesmo efeito (mesma distribuição "
           "dentro de cada bloco)",
        h1="H₁: pelo menos um tratamento difere",
        nome_estatistica="χ²",
        estatistica=float(estat),
        p_valor=float(p),
        alfa=alfa,
        gl=float(len(colunas) - 1),
        conclusao_h1="pelo menos um dos tratamentos (colunas) difere dos demais",
        amostras=[{"amostra": c, "n": len(matriz),
                   "mediana": float(matriz[c].median())} for c in colunas],
        detalhes={"blocos (linhas completas)": len(matriz)},
    )


def teste_runs(dados, coluna: str, alfa: float = 0.05) -> ResultadoTeste:
    """Teste de sequências (Wald-Wolfowitz) de aleatoriedade, acima/abaixo
    da mediana, com aproximação normal."""
    x = limpar_numerica(dados, coluna, n_minimo=10)
    mediana = float(np.median(x))
    binario = x[x != mediana] > mediana  # descarta empates com a mediana
    n1 = int(binario.sum())
    n2 = int((~binario).sum())
    if n1 == 0 or n2 == 0:
        raise ErroAnalise("Não há valores dos dois lados da mediana; o teste de "
                          "sequências não se aplica.")
    runs = int(1 + np.sum(binario[1:] != binario[:-1]))
    n = n1 + n2
    esperado = 2 * n1 * n2 / n + 1
    variancia = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n**2 * (n - 1))
    z = (runs - esperado) / np.sqrt(variancia)
    p = float(2 * stats.norm.sf(abs(z)))

    avisos = []
    if n1 < 10 or n2 < 10:
        avisos.append("Com menos de 10 observações de cada lado da mediana, a "
                      "aproximação normal do teste de sequências é grosseira.")
    return ResultadoTeste(
        titulo=f"Teste de sequências (runs): {coluna}",
        h0=f"H₀: a ordem dos dados de '{coluna}' é aleatória",
        h1="H₁: a ordem dos dados não é aleatória (tendência ou agrupamento)",
        nome_estatistica="Z",
        estatistica=float(z),
        p_valor=p,
        alfa=alfa,
        conclusao_h1=f"a ordem dos dados de '{coluna}' não é aleatória",
        amostras=[{"amostra": coluna, "acima da mediana": n1,
                   "abaixo da mediana": n2, "sequências observadas": runs,
                   "sequências esperadas": float(esperado)}],
        detalhes={"mediana": mediana},
    )
