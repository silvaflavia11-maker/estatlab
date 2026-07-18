"""Testes de outliers: Grubbs e Dixon (Q).

Estes métodos não existem em scipy/statsmodels, então são implementados
com fonte bibliográfica e testes de referência:

- Grubbs, F. E. (1969). "Procedures for detecting outlying observations in
  samples", Technometrics 11(1). Valor crítico via distribuição t; p-valor
  aproximado pela mesma relação.
- Dixon: valores críticos do teste Q (razão r10) para 3 ≤ n ≤ 10, conforme
  Rorabacher, D. B. (1991), Analytical Chemistry 63(2).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .resultados import ErroAnalise, ResultadoTeste
from .util import limpar_numerica

# Valores críticos do Q de Dixon (razão r10, teste bilateral), n = 3..10.
_Q_CRITICO = {
    0.05: {3: 0.970, 4: 0.829, 5: 0.710, 6: 0.625, 7: 0.568, 8: 0.526, 9: 0.493, 10: 0.466},
    0.01: {3: 0.994, 4: 0.926, 5: 0.821, 6: 0.740, 7: 0.680, 8: 0.634, 9: 0.598, 10: 0.568},
}


def teste_grubbs(dados, coluna: str, alfa: float = 0.05) -> ResultadoTeste:
    """Teste de Grubbs bilateral para um único outlier."""
    x = limpar_numerica(dados, coluna, n_minimo=3)
    n = x.size
    media = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    if s == 0:
        raise ErroAnalise(f"Todos os valores de '{coluna}' são iguais; não há outliers a testar.")

    idx = int(np.argmax(np.abs(x - media)))
    suspeito = float(x[idx])
    g = abs(suspeito - media) / s

    # Valor crítico (Grubbs 1969): G_crit = ((n-1)/√n)·√(t²/(n-2+t²)),
    # com t = t_{α/(2n); n-2}.
    t_crit = stats.t.ppf(1 - alfa / (2 * n), n - 2)
    g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))

    # p-valor aproximado invertendo a mesma relação.
    denom = ((n - 1) ** 2 / n) - g**2
    if denom <= 0:
        p = 0.0
    else:
        t_g = np.sqrt(g**2 * (n - 2) / denom)
        p = float(min(1.0, 2 * n * stats.t.sf(t_g, n - 2)))

    return ResultadoTeste(
        titulo=f"Teste de outlier de Grubbs: {coluna}",
        h0=f"H₀: não há outliers em '{coluna}' (todos os valores vêm da mesma população normal)",
        h1=f"H₁: o valor mais extremo de '{coluna}' é um outlier",
        nome_estatistica="G",
        estatistica=float(g),
        p_valor=p,
        alfa=alfa,
        conclusao_h1=f"o valor {suspeito:g} (linha {idx + 1} dos dados válidos) é um outlier",
        amostras=[{"amostra": coluna, "n": n, "média": media, "desvio-padrão": s}],
        detalhes={"valor suspeito": suspeito,
                  "G crítico": float(g_crit),
                  "decisão pelo valor crítico": "G > G crítico (outlier)" if g > g_crit
                  else "G ≤ G crítico (não é outlier)"},
        avisos=["O teste de Grubbs pressupõe que os demais dados seguem distribuição "
                "normal e detecta apenas um outlier por vez."],
    )


def teste_dixon(dados, coluna: str, alfa: float = 0.05) -> ResultadoTeste:
    """Teste Q de Dixon (razão r10) para 3 ≤ n ≤ 10."""
    if alfa not in _Q_CRITICO:
        raise ErroAnalise("Para o teste de Dixon use α = 0,05 ou α = 0,01.")
    x = np.sort(limpar_numerica(dados, coluna, n_minimo=3))
    n = x.size
    if n > 10:
        raise ErroAnalise(
            f"O teste Q de Dixon implementado vale para 3 ≤ n ≤ 10 (a coluna tem n = {n}). "
            "Para amostras maiores, use o teste de Grubbs."
        )
    amplitude = x[-1] - x[0]
    if amplitude == 0:
        raise ErroAnalise(f"Todos os valores de '{coluna}' são iguais; não há outliers a testar.")

    q_inferior = (x[1] - x[0]) / amplitude
    q_superior = (x[-1] - x[-2]) / amplitude
    if q_superior >= q_inferior:
        q, suspeito, lado = float(q_superior), float(x[-1]), "máximo"
    else:
        q, suspeito, lado = float(q_inferior), float(x[0]), "mínimo"
    q_crit = _Q_CRITICO[alfa][n]
    rejeita = q > q_crit

    return ResultadoTeste(
        titulo=f"Teste Q de Dixon: {coluna}",
        h0=f"H₀: não há outliers em '{coluna}'",
        h1=f"H₁: o valor {lado} de '{coluna}' é um outlier",
        nome_estatistica="Q",
        estatistica=q,
        # O teste clássico de Dixon decide por valor crítico tabelado, sem
        # p-valor exato; usamos o limiar como p artificial para reaproveitar
        # a lógica de decisão (p < α ⇔ Q > Q crítico).
        p_valor=0.0 if rejeita else 1.0,
        alfa=alfa,
        conclusao_h1=f"o valor {suspeito:g} ({lado}) é um outlier",
        amostras=[{"amostra": coluna, "n": n,
                   "média": float(np.mean(x)), "desvio-padrão": float(np.std(x, ddof=1))}],
        detalhes={"valor suspeito": suspeito,
                  "Q crítico (tabelado)": q_crit,
                  "decisão pelo valor crítico": "Q > Q crítico (outlier)" if rejeita
                  else "Q ≤ Q crítico (não é outlier)",
                  "observação": "decisão por valor crítico tabelado; o teste de Dixon "
                                "não produz p-valor exato"},
        avisos=["O teste Q de Dixon pressupõe normalidade dos demais dados e é "
                "recomendado apenas para amostras pequenas (3 a 10 observações)."],
    )
