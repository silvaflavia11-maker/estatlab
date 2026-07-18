"""Testes de normalidade."""
from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors, normal_ad

from .resultados import ResultadoTeste
from .util import limpar_numerica

MET_AD = "anderson-darling"
MET_SW = "shapiro-wilk"
MET_KS = "kolmogorov-smirnov"
METODOS = {
    MET_AD: "Anderson-Darling",
    MET_SW: "Shapiro-Wilk",
    MET_KS: "Kolmogorov-Smirnov (Lilliefors)",
}


def teste_normalidade(dados, coluna: str, metodo: str = MET_AD,
                      alfa: float = 0.05) -> ResultadoTeste:
    x = limpar_numerica(dados, coluna, n_minimo=4)
    avisos: list[str] = []

    if metodo == MET_AD:
        estat, p = normal_ad(x)
        nome = "A²"
    elif metodo == MET_SW:
        if x.size > 5000:
            avisos.append("Shapiro-Wilk com n > 5000: o p-valor pode ser impreciso.")
        estat, p = stats.shapiro(x)
        nome = "W"
    elif metodo == MET_KS:
        estat, p = lilliefors(x, dist="norm")
        nome = "D"
    else:
        from .resultados import ErroAnalise

        raise ErroAnalise(f"Método de normalidade inválido: {metodo!r}.")

    if x.size < 20:
        avisos.append(
            "Amostra pequena (n < 20): testes de normalidade têm baixo poder — "
            "a não rejeição de H₀ não garante normalidade. Observe também o "
            "gráfico de probabilidade normal."
        )
    return ResultadoTeste(
        titulo=f"Teste de normalidade ({METODOS[metodo]}): {coluna}",
        h0=f"H₀: os dados de '{coluna}' seguem uma distribuição normal",
        h1=f"H₁: os dados de '{coluna}' não seguem uma distribuição normal",
        nome_estatistica=nome,
        estatistica=float(estat),
        p_valor=float(p),
        alfa=alfa,
        conclusao_h1=f"os dados de '{coluna}' não seguem uma distribuição normal",
        amostras=[{"amostra": coluna, "n": int(x.size),
                   "média": float(np.mean(x)),
                   "desvio-padrão": float(np.std(x, ddof=1))}],
        avisos=avisos,
    )
