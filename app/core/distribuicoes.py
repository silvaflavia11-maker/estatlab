"""Calculadora de distribuições, geração de números aleatórios e amostragem."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .resultados import ErroAnalise, ResultadoTabela
from .util import limpar_numerica

# nome → (rótulos dos parâmetros, defaults, fábrica scipy, é discreta)
DISTRIBUICOES: dict[str, tuple] = {
    "Normal": (["média (μ)", "desvio-padrão (σ)"], [0.0, 1.0],
               lambda m, s: stats.norm(m, s), False),
    "t de Student": (["graus de liberdade"], [10.0],
                     lambda gl: stats.t(gl), False),
    "Qui-quadrado": (["graus de liberdade"], [5.0],
                     lambda gl: stats.chi2(gl), False),
    "F": (["GL do numerador", "GL do denominador"], [5.0, 10.0],
          lambda g1, g2: stats.f(g1, g2), False),
    "Exponencial": (["escala (1/λ)"], [1.0],
                    lambda s: stats.expon(scale=s), False),
    "Uniforme": (["mínimo (a)", "máximo (b)"], [0.0, 1.0],
                 lambda a, b: stats.uniform(a, b - a), False),
    "Weibull": (["forma (k)", "escala (λ)"], [1.5, 1.0],
                lambda k, s: stats.weibull_min(k, scale=s), False),
    "Binomial": (["n (ensaios)", "p (prob. de sucesso)"], [10.0, 0.5],
                 lambda n, p: stats.binom(int(n), p), True),
    "Poisson": (["λ (taxa média)"], [3.0],
                lambda taxa: stats.poisson(taxa), True),
}


def _validar_dist(nome: str, parametros: list[float]):
    if nome not in DISTRIBUICOES:
        raise ErroAnalise(f"Distribuição desconhecida: {nome}.")
    rotulos, _, fabrica, discreta = DISTRIBUICOES[nome]
    if len(parametros) != len(rotulos):
        raise ErroAnalise(f"A distribuição {nome} exige {len(rotulos)} parâmetro(s).")
    try:
        dist = fabrica(*parametros)
        dist.mean()  # força a validação dos parâmetros
    except Exception:
        raise ErroAnalise(f"Parâmetros inválidos para a distribuição {nome}.")
    if not np.isfinite(dist.mean()) and nome != "F":
        pass  # algumas combinações válidas têm média infinita; não bloquear
    return dist, discreta, rotulos


def calcular(nome: str, parametros: list[float], tipo: str,
             valor: float) -> ResultadoTabela:
    """``tipo``: 'densidade' (f(x) ou P(X=x)), 'acumulada' (P(X≤x)) ou
    'inversa' (quantil: menor x com P(X≤x) ≥ valor)."""
    from app.reports.formatacao import fmt

    dist, discreta, rotulos = _validar_dist(nome, parametros)
    parametros_txt = ", ".join(f"{r} = {fmt(v, 4)}" for r, v in zip(rotulos, parametros))

    if tipo == "densidade":
        resultado = dist.pmf(valor) if discreta else dist.pdf(valor)
        rotulo = f"P(X = {fmt(valor)})" if discreta else f"densidade f({fmt(valor)})"
        nota = ("Para distribuições discretas, é a probabilidade exata do valor."
                if discreta else
                "A densidade não é probabilidade; áreas sob a curva é que são.")
    elif tipo == "acumulada":
        resultado = dist.cdf(valor)
        rotulo = f"P(X ≤ {fmt(valor)})"
        nota = "Probabilidade acumulada até o valor informado."
    elif tipo == "inversa":
        if not 0 < valor < 1:
            raise ErroAnalise("Para a inversa, informe uma probabilidade entre 0 e 1.")
        resultado = dist.ppf(valor)
        rotulo = f"quantil x tal que P(X ≤ x) = {fmt(valor)}"
        nota = "Inversa da função de distribuição acumulada."
    else:
        raise ErroAnalise(f"Tipo de cálculo inválido: {tipo}.")

    return ResultadoTabela(
        titulo=f"Distribuição {nome} ({parametros_txt})",
        cabecalhos=["cálculo", "resultado"],
        linhas=[[rotulo, fmt(float(resultado), 6)],
                ["média da distribuição", fmt(float(dist.mean()))],
                ["desvio-padrão da distribuição", fmt(float(dist.std()))]],
        notas=[nota],
    )


def gerar_aleatorios(nome: str, parametros: list[float], quantidade: int,
                     semente: int | None = None) -> np.ndarray:
    if not 1 <= quantidade <= 100_000:
        raise ErroAnalise("Quantidade de valores deve estar entre 1 e 100.000.")
    dist, _, _ = _validar_dist(nome, parametros)
    return dist.rvs(size=quantidade,
                    random_state=np.random.default_rng(semente))


def amostrar_coluna(dados, coluna: str, quantidade: int, reposicao: bool,
                    semente: int | None = None) -> np.ndarray:
    serie = pd.Series(dados).dropna()
    serie = serie[serie.astype(str).str.strip() != ""]
    if serie.empty:
        raise ErroAnalise(f"A coluna '{coluna}' está vazia.")
    if not reposicao and quantidade > len(serie):
        raise ErroAnalise(f"Sem reposição, a amostra ({quantidade}) não pode exceder "
                          f"o número de valores da coluna ({len(serie)}).")
    rng = np.random.default_rng(semente)
    return rng.choice(serie.to_numpy(), size=quantidade, replace=reposicao)


def dados_curva(nome: str, parametros: list[float],
                sombra_ate: float | None = None) -> dict:
    """Pontos para o gráfico da distribuição (pdf/pmf) e área sombreada."""
    dist, discreta, _ = _validar_dist(nome, parametros)
    if discreta:
        inicio = int(max(0, np.floor(dist.ppf(0.0005))))
        fim = int(np.ceil(dist.ppf(0.9995)))
        x = np.arange(inicio, fim + 1)
        y = dist.pmf(x)
    else:
        x = np.linspace(dist.ppf(0.0005), dist.ppf(0.9995), 400)
        y = dist.pdf(x)
    return {"x": x, "y": y, "discreta": discreta,
            "sombra_ate": sombra_ate,
            "prob_sombra": float(dist.cdf(sombra_ate)) if sombra_ate is not None else None}
