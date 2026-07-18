"""Poder e tamanho de amostra.

Testes t e ANOVA: statsmodels.stats.power. Proporções: aproximação normal
(statsmodels NormalIndPower para 2 proporções; fórmula clássica para 1).
Variâncias: fórmulas exatas qui-quadrado/F (Desmond & Ostle; ver testes de
referência). Em cada função, exatamente um entre ``n`` e ``poder`` deve ser
None — o valor faltante é calculado.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats
from statsmodels.stats.power import (
    FTestAnovaPower,
    NormalIndPower,
    TTestIndPower,
    TTestPower,
)
from statsmodels.stats.proportion import proportion_effectsize

from .resultados import ErroAnalise, ResultadoComposto

_ALT_SM = {"bilateral": "two-sided", "menor": "smaller", "maior": "larger"}


def _validar_entrada(n, poder):
    if (n is None) == (poder is None):
        raise ErroAnalise("Informe exatamente um: tamanho de amostra OU poder "
                          "desejado (o outro será calculado).")
    if poder is not None and not 0 < poder < 1:
        raise ErroAnalise("O poder deve estar entre 0 e 1 (ex.: 0,80).")
    if n is not None and n < 2:
        raise ErroAnalise("O tamanho de amostra deve ser pelo menos 2.")


def _resultado(titulo: str, parametros: list[list], n, poder,
               nota_extra: str | None = None,
               dados: dict | None = None) -> ResultadoComposto:
    from app.reports.formatacao import fmt

    if poder is None:
        alvo = ["poder calculado", fmt(float(dados["poder_calc"]), 4)]
        frase = (f"Com n = {int(n)}, a probabilidade de detectar o efeito "
                 f"especificado (poder) é {fmt(100 * float(dados['poder_calc']), 1)}%.")
    else:
        alvo = ["n calculado (por grupo/amostra)", str(int(dados["n_calc"]))]
        frase = (f"Para atingir poder de {fmt(100 * poder, 0)}%, são necessárias "
                 f"aproximadamente {int(dados['n_calc'])} observações "
                 "(por amostra, quando houver mais de uma).")
    itens: list[tuple] = [
        ("tabela", ["parâmetro", "valor"], parametros + [alvo]),
        ("interpretacao", frase + " Poder é a probabilidade de rejeitar H₀ quando "
                          "o efeito realmente existe; 80% é o mínimo usual."),
    ]
    if nota_extra:
        itens.append(("nota", nota_extra))
    return ResultadoComposto(titulo=titulo, itens=itens, dados=dados or {})


def poder_t(tipo: str, diferenca: float, desvio: float, alfa: float = 0.05,
            n: float | None = None, poder: float | None = None,
            alternativa: str = "bilateral") -> ResultadoComposto:
    """``tipo``: '1amostra', 'pareado' ou '2amostras'. ``diferenca`` é a
    diferença mínima de interesse; ``desvio`` o desvio-padrão (das diferenças,
    no pareado)."""
    from app.reports.formatacao import fmt

    _validar_entrada(n, poder)
    if desvio <= 0:
        raise ErroAnalise("O desvio-padrão deve ser maior que zero.")
    if diferenca == 0:
        raise ErroAnalise("A diferença de interesse não pode ser zero.")
    efeito = diferenca / desvio
    motor = TTestIndPower() if tipo == "2amostras" else TTestPower()
    rotulos = {"1amostra": "teste t de 1 amostra", "pareado": "teste t pareado",
               "2amostras": "teste t de 2 amostras"}
    if tipo not in rotulos:
        raise ErroAnalise(f"Tipo inválido: {tipo}.")

    def resolver(**kwargs) -> float:
        base = {"effect_size": efeito, "alpha": alfa,
                "alternative": _ALT_SM[alternativa]}
        if tipo == "2amostras":
            return float(motor.solve_power(**base, ratio=1.0, **kwargs))
        return float(motor.solve_power(**base, **kwargs))

    chave_n = "nobs1" if tipo == "2amostras" else "nobs"
    dados: dict = {"efeito": efeito}
    if poder is None:
        dados["poder_calc"] = resolver(**{chave_n: n, "power": None})
    else:
        n_calc = resolver(**{chave_n: None, "power": poder})
        dados["n_calc"] = int(np.ceil(n_calc))

    dados["curva"] = lambda m: resolver(**{chave_n: m, "power": None})
    return _resultado(
        f"Poder e tamanho de amostra: {rotulos[tipo]}",
        [["diferença de interesse", fmt(diferenca)],
         ["desvio-padrão suposto", fmt(desvio)],
         ["tamanho de efeito (d)", fmt(efeito, 3)],
         ["α", fmt(alfa, 3)], ["alternativa", alternativa]]
        + ([["n informado", str(int(n))]] if n is not None else []),
        n, poder,
        nota_extra="No teste de 2 amostras, n refere-se a cada grupo.",
        dados=dados)


def poder_z_1amostra(diferenca: float, sigma: float, alfa: float = 0.05,
                     n: float | None = None, poder: float | None = None,
                     alternativa: str = "bilateral") -> ResultadoComposto:
    from app.reports.formatacao import fmt

    _validar_entrada(n, poder)
    if sigma <= 0:
        raise ErroAnalise("σ deve ser maior que zero.")
    if diferenca == 0:
        raise ErroAnalise("A diferença de interesse não pode ser zero.")

    def poder_para_n(m: float) -> float:
        efeito = abs(diferenca) / (sigma / np.sqrt(m))
        if alternativa == "bilateral":
            zc = stats.norm.ppf(1 - alfa / 2)
            return float(stats.norm.sf(zc - efeito) + stats.norm.cdf(-zc - efeito))
        zc = stats.norm.ppf(1 - alfa)
        return float(stats.norm.sf(zc - efeito))

    dados: dict = {"curva": poder_para_n}
    if poder is None:
        dados["poder_calc"] = poder_para_n(n)
    else:
        raiz = optimize.brentq(lambda m: poder_para_n(m) - poder, 2, 1e7)
        dados["n_calc"] = int(np.ceil(raiz))
    return _resultado(
        "Poder e tamanho de amostra: teste Z de 1 amostra",
        [["diferença de interesse", fmt(diferenca)], ["σ (conhecido)", fmt(sigma)],
         ["α", fmt(alfa, 3)], ["alternativa", alternativa]]
        + ([["n informado", str(int(n))]] if n is not None else []),
        n, poder, dados=dados)


def poder_proporcoes(tipo: str, p0: float, p1: float, alfa: float = 0.05,
                     n: float | None = None, poder: float | None = None,
                     alternativa: str = "bilateral") -> ResultadoComposto:
    """``tipo``: '1proporcao' (p0 = hipotética, p1 = real suposta) ou
    '2proporcoes' (p0, p1 = proporções dos dois grupos)."""
    from app.reports.formatacao import fmt

    _validar_entrada(n, poder)
    for p in (p0, p1):
        if not 0 < p < 1:
            raise ErroAnalise("As proporções devem estar entre 0 e 1.")
    if p0 == p1:
        raise ErroAnalise("As proporções comparadas devem ser diferentes.")

    if tipo == "2proporcoes":
        efeito = proportion_effectsize(p1, p0)
        motor = NormalIndPower()

        def poder_para_n(m: float) -> float:
            return float(motor.solve_power(effect_size=efeito, nobs1=m, alpha=alfa,
                                           power=None,
                                           alternative=_ALT_SM[alternativa]))

        titulo = "Poder e tamanho de amostra: 2 proporções"
        rotulos = [["proporção do grupo 1", fmt(p1, 3)],
                   ["proporção do grupo 2", fmt(p0, 3)],
                   ["tamanho de efeito (h)", fmt(float(efeito), 3)]]
        nota = "n refere-se a cada grupo; método: aproximação normal (arco-seno)."
    elif tipo == "1proporcao":
        def poder_para_n(m: float) -> float:
            ep0 = np.sqrt(p0 * (1 - p0) / m)
            ep1 = np.sqrt(p1 * (1 - p1) / m)
            if alternativa == "bilateral":
                zc = stats.norm.ppf(1 - alfa / 2)
                acima = stats.norm.sf((p0 + zc * ep0 - p1) / ep1)
                abaixo = stats.norm.cdf((p0 - zc * ep0 - p1) / ep1)
                return float(acima + abaixo)
            zc = stats.norm.ppf(1 - alfa)
            if alternativa == "maior":
                return float(stats.norm.sf((p0 + zc * ep0 - p1) / ep1))
            return float(stats.norm.cdf((p0 - zc * ep0 - p1) / ep1))

        titulo = "Poder e tamanho de amostra: 1 proporção"
        rotulos = [["proporção hipotética (p₀)", fmt(p0, 3)],
                   ["proporção real suposta (p₁)", fmt(p1, 3)]]
        nota = "Método: aproximação normal; para n pequeno o teste exato difere."
    else:
        raise ErroAnalise(f"Tipo inválido: {tipo}.")

    dados: dict = {"curva": poder_para_n}
    if poder is None:
        dados["poder_calc"] = poder_para_n(n)
    else:
        raiz = optimize.brentq(lambda m: poder_para_n(m) - poder, 2, 1e7)
        dados["n_calc"] = int(np.ceil(raiz))
    return _resultado(titulo,
                      rotulos + [["α", fmt(alfa, 3)], ["alternativa", alternativa]]
                      + ([["n informado", str(int(n))]] if n is not None else []),
                      n, poder, nota_extra=nota, dados=dados)


def poder_variancias(tipo: str, razao: float, alfa: float = 0.05,
                     n: float | None = None,
                     poder: float | None = None) -> ResultadoComposto:
    """``tipo``: '1variancia' (razão = σ/σ₀) ou '2variancias' (razão = σ₁/σ₂).
    Teste bilateral; fórmulas exatas qui-quadrado/F."""
    from app.reports.formatacao import fmt

    _validar_entrada(n, poder)
    if razao <= 0 or razao == 1:
        raise ErroAnalise("A razão de desvios-padrão deve ser positiva e "
                          "diferente de 1.")
    r2 = razao**2

    if tipo == "1variancia":
        def poder_para_n(m: float) -> float:
            gl = m - 1
            c_sup = stats.chi2.ppf(1 - alfa / 2, gl)
            c_inf = stats.chi2.ppf(alfa / 2, gl)
            return float(stats.chi2.sf(c_sup / r2, gl) + stats.chi2.cdf(c_inf / r2, gl))
        titulo = "Poder e tamanho de amostra: 1 variância (qui-quadrado)"
        rotulo_razao = "razão σ/σ₀"
    elif tipo == "2variancias":
        def poder_para_n(m: float) -> float:
            gl = m - 1
            c_sup = stats.f.ppf(1 - alfa / 2, gl, gl)
            c_inf = stats.f.ppf(alfa / 2, gl, gl)
            return float(stats.f.sf(c_sup / r2, gl, gl) + stats.f.cdf(c_inf / r2, gl, gl))
        titulo = "Poder e tamanho de amostra: 2 variâncias (teste F)"
        rotulo_razao = "razão σ₁/σ₂"
    else:
        raise ErroAnalise(f"Tipo inválido: {tipo}.")

    dados: dict = {"curva": poder_para_n}
    if poder is None:
        dados["poder_calc"] = poder_para_n(n)
    else:
        if poder_para_n(1e7) < poder:
            raise ErroAnalise("Poder inatingível com essa razão; aumente a razão.")
        raiz = optimize.brentq(lambda m: poder_para_n(m) - poder, 3, 1e7)
        dados["n_calc"] = int(np.ceil(raiz))
    return _resultado(titulo,
                      [[rotulo_razao, fmt(razao, 3)], ["α", fmt(alfa, 3)],
                       ["alternativa", "bilateral"]]
                      + ([["n informado", str(int(n))]] if n is not None else []),
                      n, poder,
                      nota_extra="No teste de 2 variâncias, n refere-se a cada grupo.",
                      dados=dados)


def poder_anova(k_grupos: int, efeito_f: float, alfa: float = 0.05,
                n: float | None = None,
                poder: float | None = None) -> ResultadoComposto:
    """ANOVA de 1 fator; ``efeito_f`` = σ_entre/σ_dentro (f de Cohen:
    0,10 pequeno, 0,25 médio, 0,40 grande). ``n`` é por grupo."""
    from app.reports.formatacao import fmt

    _validar_entrada(n, poder)
    if k_grupos < 2:
        raise ErroAnalise("São necessários pelo menos 2 grupos.")
    if efeito_f <= 0:
        raise ErroAnalise("O tamanho de efeito f deve ser maior que zero.")
    motor = FTestAnovaPower()

    def poder_para_n(m: float) -> float:
        return float(motor.solve_power(effect_size=efeito_f, nobs=m * k_grupos,
                                       alpha=alfa, power=None, k_groups=k_grupos))

    dados: dict = {"curva": poder_para_n}
    if poder is None:
        dados["poder_calc"] = poder_para_n(n)
    else:
        raiz = optimize.brentq(lambda m: poder_para_n(m) - poder, 2, 1e6)
        dados["n_calc"] = int(np.ceil(raiz))
    return _resultado(
        "Poder e tamanho de amostra: ANOVA de 1 fator",
        [["número de grupos (k)", str(k_grupos)],
         ["tamanho de efeito f (σ_entre/σ_dentro)", fmt(efeito_f, 3)],
         ["α", fmt(alfa, 3)]]
        + ([["n informado (por grupo)", str(int(n))]] if n is not None else []),
        n, poder,
        nota_extra="Referência (Cohen): f = 0,10 pequeno; 0,25 médio; 0,40 grande. "
                   "n é por grupo.",
        dados=dados)
