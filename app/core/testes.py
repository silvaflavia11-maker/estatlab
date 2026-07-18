"""Testes de hipóteses da Fase 1.

Todos os cálculos vêm de scipy/statsmodels; aqui apenas orquestração,
intervalos de confiança e textos semânticos (H0/H1/conclusão).
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportions_ztest,
)
from statsmodels.stats.rates import test_poisson_2indep

from .resultados import ErroAnalise, ResultadoTeste
from .util import (
    ALT_SCIPY,
    ALT_STATSMODELS,
    SIMBOLO_ALT,
    limpar_numerica,
    resumo_amostra,
    validar_alternativa,
)


def _ic_unilateral(ic: tuple[float, float]) -> tuple[float, float]:
    """scipy devolve ±inf no lado aberto de IC unilateral; mantemos como está."""
    return (float(ic[0]), float(ic[1]))


def _aviso_n_pequeno(avisos: list[str], *tamanhos: int, limite: int = 30) -> None:
    if any(n < limite for n in tamanhos):
        avisos.append(
            "Amostra pequena (n < 30): o resultado depende do pressuposto de "
            "normalidade dos dados. Recomenda-se executar um teste de normalidade "
            "(menu Estatística Básica → Teste de Normalidade)."
        )


# ---------------------------------------------------------------- Z e t

def teste_z_1amostra(dados, coluna: str, mu0: float, sigma: float,
                     alternativa: str = "bilateral", alfa: float = 0.05) -> ResultadoTeste:
    validar_alternativa(alternativa)
    if sigma <= 0:
        raise ErroAnalise("O desvio-padrão populacional (σ) deve ser maior que zero.")
    x = limpar_numerica(dados, coluna)
    n = x.size
    media = float(np.mean(x))
    ep = sigma / np.sqrt(n)
    z = (media - mu0) / ep

    if alternativa == "bilateral":
        p = 2 * stats.norm.sf(abs(z))
        z_crit = stats.norm.ppf(1 - alfa / 2)
        ic = (media - z_crit * ep, media + z_crit * ep)
    elif alternativa == "menor":
        p = stats.norm.cdf(z)
        ic = (-np.inf, media + stats.norm.ppf(1 - alfa) * ep)
    else:
        p = stats.norm.sf(z)
        ic = (media - stats.norm.ppf(1 - alfa) * ep, np.inf)

    avisos: list[str] = []
    _aviso_n_pequeno(avisos, n)
    linha = resumo_amostra(coluna, x)
    linha["desvio-padrão"] = sigma
    linha["EP da média"] = float(ep)
    return ResultadoTeste(
        titulo=f"Teste Z de 1 amostra: {coluna}",
        h0=f"H₀: μ = {mu0} (a média populacional de '{coluna}' é igual a {mu0})",
        h1=f"H₁: μ {SIMBOLO_ALT[alternativa]} {mu0}",
        nome_estatistica="Z",
        estatistica=float(z),
        p_valor=float(p),
        alfa=alfa,
        conclusao_h1=_frase_media(coluna, alternativa, mu0),
        ic=_ic_unilateral(ic),
        nivel_confianca=1 - alfa,
        descricao_ic="IC para a média populacional μ (σ conhecido)",
        amostras=[linha],
        detalhes={"σ (informado)": sigma},
        avisos=avisos,
    )


def _frase_media(coluna: str, alternativa: str, ref, outra: str | None = None) -> str:
    alvo = f"a média de '{outra}'" if outra else str(ref)
    verbo = {"bilateral": "difere de", "menor": "é menor que", "maior": "é maior que"}[alternativa]
    return f"a média de '{coluna}' {verbo} {alvo}"


def teste_t_1amostra(dados, coluna: str, mu0: float,
                     alternativa: str = "bilateral", alfa: float = 0.05) -> ResultadoTeste:
    validar_alternativa(alternativa)
    x = limpar_numerica(dados, coluna)
    res = stats.ttest_1samp(x, mu0, alternative=ALT_SCIPY[alternativa])
    ic = res.confidence_interval(confidence_level=1 - alfa)

    avisos: list[str] = []
    _aviso_n_pequeno(avisos, x.size)
    return ResultadoTeste(
        titulo=f"Teste t de 1 amostra: {coluna}",
        h0=f"H₀: μ = {mu0} (a média populacional de '{coluna}' é igual a {mu0})",
        h1=f"H₁: μ {SIMBOLO_ALT[alternativa]} {mu0}",
        nome_estatistica="t",
        estatistica=float(res.statistic),
        p_valor=float(res.pvalue),
        alfa=alfa,
        gl=float(res.df),
        conclusao_h1=_frase_media(coluna, alternativa, mu0),
        ic=_ic_unilateral((ic.low, ic.high)),
        nivel_confianca=1 - alfa,
        descricao_ic="IC para a média populacional μ",
        amostras=[resumo_amostra(coluna, x)],
        avisos=avisos,
    )


def teste_t_2amostras(dados1, dados2, col1: str, col2: str,
                      variancias_iguais: bool = False,
                      alternativa: str = "bilateral", alfa: float = 0.05) -> ResultadoTeste:
    validar_alternativa(alternativa)
    x1 = limpar_numerica(dados1, col1)
    x2 = limpar_numerica(dados2, col2)
    res = stats.ttest_ind(x1, x2, equal_var=variancias_iguais,
                          alternative=ALT_SCIPY[alternativa])
    ic = res.confidence_interval(confidence_level=1 - alfa)

    avisos: list[str] = []
    _aviso_n_pequeno(avisos, x1.size, x2.size)
    if variancias_iguais:
        avisos.append(
            "Foi assumida a igualdade das variâncias (t combinado). Se houver dúvida, "
            "verifique com o teste de 2 variâncias ou use a opção sem variâncias iguais (Welch)."
        )
    metodo = "variâncias iguais (combinado)" if variancias_iguais else "Welch (variâncias distintas)"
    return ResultadoTeste(
        titulo=f"Teste t de 2 amostras: {col1} × {col2}",
        h0=f"H₀: μ₁ = μ₂ (as médias populacionais de '{col1}' e '{col2}' são iguais)",
        h1=f"H₁: μ₁ {SIMBOLO_ALT[alternativa]} μ₂",
        nome_estatistica="t",
        estatistica=float(res.statistic),
        p_valor=float(res.pvalue),
        alfa=alfa,
        gl=float(res.df),
        conclusao_h1=_frase_media(col1, alternativa, None, outra=col2),
        ic=_ic_unilateral((ic.low, ic.high)),
        nivel_confianca=1 - alfa,
        descricao_ic="IC para a diferença entre as médias (μ₁ − μ₂)",
        amostras=[resumo_amostra(col1, x1), resumo_amostra(col2, x2)],
        detalhes={"método": metodo},
        avisos=avisos,
    )


def teste_t_pareado(dados1, dados2, col1: str, col2: str,
                    alternativa: str = "bilateral", alfa: float = 0.05) -> ResultadoTeste:
    validar_alternativa(alternativa)
    import pandas as pd

    par = pd.DataFrame({"a": pd.to_numeric(pd.Series(dados1), errors="coerce"),
                        "b": pd.to_numeric(pd.Series(dados2), errors="coerce")}).dropna()
    if len(par) < 2:
        raise ErroAnalise(
            f"São necessários pelo menos 2 pares completos de '{col1}' e '{col2}' "
            f"(há {len(par)})."
        )
    x1, x2 = par["a"].to_numpy(), par["b"].to_numpy()
    res = stats.ttest_rel(x1, x2, alternative=ALT_SCIPY[alternativa])
    ic = res.confidence_interval(confidence_level=1 - alfa)
    dif = x1 - x2

    avisos: list[str] = []
    _aviso_n_pequeno(avisos, len(par))
    return ResultadoTeste(
        titulo=f"Teste t pareado: {col1} − {col2}",
        h0=f"H₀: μ_d = 0 (a média das diferenças '{col1}' − '{col2}' é zero)",
        h1=f"H₁: μ_d {SIMBOLO_ALT[alternativa]} 0",
        nome_estatistica="t",
        estatistica=float(res.statistic),
        p_valor=float(res.pvalue),
        alfa=alfa,
        gl=float(res.df),
        conclusao_h1=f"a média de '{col1}' difere da média de '{col2}' (dados pareados)"
        if alternativa == "bilateral"
        else _frase_media(col1, alternativa, None, outra=col2) + " (dados pareados)",
        ic=_ic_unilateral((ic.low, ic.high)),
        nivel_confianca=1 - alfa,
        descricao_ic="IC para a média das diferenças (μ_d)",
        amostras=[resumo_amostra(col1, x1), resumo_amostra(col2, x2),
                  resumo_amostra("diferença", dif)],
        avisos=avisos,
    )


# ---------------------------------------------------------------- Proporções

def teste_1proporcao(sucessos: int, n: int, p0: float,
                     alternativa: str = "bilateral", alfa: float = 0.05) -> ResultadoTeste:
    validar_alternativa(alternativa)
    if not 0 < p0 < 1:
        raise ErroAnalise("A proporção hipotética p₀ deve estar entre 0 e 1.")
    if n <= 0 or sucessos < 0 or sucessos > n:
        raise ErroAnalise("Informe 0 ≤ sucessos ≤ n, com n > 0.")
    res = stats.binomtest(sucessos, n, p0, alternative=ALT_SCIPY[alternativa])
    ic = res.proportion_ci(confidence_level=1 - alfa, method="exact")

    return ResultadoTeste(
        titulo="Teste para 1 proporção (binomial exato)",
        h0=f"H₀: p = {p0} (a proporção populacional é igual a {p0})",
        h1=f"H₁: p {SIMBOLO_ALT[alternativa]} {p0}",
        nome_estatistica="X (nº de sucessos)",
        estatistica=float(sucessos),
        p_valor=float(res.pvalue),
        alfa=alfa,
        conclusao_h1="a proporção populacional "
        + {"bilateral": "difere de", "menor": "é menor que", "maior": "é maior que"}[alternativa]
        + f" {p0}",
        ic=(float(ic.low), float(ic.high)),
        nivel_confianca=1 - alfa,
        descricao_ic="IC exato (Clopper-Pearson) para a proporção p",
        amostras=[{"amostra": "dados", "n": n, "sucessos": sucessos,
                   "proporção amostral": sucessos / n}],
        detalhes={"método": "teste binomial exato"},
    )


def teste_2proporcoes(sucessos1: int, n1: int, sucessos2: int, n2: int,
                      alternativa: str = "bilateral", alfa: float = 0.05) -> ResultadoTeste:
    validar_alternativa(alternativa)
    for s, n, rot in ((sucessos1, n1, "1"), (sucessos2, n2, "2")):
        if n <= 0 or s < 0 or s > n:
            raise ErroAnalise(f"Amostra {rot}: informe 0 ≤ sucessos ≤ n, com n > 0.")
    z, p = proportions_ztest([sucessos1, sucessos2], [n1, n2],
                             alternative=ALT_STATSMODELS[alternativa])
    lo, hi = confint_proportions_2indep(sucessos1, n1, sucessos2, n2,
                                        method="wald", alpha=alfa)

    avisos: list[str] = []
    for s, n, rot in ((sucessos1, n1, "1"), (sucessos2, n2, "2")):
        if min(s, n - s) < 5:
            avisos.append(
                f"Amostra {rot}: menos de 5 sucessos ou fracassos — a aproximação "
                "normal do teste Z pode ser imprecisa."
            )
    return ResultadoTeste(
        titulo="Teste para 2 proporções (aproximação normal)",
        h0="H₀: p₁ = p₂ (as proporções populacionais são iguais)",
        h1=f"H₁: p₁ {SIMBOLO_ALT[alternativa]} p₂",
        nome_estatistica="Z",
        estatistica=float(z),
        p_valor=float(p),
        alfa=alfa,
        conclusao_h1="a proporção da amostra 1 "
        + {"bilateral": "difere da", "menor": "é menor que a", "maior": "é maior que a"}[alternativa]
        + " proporção da amostra 2",
        ic=(float(lo), float(hi)),
        nivel_confianca=1 - alfa,
        descricao_ic="IC (Wald) para a diferença p₁ − p₂",
        amostras=[
            {"amostra": "1", "n": n1, "sucessos": sucessos1, "proporção amostral": sucessos1 / n1},
            {"amostra": "2", "n": n2, "sucessos": sucessos2, "proporção amostral": sucessos2 / n2},
        ],
        avisos=avisos,
    )


# ---------------------------------------------------------------- Variâncias

def teste_1variancia(dados, coluna: str, sigma0: float,
                     alternativa: str = "bilateral", alfa: float = 0.05) -> ResultadoTeste:
    """Teste qui-quadrado para σ (H₀: σ = σ₀)."""
    validar_alternativa(alternativa)
    if sigma0 <= 0:
        raise ErroAnalise("O desvio-padrão hipotético σ₀ deve ser maior que zero.")
    x = limpar_numerica(dados, coluna)
    n = x.size
    s2 = float(np.var(x, ddof=1))
    gl = n - 1
    qui2 = gl * s2 / sigma0**2

    if alternativa == "bilateral":
        p = 2 * min(stats.chi2.cdf(qui2, gl), stats.chi2.sf(qui2, gl))
        ic_var = (gl * s2 / stats.chi2.ppf(1 - alfa / 2, gl),
                  gl * s2 / stats.chi2.ppf(alfa / 2, gl))
    elif alternativa == "menor":
        p = stats.chi2.cdf(qui2, gl)
        ic_var = (0.0, gl * s2 / stats.chi2.ppf(alfa, gl))
    else:
        p = stats.chi2.sf(qui2, gl)
        ic_var = (gl * s2 / stats.chi2.ppf(1 - alfa, gl), np.inf)

    return ResultadoTeste(
        titulo=f"Teste para 1 variância (qui-quadrado): {coluna}",
        h0=f"H₀: σ = {sigma0} (o desvio-padrão populacional é igual a {sigma0})",
        h1=f"H₁: σ {SIMBOLO_ALT[alternativa]} {sigma0}",
        nome_estatistica="χ²",
        estatistica=float(qui2),
        p_valor=float(p),
        alfa=alfa,
        gl=float(gl),
        conclusao_h1="o desvio-padrão de "
        + f"'{coluna}' "
        + {"bilateral": "difere de", "menor": "é menor que", "maior": "é maior que"}[alternativa]
        + f" {sigma0}",
        ic=(float(np.sqrt(ic_var[0])), float(np.sqrt(ic_var[1]))),
        nivel_confianca=1 - alfa,
        descricao_ic="IC para o desvio-padrão populacional σ",
        amostras=[resumo_amostra(coluna, x)],
        avisos=["O teste qui-quadrado para variância é sensível à não normalidade "
                "dos dados; verifique a normalidade antes de interpretar."],
    )


def teste_2variancias(dados1, dados2, col1: str, col2: str,
                      alternativa: str = "bilateral", alfa: float = 0.05) -> ResultadoTeste:
    """Teste F para razão de variâncias (H₀: σ₁ = σ₂)."""
    validar_alternativa(alternativa)
    x1 = limpar_numerica(dados1, col1)
    x2 = limpar_numerica(dados2, col2)
    s1, s2 = float(np.var(x1, ddof=1)), float(np.var(x2, ddof=1))
    if s2 == 0:
        raise ErroAnalise(f"A variância de '{col2}' é zero; o teste F não é definido.")
    gl1, gl2 = x1.size - 1, x2.size - 1
    f = s1 / s2

    if alternativa == "bilateral":
        p = 2 * min(stats.f.cdf(f, gl1, gl2), stats.f.sf(f, gl1, gl2))
        ic = (f / stats.f.ppf(1 - alfa / 2, gl1, gl2), f / stats.f.ppf(alfa / 2, gl1, gl2))
    elif alternativa == "menor":
        p = stats.f.cdf(f, gl1, gl2)
        ic = (0.0, f / stats.f.ppf(alfa, gl1, gl2))
    else:
        p = stats.f.sf(f, gl1, gl2)
        ic = (f / stats.f.ppf(1 - alfa, gl1, gl2), np.inf)

    return ResultadoTeste(
        titulo=f"Teste F para 2 variâncias: {col1} × {col2}",
        h0=f"H₀: σ₁² / σ₂² = 1 (as variâncias de '{col1}' e '{col2}' são iguais)",
        h1=f"H₁: σ₁² / σ₂² {SIMBOLO_ALT[alternativa]} 1",
        nome_estatistica="F",
        estatistica=float(f),
        p_valor=float(p),
        alfa=alfa,
        gl=None,
        conclusao_h1=f"a variância de '{col1}' "
        + {"bilateral": "difere da", "menor": "é menor que a", "maior": "é maior que a"}[alternativa]
        + f" variância de '{col2}'",
        ic=(float(ic[0]), float(ic[1])),
        nivel_confianca=1 - alfa,
        descricao_ic="IC para a razão de variâncias σ₁²/σ₂²",
        amostras=[resumo_amostra(col1, x1), resumo_amostra(col2, x2)],
        detalhes={"gl numerador": gl1, "gl denominador": gl2},
        avisos=["O teste F é sensível à não normalidade dos dados; verifique a "
                "normalidade antes de interpretar."],
    )


# ---------------------------------------------------------------- Poisson

def _ic_taxa_poisson(eventos: int, exposicao: float, alfa: float) -> tuple[float, float]:
    """IC exato (Garwood) para taxa de Poisson via relação com a qui-quadrado."""
    lo = 0.0 if eventos == 0 else stats.chi2.ppf(alfa / 2, 2 * eventos) / (2 * exposicao)
    hi = stats.chi2.ppf(1 - alfa / 2, 2 * eventos + 2) / (2 * exposicao)
    return (float(lo), float(hi))


def teste_taxa_poisson_1amostra(eventos: int, exposicao: float, taxa0: float,
                                alternativa: str = "bilateral",
                                alfa: float = 0.05) -> ResultadoTeste:
    """Teste exato para taxa de ocorrência (H₀: λ = λ₀)."""
    validar_alternativa(alternativa)
    if exposicao <= 0 or taxa0 <= 0 or eventos < 0:
        raise ErroAnalise("Informe eventos ≥ 0, exposição > 0 e taxa hipotética λ₀ > 0.")
    mu = taxa0 * exposicao
    p_menor = float(stats.poisson.cdf(eventos, mu))
    p_maior = float(stats.poisson.sf(eventos - 1, mu))
    if alternativa == "bilateral":
        p = min(1.0, 2 * min(p_menor, p_maior))
    elif alternativa == "menor":
        p = p_menor
    else:
        p = p_maior

    return ResultadoTeste(
        titulo="Teste para taxa de Poisson (1 amostra, exato)",
        h0=f"H₀: λ = {taxa0} (a taxa de ocorrência é igual a {taxa0})",
        h1=f"H₁: λ {SIMBOLO_ALT[alternativa]} {taxa0}",
        nome_estatistica="X (nº de eventos)",
        estatistica=float(eventos),
        p_valor=float(p),
        alfa=alfa,
        conclusao_h1="a taxa de ocorrência "
        + {"bilateral": "difere de", "menor": "é menor que", "maior": "é maior que"}[alternativa]
        + f" {taxa0}",
        ic=_ic_taxa_poisson(eventos, exposicao, alfa),
        nivel_confianca=1 - alfa,
        descricao_ic="IC exato (Garwood) para a taxa λ",
        amostras=[{"amostra": "dados", "eventos": eventos, "exposição": exposicao,
                   "taxa amostral": eventos / exposicao}],
        detalhes={"método": "teste exato de Poisson"},
    )


def teste_taxa_poisson_2amostras(eventos1: int, exposicao1: float,
                                 eventos2: int, exposicao2: float,
                                 alternativa: str = "bilateral",
                                 alfa: float = 0.05) -> ResultadoTeste:
    """Compara duas taxas de Poisson (H₀: λ₁ = λ₂), método escore."""
    validar_alternativa(alternativa)
    if exposicao1 <= 0 or exposicao2 <= 0 or eventos1 < 0 or eventos2 < 0:
        raise ErroAnalise("Informe eventos ≥ 0 e exposições > 0 nas duas amostras.")
    res = test_poisson_2indep(eventos1, exposicao1, eventos2, exposicao2,
                              value=1.0, method="score",
                              alternative=ALT_STATSMODELS[alternativa])
    return ResultadoTeste(
        titulo="Teste para 2 taxas de Poisson (método escore)",
        h0="H₀: λ₁ = λ₂ (as taxas de ocorrência são iguais)",
        h1=f"H₁: λ₁ {SIMBOLO_ALT[alternativa]} λ₂",
        nome_estatistica="Z",
        estatistica=float(res.statistic),
        p_valor=float(res.pvalue),
        alfa=alfa,
        conclusao_h1="a taxa da amostra 1 "
        + {"bilateral": "difere da", "menor": "é menor que a", "maior": "é maior que a"}[alternativa]
        + " taxa da amostra 2",
        amostras=[
            {"amostra": "1", "eventos": eventos1, "exposição": exposicao1,
             "taxa amostral": eventos1 / exposicao1},
            {"amostra": "2", "eventos": eventos2, "exposição": exposicao2,
             "taxa amostral": eventos2 / exposicao2},
        ],
        detalhes={"razão de taxas (amostral)":
                  (eventos1 / exposicao1) / (eventos2 / exposicao2)
                  if eventos2 > 0 else float("nan")},
    )
