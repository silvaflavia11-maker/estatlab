"""Validação numérica do motor estatístico.

Estratégia: os testes comparam a orquestração de app.core com (a) as
bibliotecas de referência chamadas diretamente, (b) fórmulas recomputadas
de forma independente e (c) valores clássicos publicados (IC exato de
Poisson, valores críticos de Dixon).
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from app.core.correlacao import correlacao, covariancia
from app.core.descritiva import descritiva
from app.core.normalidade import MET_AD, MET_KS, MET_SW, teste_normalidade
from app.core.outliers import teste_dixon, teste_grubbs
from app.core.resultados import ErroAnalise
from app.core.testes import (
    teste_1proporcao,
    teste_1variancia,
    teste_2proporcoes,
    teste_2variancias,
    teste_t_1amostra,
    teste_t_2amostras,
    teste_t_pareado,
    teste_taxa_poisson_1amostra,
    teste_taxa_poisson_2amostras,
    teste_z_1amostra,
)

rng = np.random.default_rng(42)
X1 = rng.normal(10, 2, 25)
X2 = rng.normal(11, 3, 30)


def test_descritiva_valores():
    r = descritiva(X1, "x")
    assert r.n == 25
    assert r.media == pytest.approx(np.mean(X1), abs=1e-10)
    assert r.dp == pytest.approx(np.std(X1, ddof=1), abs=1e-10)
    assert r.mediana == pytest.approx(np.median(X1), abs=1e-10)
    # IC t recomputado de forma independente
    ep = np.std(X1, ddof=1) / np.sqrt(25)
    t = stats.t.ppf(0.975, 24)
    assert r.ic_media[0] == pytest.approx(np.mean(X1) - t * ep, abs=1e-10)
    assert r.ic_media[1] == pytest.approx(np.mean(X1) + t * ep, abs=1e-10)
    # skew/kurt na convenção ajustada (pandas/Minitab)
    assert r.assimetria == pytest.approx(pd.Series(X1).skew(), abs=1e-12)


def test_descritiva_ausentes():
    dados = [1.0, 2.0, None, 4.0, float("nan")]
    r = descritiva(dados, "x")
    assert r.n == 3
    assert r.ausentes == 2


def test_z_1amostra_recalculo_manual():
    r = teste_z_1amostra(X1, "x", mu0=10, sigma=2)
    z_esperado = (np.mean(X1) - 10) / (2 / np.sqrt(25))
    assert r.estatistica == pytest.approx(z_esperado, abs=1e-10)
    assert r.p_valor == pytest.approx(2 * stats.norm.sf(abs(z_esperado)), abs=1e-10)


def test_z_unilateral():
    r = teste_z_1amostra(X1, "x", mu0=9, sigma=2, alternativa="maior")
    z = (np.mean(X1) - 9) / (2 / np.sqrt(25))
    assert r.p_valor == pytest.approx(stats.norm.sf(z), abs=1e-12)


def test_t_1amostra_vs_scipy():
    r = teste_t_1amostra(X1, "x", mu0=10)
    ref = stats.ttest_1samp(X1, 10)
    assert r.estatistica == pytest.approx(ref.statistic, abs=1e-12)
    assert r.p_valor == pytest.approx(ref.pvalue, abs=1e-12)
    assert r.gl == 24


def test_t_2amostras_welch_vs_scipy():
    r = teste_t_2amostras(X1, X2, "a", "b", variancias_iguais=False)
    ref = stats.ttest_ind(X1, X2, equal_var=False)
    assert r.estatistica == pytest.approx(ref.statistic, abs=1e-12)
    assert r.p_valor == pytest.approx(ref.pvalue, abs=1e-12)
    ic = ref.confidence_interval(0.95)
    assert r.ic == pytest.approx((ic.low, ic.high), abs=1e-12)


def test_t_pareado_vs_scipy():
    a, b = X1, X1 + rng.normal(0.5, 1, 25)
    r = teste_t_pareado(a, b, "a", "b")
    ref = stats.ttest_rel(a, b)
    assert r.estatistica == pytest.approx(ref.statistic, abs=1e-12)
    assert r.p_valor == pytest.approx(ref.pvalue, abs=1e-12)


def test_t_pareado_remove_pares_incompletos():
    a = [1.0, 2.0, 3.0, None]
    b = [1.1, None, 3.3, 4.0]
    r = teste_t_pareado(a, b, "a", "b")
    assert r.amostras[0]["n"] == 2  # só 2 pares completos


def test_1proporcao_exato_vs_scipy():
    r = teste_1proporcao(8, 50, 0.25)
    ref = stats.binomtest(8, 50, 0.25)
    assert r.p_valor == pytest.approx(ref.pvalue, abs=1e-12)
    ci = ref.proportion_ci(0.95, method="exact")
    assert r.ic == pytest.approx((ci.low, ci.high), abs=1e-12)


def test_2proporcoes_vs_statsmodels():
    from statsmodels.stats.proportion import proportions_ztest

    r = teste_2proporcoes(30, 100, 45, 110)
    z, p = proportions_ztest([30, 45], [100, 110])
    assert r.estatistica == pytest.approx(z, abs=1e-12)
    assert r.p_valor == pytest.approx(p, abs=1e-12)


def test_1variancia_formula():
    r = teste_1variancia(X1, "x", sigma0=2.0)
    s2 = np.var(X1, ddof=1)
    qui2 = 24 * s2 / 4.0
    assert r.estatistica == pytest.approx(qui2, abs=1e-10)
    p = 2 * min(stats.chi2.cdf(qui2, 24), stats.chi2.sf(qui2, 24))
    assert r.p_valor == pytest.approx(p, abs=1e-12)
    # IC bilateral para sigma
    lo = np.sqrt(24 * s2 / stats.chi2.ppf(0.975, 24))
    hi = np.sqrt(24 * s2 / stats.chi2.ppf(0.025, 24))
    assert r.ic == pytest.approx((lo, hi), abs=1e-10)


def test_2variancias_formula():
    r = teste_2variancias(X1, X2, "a", "b")
    f = np.var(X1, ddof=1) / np.var(X2, ddof=1)
    assert r.estatistica == pytest.approx(f, abs=1e-12)
    p = 2 * min(stats.f.cdf(f, 24, 29), stats.f.sf(f, 24, 29))
    assert r.p_valor == pytest.approx(p, abs=1e-12)


def test_poisson_1amostra_ic_garwood_classico():
    # IC exato 95% clássico para 10 eventos em exposição 1: (4,7954; 18,3904)
    r = teste_taxa_poisson_1amostra(10, 1.0, taxa0=5.0)
    assert r.ic[0] == pytest.approx(4.7954, abs=1e-3)
    assert r.ic[1] == pytest.approx(18.3904, abs=1e-3)


def test_poisson_1amostra_p_exato():
    r = teste_taxa_poisson_1amostra(15, 10.0, taxa0=1.0, alternativa="maior")
    # P(X >= 15) com mu = 10
    assert r.p_valor == pytest.approx(stats.poisson.sf(14, 10.0), abs=1e-12)


def test_poisson_2amostras_vs_statsmodels():
    from statsmodels.stats.rates import test_poisson_2indep

    r = teste_taxa_poisson_2amostras(20, 100.0, 35, 120.0)
    ref = test_poisson_2indep(20, 100.0, 35, 120.0, value=1.0, method="score")
    assert r.estatistica == pytest.approx(ref.statistic, abs=1e-12)
    assert r.p_valor == pytest.approx(ref.pvalue, abs=1e-12)


def test_correlacao_pearson_vs_scipy():
    df = pd.DataFrame({"a": X1, "b": X1 * 0.5 + rng.normal(0, 1, 25)})
    r = correlacao(df, ["a", "b"], metodo="pearson")
    ref = stats.pearsonr(df["a"], df["b"])
    # r está na 3ª coluna da 1ª linha, formatado PT-BR
    r_valor = float(r.linhas[0][2].replace(".", "").replace(",", "."))
    assert r_valor == pytest.approx(ref.statistic, abs=1e-4)


def test_covariancia_vs_pandas():
    df = pd.DataFrame({"a": X1, "b": X1 * 2})
    r = covariancia(df, ["a", "b"])
    cov = float(r.linhas[0][2].replace(".", "").replace(",", "."))
    assert cov == pytest.approx(df.cov().loc["a", "b"], rel=1e-3)


def test_normalidade_vs_bibliotecas():
    from statsmodels.stats.diagnostic import lilliefors, normal_ad

    r_ad = teste_normalidade(X1, "x", MET_AD)
    stat, p = normal_ad(X1)
    assert r_ad.estatistica == pytest.approx(stat, abs=1e-12)
    assert r_ad.p_valor == pytest.approx(p, abs=1e-12)

    r_sw = teste_normalidade(X1, "x", MET_SW)
    ref = stats.shapiro(X1)
    assert r_sw.estatistica == pytest.approx(ref.statistic, abs=1e-12)

    r_ks = teste_normalidade(X1, "x", MET_KS)
    stat, p = lilliefors(X1, dist="norm")
    assert r_ks.estatistica == pytest.approx(stat, abs=1e-12)


def test_grubbs_deteccao_e_formula():
    dados = [9.8, 10.1, 10.0, 10.2, 9.9, 10.1, 25.0]
    r = teste_grubbs(dados, "x")
    x = np.array(dados)
    g_manual = abs(25.0 - x.mean()) / x.std(ddof=1)
    assert r.estatistica == pytest.approx(g_manual, abs=1e-10)
    assert r.p_valor < 0.05  # 25,0 é outlier claro
    assert r.detalhes["valor suspeito"] == 25.0
    # G crítico recomputado (Grubbs 1969)
    n = 7
    t = stats.t.ppf(1 - 0.05 / (2 * n), n - 2)
    g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t**2 / (n - 2 + t**2))
    assert r.detalhes["G crítico"] == pytest.approx(g_crit, abs=1e-10)


def test_grubbs_sem_outlier():
    r = teste_grubbs(X1, "x")
    assert r.p_valor > 0.05


def test_dixon_calculo_manual():
    # Q = (10-4)/(10-1) = 0,6667 < 0,710 (crítico para n=5, α=0,05) → não rejeita
    dados = [1.0, 2.0, 3.0, 4.0, 10.0]
    r = teste_dixon(dados, "x")
    assert r.estatistica == pytest.approx(6 / 9, abs=1e-10)
    assert r.detalhes["Q crítico (tabelado)"] == 0.710
    assert r.p_valor == 1.0  # não rejeita


def test_dixon_rejeita():
    dados = [1.0, 1.1, 1.05, 1.02, 9.0]
    r = teste_dixon(dados, "x")  # Q = (9-1,1)/(9-1) = 0,9875 > 0,710
    assert r.p_valor == 0.0
    assert r.detalhes["valor suspeito"] == 9.0


def test_dixon_n_grande_erro():
    with pytest.raises(ErroAnalise, match="Grubbs"):
        teste_dixon(list(range(20)), "x")


def test_erros_amigaveis():
    with pytest.raises(ErroAnalise):
        teste_t_1amostra([1.0], "x", 0)  # n insuficiente
    with pytest.raises(ErroAnalise):
        teste_z_1amostra(X1, "x", 10, sigma=-1)  # sigma inválido
    with pytest.raises(ErroAnalise):
        teste_1proporcao(60, 50, 0.5)  # sucessos > n
    with pytest.raises(ErroAnalise):
        teste_t_1amostra(["a", "b", "c"], "x", 0)  # coluna não numérica


def test_formatacao_ptbr():
    from app.reports.formatacao import fmt, fmt_p

    assert fmt(1234.5678) == "1.234,5678"
    assert fmt(0.05, 2) == "0,05"
    assert fmt_p(0.0004) == "< 0,001"
    assert fmt_p(0.032) == "0,032"
    assert fmt(float("nan")) == "—"


def test_render_teste_interpretacao():
    from app.reports.formatacao import render_teste

    r = teste_t_1amostra(X1, "medidas", mu0=0)  # p certamente < 0,05
    html = render_teste(r)
    assert "rejeita-se H₀" in html
    assert "Hipóteses" in html
    assert "α = 0,05" in html
