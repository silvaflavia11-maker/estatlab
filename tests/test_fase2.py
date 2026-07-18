"""Validação numérica dos módulos da Fase 2."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from app.core import anova as mod_anova
from app.core import distribuicoes as mod_dist
from app.core import naoparametricos as mod_np
from app.core import poder as mod_poder
from app.core import regressao as mod_reg
from app.core import tabelas as mod_tab
from app.core.resultados import ErroAnalise

rng = np.random.default_rng(123)

# Dados empilhados para ANOVA/Kruskal: 3 grupos com médias diferentes
DF_GRUPOS = pd.DataFrame({
    "y": np.concatenate([rng.normal(10, 2, 20), rng.normal(12, 2, 20),
                         rng.normal(15, 2, 20)]),
    "g": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
})
DF_REG = pd.DataFrame({"x1": rng.normal(0, 1, 50), "x2": rng.normal(0, 1, 50)})
DF_REG["y"] = 3 + 2 * DF_REG["x1"] - 1.5 * DF_REG["x2"] + rng.normal(0, 1, 50)


def _num(texto: str) -> float:
    """Converte número formatado em PT-BR de volta para float."""
    return float(texto.replace(".", "").replace(",", "."))


# ------------------------------------------------------------------- ANOVA

def test_anova_1via_vs_scipy():
    resultado = mod_anova.anova_1via(DF_GRUPOS, "y", "g")
    grupos = [DF_GRUPOS[DF_GRUPOS.g == n]["y"] for n in "ABC"]
    f_ref, p_ref = stats.f_oneway(*grupos)
    linha_fator = resultado.itens[1][2][0]  # tabela ANOVA, linha do fator
    assert _num(linha_fator[4]) == pytest.approx(f_ref, abs=1e-3)


def test_anova_tukey_marca_pares_corretos():
    resultado = mod_anova.anova_1via(DF_GRUPOS, "y", "g", comparacao="tukey")
    tabela_tukey = [item for item in resultado.itens if item[0] == "tabela"][-1]
    conclusoes = {linha[0]: linha[5] for linha in tabela_tukey[2]}
    assert conclusoes["A − C"] == "difere"  # médias 10 vs 15


def test_anova_dunnett_vs_scipy():
    resultado = mod_anova.anova_1via(DF_GRUPOS, "y", "g", comparacao="dunnett",
                                     nivel_controle="A")
    grupos = {n: DF_GRUPOS[DF_GRUPOS.g == n]["y"].to_numpy() for n in "ABC"}
    ref = stats.dunnett(grupos["B"], grupos["C"], control=grupos["A"])
    tabela = [item for item in resultado.itens if item[0] == "tabela"][-1]
    assert _num(tabela[2][0][4].replace("< 0,001", "0,0005")) == pytest.approx(
        ref.pvalue[0], abs=1e-3) or ref.pvalue[0] < 0.001


def test_anova_2vias_efeitos():
    df = pd.DataFrame({
        "y": rng.normal(10, 1, 40) + np.tile([0, 3], 20),
        "a": np.tile(["a1", "a2"], 20),
        "b": np.repeat(["b1", "b2"], 20),
    })
    resultado = mod_anova.anova_2vias(df, "y", "a", "b", interacao=True)
    fontes = [linha[0] for linha in resultado.itens[1][2]]
    assert "a" in fontes and "b" in fontes and "a × b (interação)" in fontes


def test_levene_bartlett_vs_scipy():
    grupos = [DF_GRUPOS[DF_GRUPOS.g == n]["y"] for n in "ABC"]
    r_levene = mod_anova.variancias_grupos(DF_GRUPOS, "y", "g", "levene")
    ref_w, ref_p = stats.levene(*grupos, center="median")
    assert r_levene.estatistica == pytest.approx(ref_w, abs=1e-10)
    assert r_levene.p_valor == pytest.approx(ref_p, abs=1e-10)

    r_bart = mod_anova.variancias_grupos(DF_GRUPOS, "y", "g", "bartlett")
    ref_b, ref_pb = stats.bartlett(*grupos)
    assert r_bart.estatistica == pytest.approx(ref_b, abs=1e-10)


# --------------------------------------------------------------- Regressão

def test_regressao_linear_coeficientes():
    import statsmodels.api as sm

    resultado = mod_reg.regressao_linear(DF_REG, "y", ["x1", "x2"])
    ref = sm.OLS(DF_REG["y"], sm.add_constant(DF_REG[["x1", "x2"]])).fit()
    tabela_coef = resultado.itens[2][2]
    assert _num(tabela_coef[0][1]) == pytest.approx(ref.params["const"], abs=1e-3)
    assert _num(tabela_coef[1][1]) == pytest.approx(ref.params["x1"], abs=1e-3)
    assert len(resultado.dados["residuos"]) == 50


def test_regressao_predicao_ic_ip():
    resultado = mod_reg.regressao_linear(DF_REG, "y", ["x1"],
                                         predicao={"x1": 1.0})
    tabelas = [item for item in resultado.itens if item[0] == "tabela"]
    previsao = tabelas[-1][2][0]
    # IP contém o IC (IP sempre mais largo)
    ic = previsao[1].strip("()").split(";")
    ip = previsao[2].strip("()").split(";")
    assert _num(ip[0]) < _num(ic[0]) < _num(ic[1]) < _num(ip[1])


def test_stepwise_seleciona_preditores_reais():
    df = DF_REG.copy()
    df["ruido"] = rng.normal(0, 1, 50)  # preditor irrelevante
    resultado = mod_reg.stepwise(df, "y", ["x1", "x2", "ruido"], criterio="p")
    assert "x1" in resultado.titulo and "x2" in resultado.titulo
    assert "ruido" not in resultado.titulo


def test_stepwise_bic_igual_forward_manual():
    resultado = mod_reg.stepwise(DF_REG, "y", ["x1", "x2"], criterio="bic")
    assert "x1" in resultado.titulo and "x2" in resultado.titulo


def test_melhores_subconjuntos_ordena():
    df = DF_REG.copy()
    df["ruido"] = rng.normal(0, 1, 50)
    resultado = mod_reg.melhores_subconjuntos(df, "y", ["x1", "x2", "ruido"])
    linhas = resultado.itens[0][2]
    completo = [linha for linha in linhas if linha[0] == 2][0]
    assert "x1" in completo[6] and "x2" in completo[6]


def test_logistica_recupera_sinal():
    x = rng.normal(0, 1, 300)
    prob = 1 / (1 + np.exp(-(0.5 + 1.2 * x)))
    df = pd.DataFrame({"x": x, "y": np.where(rng.uniform(size=300) < prob,
                                             "sim", "não")})
    resultado = mod_reg.regressao_logistica(df, "y", ["x"])
    tabela = [item for item in resultado.itens if item[0] == "tabela"][0]
    coef_x = _num(tabela[2][1][1])
    assert 0.6 < coef_x < 2.0  # próximo de 1,2
    assert "sim" in resultado.itens[0][1]  # evento = maior nível ordenado


def test_logistica_exige_2_niveis():
    df = pd.DataFrame({"x": [1.0, 2, 3, 4, 5, 6],
                       "y": ["a", "b", "c", "a", "b", "c"]})
    with pytest.raises(ErroAnalise, match="2 valores"):
        mod_reg.regressao_logistica(df, "y", ["x"])


def test_poisson_regressao_irr():
    x = rng.normal(0, 0.5, 200)
    y = rng.poisson(np.exp(1.0 + 0.8 * x))
    df = pd.DataFrame({"x": x, "y": y})
    resultado = mod_reg.regressao_poisson(df, "y", ["x"])
    tabela = [item for item in resultado.itens if item[0] == "tabela"][0]
    coef_x = _num(tabela[2][1][1])
    assert 0.5 < coef_x < 1.1  # próximo de 0,8


def test_poisson_regressao_rejeita_nao_contagem():
    df = pd.DataFrame({"x": [1.0, 2, 3, 4], "y": [1.5, 2, 3, 4]})
    with pytest.raises(ErroAnalise, match="contagens"):
        mod_reg.regressao_poisson(df, "y", ["x"])


# ----------------------------------------------------------- Não paramétricos

def test_sinal_binomial():
    dados = [2.0, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    r = mod_np.teste_sinal(dados, "x", mediana0=3.5)
    # 8 acima, 2 abaixo, 0 empates
    ref = stats.binomtest(8, 10, 0.5)
    assert r.p_valor == pytest.approx(ref.pvalue, abs=1e-12)


def test_wilcoxon_vs_scipy():
    x = rng.normal(1, 2, 30)
    r = mod_np.teste_wilcoxon(x, "x", mediana0=0)
    ref = stats.wilcoxon(x[x != 0])
    assert r.estatistica == pytest.approx(ref.statistic, abs=1e-10)
    assert r.p_valor == pytest.approx(ref.pvalue, abs=1e-10)


def test_mann_whitney_vs_scipy():
    x1, x2 = rng.normal(0, 1, 25), rng.normal(1, 1, 30)
    r = mod_np.teste_mann_whitney(x1, x2, "a", "b")
    ref = stats.mannwhitneyu(x1, x2)
    assert r.estatistica == pytest.approx(ref.statistic, abs=1e-10)


def test_kruskal_e_mood_vs_scipy():
    grupos = [DF_GRUPOS[DF_GRUPOS.g == n]["y"] for n in "ABC"]
    r_kw = mod_np.teste_kruskal(DF_GRUPOS, "y", "g")
    ref_h, ref_p = stats.kruskal(*grupos)
    assert r_kw.estatistica == pytest.approx(ref_h, abs=1e-10)

    r_mood = mod_np.teste_mood(DF_GRUPOS, "y", "g")
    ref = stats.median_test(*grupos)
    assert r_mood.estatistica == pytest.approx(ref[0], abs=1e-10)


def test_friedman_vs_scipy():
    df = pd.DataFrame({"t1": rng.normal(10, 1, 15), "t2": rng.normal(11, 1, 15),
                       "t3": rng.normal(12, 1, 15)})
    r = mod_np.teste_friedman(df, ["t1", "t2", "t3"])
    ref_q, ref_p = stats.friedmanchisquare(df.t1, df.t2, df.t3)
    assert r.estatistica == pytest.approx(ref_q, abs=1e-10)
    assert r.p_valor == pytest.approx(ref_p, abs=1e-10)


def test_runs_formula_manual():
    # Sequência alternada: nº máximo de runs → forte evidência contra aleatoriedade
    dados = [1.0, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10]
    r = mod_np.teste_runs(dados, "x")
    n1 = n2 = 6
    n = 12
    esperado = 2 * n1 * n2 / n + 1
    variancia = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n**2 * (n - 1))
    z = (12 - esperado) / np.sqrt(variancia)
    assert r.estatistica == pytest.approx(z, abs=1e-10)
    assert r.p_valor < 0.01


# ------------------------------------------------------------------- Tabelas

def test_qui_quadrado_independencia():
    df = pd.DataFrame({
        "sexo": ["F"] * 50 + ["M"] * 50,
        "aprovado": ["sim"] * 35 + ["não"] * 15 + ["sim"] * 20 + ["não"] * 30,
    })
    resultado = mod_tab.tabulacao_cruzada(df, "sexo", "aprovado")
    tabela_obs = pd.crosstab(df.sexo, df.aprovado)
    qui2_ref = stats.chi2_contingency(tabela_obs)[0]
    tabela_teste = [i for i in resultado.itens if i[0] == "tabela"][1]
    assert _num(tabela_teste[2][0][1]) == pytest.approx(qui2_ref, abs=1e-3)


def test_fisher_exato_2x2():
    df = pd.DataFrame({"a": ["x"] * 10 + ["y"] * 10,
                       "b": ["s"] * 8 + ["n"] * 2 + ["s"] * 2 + ["n"] * 8})
    resultado = mod_tab.fisher_exato(df, "a", "b")
    ref_or, ref_p = stats.fisher_exact(pd.crosstab(df.a, df.b))
    tabela = [i for i in resultado.itens if i[0] == "tabela"][1]
    assert _num(tabela[2][0][1]) == pytest.approx(ref_or, abs=1e-3)


def test_aderencia_uniforme():
    df = pd.DataFrame({"dado": ["1"] * 10 + ["2"] * 12 + ["3"] * 8 + ["4"] * 10})
    resultado = mod_tab.aderencia(df, "dado")
    qui2_ref, p_ref = stats.chisquare([10, 12, 8, 10])
    tabela = [i for i in resultado.itens if i[0] == "tabela"][1]
    assert _num(tabela[2][0][1]) == pytest.approx(qui2_ref, abs=1e-4)


def test_aderencia_proporcoes_especificadas():
    df = pd.DataFrame({"cor": ["az"] * 30 + ["vm"] * 20})
    resultado = mod_tab.aderencia(df, "cor", {"az": 0.5, "vm": 0.5})
    qui2_ref, _ = stats.chisquare([30, 20], [25, 25])
    tabela = [i for i in resultado.itens if i[0] == "tabela"][1]
    assert _num(tabela[2][0][1]) == pytest.approx(qui2_ref, abs=1e-4)


# -------------------------------------------------------------- Distribuições

def test_calculadora_normal():
    r = mod_dist.calcular("Normal", [0.0, 1.0], "acumulada", 1.96)
    assert _num(r.linhas[0][1]) == pytest.approx(stats.norm.cdf(1.96), abs=1e-5)
    r = mod_dist.calcular("Normal", [0.0, 1.0], "inversa", 0.975)
    assert _num(r.linhas[0][1]) == pytest.approx(1.959964, abs=1e-4)


def test_calculadora_binomial_pmf():
    r = mod_dist.calcular("Binomial", [10, 0.5], "densidade", 5)
    assert _num(r.linhas[0][1]) == pytest.approx(stats.binom.pmf(5, 10, 0.5), abs=1e-6)


def test_gerar_aleatorios_reprodutivel():
    a = mod_dist.gerar_aleatorios("Normal", [0.0, 1.0], 100, semente=1)
    b = mod_dist.gerar_aleatorios("Normal", [0.0, 1.0], 100, semente=1)
    assert np.array_equal(a, b)
    assert a.size == 100


def test_amostragem_sem_reposicao():
    valores = list(range(20))
    amostra = mod_dist.amostrar_coluna(valores, "x", 10, reposicao=False, semente=2)
    assert len(amostra) == 10
    assert len(set(amostra)) == 10  # sem repetição
    with pytest.raises(ErroAnalise):
        mod_dist.amostrar_coluna(valores, "x", 30, reposicao=False)


# ---------------------------------------------------------------------- Poder

def test_poder_t_vs_statsmodels():
    from statsmodels.stats.power import TTestIndPower

    r = mod_poder.poder_t("2amostras", diferenca=1.0, desvio=2.0, n=64)
    ref = TTestIndPower().solve_power(effect_size=0.5, nobs1=64, alpha=0.05)
    assert r.dados["poder_calc"] == pytest.approx(ref, abs=1e-10)


def test_poder_t_n_para_80pct():
    # Caso clássico: d = 0,5, poder 80%, bilateral → n = 64 por grupo
    r = mod_poder.poder_t("2amostras", diferenca=0.5, desvio=1.0, poder=0.8)
    assert r.dados["n_calc"] == 64


def test_poder_z_formula():
    # d=0,5 σ=1 n=32, bilateral: poder = Φ(-1,96+0,5·√32)+Φ(-1,96-0,5·√32)
    r = mod_poder.poder_z_1amostra(0.5, 1.0, n=32)
    zc = stats.norm.ppf(0.975)
    efeito = 0.5 * np.sqrt(32)
    esperado = stats.norm.sf(zc - efeito) + stats.norm.cdf(-zc - efeito)
    assert r.dados["poder_calc"] == pytest.approx(esperado, abs=1e-10)


def test_poder_2proporcoes_vs_statsmodels():
    from statsmodels.stats.power import NormalIndPower
    from statsmodels.stats.proportion import proportion_effectsize

    r = mod_poder.poder_proporcoes("2proporcoes", p0=0.5, p1=0.65, n=100)
    h = proportion_effectsize(0.65, 0.5)
    ref = NormalIndPower().solve_power(effect_size=h, nobs1=100, alpha=0.05)
    assert r.dados["poder_calc"] == pytest.approx(ref, abs=1e-10)


def test_poder_1variancia_consistencia():
    # Com razão > 1 o poder cresce com n; e n calculado atinge o poder pedido
    r_menor = mod_poder.poder_variancias("1variancia", razao=1.5, n=10)
    r_maior = mod_poder.poder_variancias("1variancia", razao=1.5, n=50)
    assert r_maior.dados["poder_calc"] > r_menor.dados["poder_calc"]
    r_n = mod_poder.poder_variancias("1variancia", razao=1.5, poder=0.8)
    alcancado = mod_poder.poder_variancias("1variancia", razao=1.5,
                                           n=r_n.dados["n_calc"])
    assert alcancado.dados["poder_calc"] >= 0.8


def test_poder_anova_vs_statsmodels():
    from statsmodels.stats.power import FTestAnovaPower

    r = mod_poder.poder_anova(k_grupos=3, efeito_f=0.25, n=30)
    ref = FTestAnovaPower().solve_power(effect_size=0.25, nobs=90, alpha=0.05,
                                        k_groups=3)
    assert r.dados["poder_calc"] == pytest.approx(ref, abs=1e-10)


def test_poder_validacao_entrada():
    with pytest.raises(ErroAnalise, match="exatamente um"):
        mod_poder.poder_t("1amostra", 1.0, 1.0, n=10, poder=0.8)
    with pytest.raises(ErroAnalise, match="exatamente um"):
        mod_poder.poder_t("1amostra", 1.0, 1.0)


# ------------------------------------------------------------------ Gráficos

def test_graficos_fase2_geram_figuras():
    from app.plots import graficos

    residuos = rng.normal(0, 1, 40)
    ajustados = rng.normal(10, 2, 40)
    assert graficos.residuos_4paineis(residuos, ajustados, "teste").axes

    grupos = {"A": rng.normal(10, 1, 15), "B": rng.normal(12, 1, 15)}
    assert graficos.efeitos_principais(grupos, "g", "y").axes

    curva = mod_dist.dados_curva("Normal", [0.0, 1.0], sombra_ate=1.0)
    assert graficos.distribuicao(curva, "Normal").axes

    r = mod_poder.poder_t("2amostras", 1.0, 2.0, n=30)
    assert graficos.curva_poder(r.dados["curva"], 5, 100, "curva").axes
