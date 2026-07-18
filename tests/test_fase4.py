"""Validação numérica da Fase 4: DOE, séries, multivariada, confiabilidade
e regressões adicionais."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from app.core import confiabilidade as mod_conf
from app.core import doe as mod_doe
from app.core import multivariada as mod_mv
from app.core import regressao_extra as mod_rx
from app.core import series as mod_se
from app.core.resultados import ErroAnalise

rng = np.random.default_rng(44)


def _num(texto: str) -> float:
    return float(texto.replace("%", "").replace(".", "").replace(",", "."))


# ------------------------------------------------------------------- DOE

def test_gerar_fatorial_completo():
    plano = mod_doe.gerar_fatorial_2k(3, replicas=2, aleatorizar=False)
    assert len(plano) == 16
    assert set(plano["F1"].unique()) == {-1.0, 1.0}
    # plano completo: todas as 8 combinações presentes em cada réplica
    combos = plano.groupby(["F1", "F2", "F3"]).size()
    assert len(combos) == 8 and (combos == 2).all()


def test_gerar_fracao_metade():
    plano = mod_doe.gerar_fatorial_2k(4, fracao=1, aleatorizar=False)
    assert len(plano) == 8  # 2^(4-1)


def test_plackett_burman_ortogonal():
    plano = mod_doe.gerar_plackett_burman(7, aleatorizar=False)
    matriz = plano[[f"F{i + 1}" for i in range(7)]].to_numpy()
    assert len(plano) == 8
    # colunas ortogonais: produto interno nulo
    produto = matriz.T @ matriz
    assert np.allclose(produto - np.diag(np.diag(produto)), 0)


def test_taguchi_l9():
    plano = mod_doe.gerar_taguchi("L9")
    matriz = plano[["F1", "F2", "F3", "F4"]].to_numpy()
    assert matriz.shape == (9, 4)
    for coluna in range(4):
        contagens = np.bincount(matriz[:, coluna])[1:]
        assert (contagens == 3).all()  # cada nível aparece 3 vezes


def test_analise_fatorial_recupera_efeitos():
    plano = mod_doe.gerar_fatorial_2k(3, replicas=2, aleatorizar=False)
    # y = 10 + 3·F1 − 2·F2 + 1,5·F1F2 (efeitos: 6, −4, 3) + ruído pequeno
    plano["resposta"] = (10 + 3 * plano.F1 - 2 * plano.F2
                         + 1.5 * plano.F1 * plano.F2
                         + rng.normal(0, 0.1, len(plano)))
    resultado = mod_doe.analise_fatorial(plano, "resposta", ["F1", "F2", "F3"])
    efeitos = resultado.dados["efeitos"]
    assert efeitos["F1"] == pytest.approx(6, abs=0.3)
    assert efeitos["F2"] == pytest.approx(-4, abs=0.3)
    assert efeitos["F1*F2"] == pytest.approx(3, abs=0.3)
    assert abs(efeitos["F3"]) < 0.3


def test_analise_fatorial_lenth_sem_replicas():
    plano = mod_doe.gerar_fatorial_2k(3, replicas=1, aleatorizar=False)
    plano["resposta"] = 5 + 4 * plano.F1 + rng.normal(0, 0.2, 8)
    # ordem 3 satura o plano (7 termos + constante = 8 corridas) → Lenth
    resultado = mod_doe.analise_fatorial(plano, "resposta", ["F1", "F2", "F3"],
                                         ordem_interacao=3)
    assert resultado.dados["sem_erro"]
    assert resultado.dados["margem_lenth"] is not None
    assert resultado.dados["efeitos"]["F1"] == pytest.approx(8, abs=0.5)


def test_corridas_perdidas():
    plano = mod_doe.gerar_fatorial_2k(3, replicas=2, aleatorizar=False)
    plano["resposta"] = 10 + 3 * plano.F1 + rng.normal(0, 0.1, 16)
    plano.loc[2, "resposta"] = np.nan  # corrida perdida
    resultado = mod_doe.analise_fatorial(plano, "resposta", ["F1", "F2", "F3"])
    assert any("corridas perdidas" in str(item) for item in resultado.itens)
    assert resultado.dados["efeitos"]["F1"] == pytest.approx(6, abs=0.4)


def test_superficie_e_otimizacao():
    plano = mod_doe.gerar_superficie(2, tipo="ccd", aleatorizar=False)
    # máximo em (0,5; −0,5): y = 20 − (x1−0,5)² − (x2+0,5)²
    plano["resposta"] = (20 - (plano.F1 - 0.5) ** 2 - (plano.F2 + 0.5) ** 2
                         + rng.normal(0, 0.05, len(plano)))
    resultado = mod_doe.analise_superficie(plano, "resposta", ["F1", "F2"])
    otimo = mod_doe.otimizar_resposta(resultado.dados, "maximizar")
    ajustes = {linha[0]: _num(linha[1]) for linha in otimo.itens[0][2]}
    assert ajustes["F1"] == pytest.approx(0.5, abs=0.15)
    assert ajustes["F2"] == pytest.approx(-0.5, abs=0.15)


def test_prever_modelo_quadratico():
    plano = mod_doe.gerar_superficie(2, tipo="ccd", aleatorizar=False)
    plano["resposta"] = 5 + 2 * plano.F1 + plano.F1**2
    resultado = mod_doe.analise_superficie(plano, "resposta", ["F1", "F2"])
    previsto = mod_doe.prever_modelo(resultado.dados, {"F1": 1.0, "F2": 0.0})
    assert previsto == pytest.approx(8, abs=0.1)


def test_taguchi_analise_sn():
    plano = mod_doe.gerar_taguchi("L4")
    # F1 no nível 2 dobra a resposta (maior-melhor deve preferir nível 2)
    base = np.where(plano.F1 == 2, 20.0, 10.0)
    plano["resposta_1"] = base + rng.normal(0, 0.5, 4)
    plano["resposta_2"] = base + rng.normal(0, 0.5, 4)
    resultado = mod_doe.analise_taguchi(plano, ["F1", "F2", "F3"],
                                        ["resposta_1", "resposta_2"],
                                        "maior-melhor")
    assert "F1 = 2" in resultado.itens[-1][1]


# ------------------------------------------------------------------ séries

def test_tendencia_linear_recupera_coeficientes():
    y = 5 + 2 * np.arange(1, 31) + rng.normal(0, 0.5, 30)
    resultado = mod_se.tendencia(y, "y", "linear", horizonte=3)
    previsoes = [(int(linha[0]), _num(linha[1]))
                 for linha in resultado.itens[3][2]]
    for t, previsto in previsoes:
        assert previsto == pytest.approx(5 + 2 * t, abs=1.5)


def test_decomposicao_recupera_sazonalidade():
    t = np.arange(48)
    sazonal = np.tile([5.0, 0, -5, 0], 12)
    y = 10 + 0.2 * t + sazonal + rng.normal(0, 0.3, 48)
    resultado = mod_se.decomposicao(y, "y", periodo=4)
    indices = [_num(linha[1]) for linha in resultado.itens[1][2]]
    assert max(indices) == pytest.approx(5, abs=1)
    assert min(indices) == pytest.approx(-5, abs=1)


def test_suavizacao_ses_e_winters():
    y = 10 + np.tile([3.0, 0, -3, 0], 10) + rng.normal(0, 0.3, 40)
    r_ses = mod_se.suavizacao(y, "y", "ses", parametro=0.3)
    assert r_ses.dados["previsao"].size == 6
    r_w = mod_se.suavizacao(y, "y", "winters", periodo=4)
    # Winters deve prever o padrão sazonal: 1ª previsão ≈ 13 (pico)
    assert r_w.dados["previsao"][0] == pytest.approx(13, abs=1.5)


def test_acf_vs_statsmodels():
    from statsmodels.tsa.stattools import acf

    y = rng.normal(0, 1, 100).cumsum()  # série com forte autocorrelação
    resultado = mod_se.autocorrelacao(y, "y", defasagens=10)
    ref = acf(y, nlags=10)
    assert resultado.dados["acf"][1] == pytest.approx(ref[1], abs=1e-10)
    assert "significativa" in resultado.itens[-1][1]


def test_arima_ajusta_e_preve():
    # AR(1) com φ = 0,7
    y = np.empty(200)
    y[0] = 0
    for i in range(1, 200):
        y[i] = 0.7 * y[i - 1] + rng.normal(0, 1)
    resultado = mod_se.arima(y, "y", p=1, d=0, q=0, horizonte=5)
    linha_ar = [linha for linha in resultado.itens[1][2]
                if "ar" in linha[0].lower()][0]
    assert _num(linha_ar[1]) == pytest.approx(0.7, abs=0.15)
    assert resultado.dados["previsao"].size == 5


# ------------------------------------------------------------- multivariada

def test_pca_variancia_total():
    df = pd.DataFrame({"a": rng.normal(0, 1, 100)})
    df["b"] = df.a * 0.9 + rng.normal(0, 0.3, 100)
    df["c"] = rng.normal(0, 1, 100)
    resultado = mod_mv.pca(df, ["a", "b", "c"])
    autovalores = resultado.dados["autovalores"]
    assert autovalores.sum() == pytest.approx(3, abs=1e-8)  # traço da correlação
    assert autovalores[0] > 1.5  # a+b correlacionados dominam


def test_discriminante_separa_grupos():
    df = pd.DataFrame({
        "x1": np.concatenate([rng.normal(0, 1, 40), rng.normal(4, 1, 40)]),
        "x2": np.concatenate([rng.normal(0, 1, 40), rng.normal(4, 1, 40)]),
        "grupo": ["A"] * 40 + ["B"] * 40,
    })
    resultado = mod_mv.discriminante(df, "grupo", ["x1", "x2"])
    acerto = _num(resultado.itens[3][2][0][1])
    assert acerto > 95


def test_cluster_kmeans_recupera_grupos():
    df = pd.DataFrame({
        "x": np.concatenate([rng.normal(0, 0.5, 30), rng.normal(5, 0.5, 30)]),
        "y": np.concatenate([rng.normal(0, 0.5, 30), rng.normal(5, 0.5, 30)]),
    })
    resultado = mod_mv.cluster(df, ["x", "y"], k=2, metodo="kmeans")
    rotulos = resultado.dados["rotulos"].dropna().astype(int).to_numpy()
    # os 30 primeiros devem ficar juntos, os 30 últimos também
    assert len(set(rotulos[:30])) == 1 and len(set(rotulos[30:])) == 1
    assert rotulos[0] != rotulos[-1]


def test_cluster_hierarquico_dendrograma():
    df = pd.DataFrame({"x": rng.normal(0, 1, 40), "y": rng.normal(0, 1, 40)})
    resultado = mod_mv.cluster(df, ["x", "y"], k=3, metodo="hierarquico")
    assert resultado.dados["ligacao"] is not None
    assert resultado.dados["rotulos"].dropna().nunique() == 3


def test_correspondencia_inercia_igual_qui2():
    df = pd.DataFrame({
        "linha": ["a"] * 60 + ["b"] * 40,
        "coluna": ["x"] * 40 + ["y"] * 20 + ["x"] * 10 + ["y"] * 30,
    })
    resultado = mod_mv.correspondencia(df, "linha", "coluna")
    tabela = pd.crosstab(df.linha, df.coluna)
    qui2 = stats.chi2_contingency(tabela, correction=False)[0]
    inercia_total = float(sum(_num(linha[1]) for linha in resultado.itens[0][2]))
    assert inercia_total == pytest.approx(qui2 / 100, abs=1e-3)


def test_cronbach_formula_manual():
    df = pd.DataFrame(rng.normal(0, 1, (50, 1)) + rng.normal(0, 0.5, (50, 4)),
                      columns=["i1", "i2", "i3", "i4"])
    resultado = mod_mv.cronbach(df, ["i1", "i2", "i3", "i4"])
    matriz = df.to_numpy()
    k = 4
    alfa_manual = k / (k - 1) * (1 - matriz.var(axis=0, ddof=1).sum()
                                 / matriz.sum(axis=1).var(ddof=1))
    assert _num(resultado.itens[0][2][0][1]) == pytest.approx(alfa_manual,
                                                              abs=1e-3)


# ----------------------------------------------------------- confiabilidade

def test_kaplan_meier_sem_censura():
    tempos = np.array([2.0, 4, 6, 8, 10])
    resultado = mod_conf.kaplan_meier(pd.DataFrame({"t": tempos}), "t")
    km = resultado.dados["km"]
    # sem censura, S(t) é a fração sobrevivente: S(4) = 3/5
    assert float(km.survival_function_at_times(4.0).iloc[0]) == pytest.approx(0.6)


def test_kaplan_meier_com_censura():
    df = pd.DataFrame({"t": [3.0, 5, 5, 7, 9, 11],
                       "evento": [1, 1, 0, 1, 0, 1]})
    resultado = mod_conf.kaplan_meier(df, "t", "evento")
    km = resultado.dados["km"]
    # cálculo manual: S(3)=5/6; S(5)=5/6·(4/5)=2/3; S(7)=2/3·(2/3)=4/9
    assert float(km.survival_function_at_times(7.0).iloc[0]) == pytest.approx(
        4 / 9, abs=1e-9)


def test_weibull_recupera_parametros():
    verdadeiro_forma, verdadeira_escala = 2.0, 100.0
    tempos = verdadeira_escala * rng.weibull(verdadeiro_forma, 300)
    resultado = mod_conf.analise_parametrica(pd.DataFrame({"t": tempos}), "t",
                                             familia="Weibull")
    ajuste = resultado.dados["ajuste"]
    assert float(ajuste.rho_) == pytest.approx(verdadeiro_forma, rel=0.15)
    assert float(ajuste.lambda_) == pytest.approx(verdadeira_escala, rel=0.1)


def test_weibull_censura_direita():
    tempos = 50 * rng.weibull(1.5, 200)
    corte = np.quantile(tempos, 0.7)
    observado = np.minimum(tempos, corte)
    evento = (tempos <= corte).astype(float)
    df = pd.DataFrame({"t": observado, "evento": evento})
    resultado = mod_conf.analise_parametrica(df, "t", "evento",
                                             familia="Weibull")
    ajuste = resultado.dados["ajuste"]
    assert float(ajuste.rho_) == pytest.approx(1.5, rel=0.25)


# --------------------------------------------------- regressões adicionais

def test_nao_linear_exponencial():
    x = np.linspace(0, 5, 60)
    y = 2.0 * np.exp(0.5 * x) + rng.normal(0, 0.2, 60)
    df = pd.DataFrame({"x": x, "y": y})
    resultado = mod_rx.regressao_nao_linear(df, "y", "x",
                                            "Exponencial: a·exp(b·x)")
    parametros = resultado.dados["parametros"]
    assert parametros[0] == pytest.approx(2.0, abs=0.3)
    assert parametros[1] == pytest.approx(0.5, abs=0.05)


def test_logistica_ordinal_sinal():
    x = rng.normal(0, 1, 400)
    latente = 1.5 * x + rng.logistic(0, 1, 400)
    y = pd.cut(latente, [-np.inf, -1, 1, np.inf],
               labels=["baixo", "médio", "alto"]).astype(str)
    df = pd.DataFrame({"x": x, "y": y})
    resultado = mod_rx.logistica_ordinal(df, "y", ["x"],
                                         ordem=["baixo", "médio", "alto"])
    coef = _num(resultado.itens[1][2][0][1])
    assert coef == pytest.approx(1.5, abs=0.4)


def test_logistica_nominal_3_categorias():
    x = rng.normal(0, 1, 300)
    y = np.where(x < -0.5, "a", np.where(x > 0.5, "c", "b"))
    df = pd.DataFrame({"x": x, "y": y})
    resultado = mod_rx.logistica_nominal(df, "y", ["x"])
    assert sum(1 for item in resultado.itens if item[0] == "subtitulo") == 2


def test_pls_vs_sklearn():
    from sklearn.cross_decomposition import PLSRegression

    df = pd.DataFrame(rng.normal(0, 1, (60, 3)), columns=["a", "b", "c"])
    df["y"] = 2 * df.a + df.b + rng.normal(0, 0.3, 60)
    resultado = mod_rx.pls(df, "y", ["a", "b", "c"], n_componentes=2)
    ref = PLSRegression(n_components=2).fit(df[["a", "b", "c"]], df["y"])
    r2_ref = ref.score(df[["a", "b", "c"]], df["y"])
    assert 1 - float(np.var(resultado.dados["residuos"])
                     / np.var(resultado.dados["y"])) == pytest.approx(r2_ref,
                                                                      abs=0.01)


def test_ortogonal_deming_slope():
    x_verdadeiro = rng.normal(10, 3, 200)
    x = x_verdadeiro + rng.normal(0, 0.5, 200)
    y = 2 * x_verdadeiro + rng.normal(0, 0.5, 200)
    df = pd.DataFrame({"x": x, "y": y})
    resultado = mod_rx.regressao_ortogonal(df, "y", "x", razao_variancias=4.0)
    assert resultado.dados["inclinacao"] == pytest.approx(2.0, abs=0.1)


def test_equivalencia_tost():
    from statsmodels.stats.weightstats import ttost_ind

    r_local = np.random.default_rng(44)  # independente da ordem dos testes
    x1 = r_local.normal(10, 1, 50)
    x2 = r_local.normal(10.1, 1, 50)
    r = mod_rx.equivalencia(x1, "a", -0.5, 0.5, dados2=x2, col2="b")
    p_ref, _, _ = ttost_ind(x1, x2, -0.5, 0.5)
    assert r.p_valor == pytest.approx(p_ref, abs=1e-10)
    assert r.p_valor < 0.05  # equivalentes dentro de ±0,5


def test_bootstrap_ic_cobre_media():
    x = rng.normal(50, 5, 100)
    resultado = mod_rx.bootstrap_ic(x, "x", "média", semente=7)
    tabela = resultado.itens[0][2]
    ic = tabela[1][1].strip("()").split(";")
    assert _num(ic[0]) < 50 < _num(ic[1])


def test_aleatorizacao_detecta_diferenca():
    x1 = rng.normal(10, 1, 40)
    x2 = rng.normal(12, 1, 40)
    r = mod_rx.teste_aleatorizacao(x1, x2, "a", "b", semente=3)
    assert r.p_valor < 0.01


def test_arvore_classificacao_e_regressao():
    df = pd.DataFrame({"x1": rng.normal(0, 1, 100),
                       "x2": rng.normal(0, 1, 100)})
    df["classe"] = np.where(df.x1 > 0, "sim", "não")
    r_cls = mod_rx.arvore(df, "classe", ["x1", "x2"], profundidade=2)
    assert r_cls.dados["classificacao"]
    importancias = dict((linha[0], _num(linha[1]))
                        for linha in r_cls.itens[2][2])
    assert importancias.get("x1", 0) > 90  # x1 domina

    df["alvo"] = 3 * df.x1 + rng.normal(0, 0.2, 100)
    r_reg = mod_rx.arvore(df, "alvo", ["x1", "x2"], profundidade=3)
    assert not r_reg.dados["classificacao"]


# ------------------------------------------------------------------ gráficos

def test_graficos_fase4_geram_figuras():
    from app.plots import fase4_plots as f4

    plano = mod_doe.gerar_fatorial_2k(3, replicas=2, aleatorizar=False)
    plano["resposta"] = 10 + 3 * plano.F1 + rng.normal(0, 0.2, 16)
    resultado = mod_doe.analise_fatorial(plano, "resposta", ["F1", "F2", "F3"])
    assert f4.pareto_efeitos(resultado.dados["efeitos"], None, "t").axes
    assert f4.normal_efeitos(resultado.dados["efeitos"], None, "t").axes
    assert f4.cubo(resultado.dados).axes
    assert f4.contorno_superficie(resultado.dados, "F1", "F2").axes
    assert f4.contorno_superficie(resultado.dados, "F1", "F2", em_3d=True).axes

    r_tend = mod_se.tendencia(rng.normal(10, 1, 30), "y", "linear")
    assert f4.serie_previsao(r_tend.dados, "t").axes

    df = pd.DataFrame({"a": rng.normal(0, 1, 50), "b": rng.normal(0, 1, 50),
                       "c": rng.normal(0, 1, 50)})
    r_pca = mod_mv.pca(df, ["a", "b", "c"])
    assert f4.scree_escores(r_pca.dados).axes

    r_km = mod_conf.kaplan_meier(
        pd.DataFrame({"t": 50 * rng.weibull(1.5, 30)}), "t")
    assert f4.sobrevivencia(r_km.dados).axes
    assert f4.probabilidade_weibull(r_km.dados).axes
