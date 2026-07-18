"""Validação numérica da Fase 3: CEP, capabilidade, qualidade e MSA."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from app.core import capabilidade as mod_cap
from app.core import cep as mod_cep
from app.core import msa as mod_msa
from app.core import qualidade as mod_qual
from app.core.resultados import ErroAnalise

rng = np.random.default_rng(2026)


def _num(texto: str) -> float:
    return float(texto.replace("%", "").replace(".", "").replace(",", "."))


# ----------------------------------------------------------------- CEP

def test_c4_valores_tabelados():
    # valores clássicos: c4(2)=0,7979; c4(5)=0,9400; c4(10)=0,9727
    assert mod_cep.c4(2) == pytest.approx(0.7979, abs=1e-4)
    assert mod_cep.c4(5) == pytest.approx(0.9400, abs=1e-4)
    assert mod_cep.c4(10) == pytest.approx(0.9727, abs=1e-4)


def test_carta_i_mr_limites_manuais():
    x = np.array([10.0, 12, 11, 13, 12, 11, 10, 12, 13, 11])
    cartas, _ = mod_cep.carta_i_mr(x, "x")
    carta_i, carta_mr = cartas
    mr = np.abs(np.diff(x))
    mr_barra = mr.mean()
    assert carta_i.lc[0] == pytest.approx(x.mean(), abs=1e-10)
    assert carta_i.lsc[0] == pytest.approx(x.mean() + 3 * mr_barra / 1.128, abs=1e-4)
    assert carta_mr.lsc[0] == pytest.approx(3.267 * mr_barra, abs=1e-4)


def test_carta_xbar_r_constantes():
    # limites Xbarra: media ± A2·R-barra, com A2 = 3/(d2·√n)
    x = rng.normal(50, 2, 100)
    cartas, _ = mod_cep.carta_xbar_r(x, "x", tamanho=5)
    carta_m = cartas[0]
    matriz = x[:100].reshape(-1, 5)
    a2 = 3 / (2.326 * np.sqrt(5))  # d2(5) = 2,326 → A2 = 0,577
    esperado = matriz.mean() + a2 * np.ptp(matriz, axis=1).mean()
    assert a2 == pytest.approx(0.577, abs=1e-3)
    assert carta_m.lsc[0] == pytest.approx(esperado, abs=1e-6)


def test_carta_xbar_s_constantes():
    x = rng.normal(50, 2, 120)
    cartas, _ = mod_cep.carta_xbar_s(x, "x", tamanho=12)  # >10 exige S
    carta_m, carta_s = cartas
    matriz = x.reshape(-1, 12)
    c = mod_cep.c4(12)
    s_barra = matriz.std(axis=1, ddof=1).mean()
    assert carta_m.lsc[0] == pytest.approx(
        matriz.mean() + 3 * (s_barra / c) / np.sqrt(12), abs=1e-6)
    assert carta_s.lsc[0] == pytest.approx(
        s_barra * (1 + 3 * np.sqrt(1 - c**2) / c), abs=1e-6)


def test_carta_xbar_r_recusa_subgrupo_grande():
    with pytest.raises(ErroAnalise, match="Xbarra-S"):
        mod_cep.carta_xbar_r(rng.normal(0, 1, 100), "x", tamanho=12)


def test_carta_p_limites():
    d = np.array([4.0, 6, 5, 3, 7, 5, 4, 6])
    cartas, _ = mod_cep.carta_p(d, "def", tamanhos=100)
    carta = cartas[0]
    pbar = d.sum() / 800
    assert carta.lc[0] == pytest.approx(pbar, abs=1e-12)
    assert carta.lsc[0] == pytest.approx(
        pbar + 3 * np.sqrt(pbar * (1 - pbar) / 100), abs=1e-10)


def test_carta_c_e_u():
    c = np.array([3.0, 5, 4, 6, 2, 4, 5, 3])
    cartas, _ = mod_cep.carta_c(c, "defeitos")
    cbar = c.mean()
    assert cartas[0].lsc[0] == pytest.approx(cbar + 3 * np.sqrt(cbar), abs=1e-10)

    tamanhos = np.array([10.0, 12, 10, 11, 10, 12, 10, 11])
    cartas_u, _ = mod_cep.carta_c(c, "defeitos", tamanhos=tamanhos)
    ubar = c.sum() / tamanhos.sum()
    assert cartas_u[0].lc[0] == pytest.approx(ubar, abs=1e-12)
    assert cartas_u[0].lsc[0] == pytest.approx(
        ubar + 3 * np.sqrt(ubar / 10), abs=1e-10)


def test_ewma_formula():
    x = rng.normal(20, 1, 30)
    lam = 0.2
    cartas, _ = mod_cep.carta_ewma(x, "x", lam=lam)
    carta = cartas[0]
    z = x[0] * lam + (1 - lam) * x.mean()
    assert carta.pontos[0] == pytest.approx(z, abs=1e-10)
    sigma = np.abs(np.diff(x)).mean() / 1.128
    dist1 = 3 * sigma * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** 2))
    assert carta.lsc[0] - carta.lc[0] == pytest.approx(dist1, abs=1e-10)


def test_cusum_detecta_deslocamento():
    x = np.concatenate([rng.normal(10, 1, 20), rng.normal(11.5, 1, 15)])
    cartas, _ = mod_cep.carta_cusum(x, "x", alvo=10.0)
    assert len(cartas[0].indices_violacao()) > 0  # deslocamento detectado


def test_testes_causas_especiais():
    # 9 pontos do mesmo lado (teste 2) sem estourar 3σ
    pontos = np.array([10.0, 9.8, 10.1, 10.4, 10.4, 10.3, 10.4, 10.2, 10.3,
                       10.4, 10.2, 10.3])
    lc = np.full(12, 10.0)
    violacoes = mod_cep.aplicar_testes(pontos, lc, lc + 3, lc - 3, {1, 2})
    assert 2 in violacoes and 1 not in violacoes
    # tendência de 6 pontos (teste 3)
    pontos_t = np.array([1.0, 2, 3, 4, 5, 6, 7])
    lc7 = np.full(7, 4.0)
    v3 = mod_cep.aplicar_testes(pontos_t, lc7, lc7 + 10, lc7 - 10, {3})
    assert 3 in v3


def test_estagios_limites_por_segmento():
    x = np.concatenate([rng.normal(10, 1, 15), rng.normal(15, 1, 15)])
    estagios = ["antes"] * 15 + ["depois"] * 15
    cartas, _ = mod_cep.carta_i_mr(x, "x", estagios=estagios)
    carta_i = cartas[0]
    assert carta_i.lc[0] == pytest.approx(x[:15].mean(), abs=1e-10)
    assert carta_i.lc[-1] == pytest.approx(x[15:].mean(), abs=1e-10)
    assert carta_i.lc[0] != pytest.approx(carta_i.lc[-1], abs=0.5)


# ---------------------------------------------------------- Capabilidade

def test_capabilidade_indices_manuais():
    x = rng.normal(100, 2, 200)
    resultado = mod_cap.capabilidade_normal(x, "x", lie=94, lse=106)
    sigma_d = np.abs(np.diff(x)).mean() / 1.128
    cp_esperado = 12 / (6 * sigma_d)
    tabela_indices = [i for i in resultado.itens if i[0] == "tabela"][1]
    assert _num(tabela_indices[2][0][1]) == pytest.approx(cp_esperado, abs=5e-3)
    # Cpk = min(CPU, CPL)
    cpu = (106 - x.mean()) / (3 * sigma_d)
    cpl = (x.mean() - 94) / (3 * sigma_d)
    assert _num(tabela_indices[2][3][1]) == pytest.approx(min(cpu, cpl), abs=5e-3)


def test_capabilidade_ppm():
    x = rng.normal(10, 1, 500)
    resultado = mod_cap.capabilidade_normal(x, "x", lie=None, lse=11)
    ppm_esp = 1e6 * stats.norm.sf(11, x.mean(),
                                  np.abs(np.diff(x)).mean() / 1.128)
    tabela_ppm = [i for i in resultado.itens if i[0] == "tabela"][2]
    assert _num(tabela_ppm[2][1][1]) == pytest.approx(ppm_esp, rel=0.02)


def test_capabilidade_boxcox():
    x = rng.lognormal(1, 0.4, 300)
    resultado = mod_cap.capabilidade_normal(x, "x", lie=0.5, lse=12,
                                            transformacao="boxcox")
    assert any("Box-Cox" in str(item) for item in resultado.itens)
    # dados transformados devem ser ~normais
    _, p = mod_cap._ad_normal(resultado.dados["x"])
    assert p > 0.01


def test_capabilidade_johnson():
    x = rng.lognormal(1, 0.5, 300)
    resultado = mod_cap.capabilidade_normal(x, "x", lie=0.4, lse=15,
                                            transformacao="johnson")
    _, p = mod_cap._ad_normal(resultado.dados["x"])
    assert p > 0.01


def test_identificacao_distribuicao():
    x = rng.weibull(2.0, 400) * 10
    resultado = mod_cap.identificar_distribuicao(x, "x")
    tabela = resultado.itens[0][2]
    melhores3 = [linha[0] for linha in tabela[:3]]
    assert "Weibull" in melhores3  # o melhor ajuste deve estar no topo


def test_capabilidade_atributos():
    resultado = mod_cap.capabilidade_atributos(25, 1000)
    tabela = resultado.itens[0][2]
    assert _num(tabela[2][1]) == pytest.approx(2.5, abs=1e-6)   # % defeituosa
    assert _num(tabela[3][1]) == pytest.approx(25000, abs=1)    # PPM


# ------------------------------------------------------------- Qualidade

def test_pareto_acumulado():
    dados = ["risco"] * 50 + ["mancha"] * 30 + ["trinca"] * 15 + ["outro"] * 5
    resultado = mod_qual.pareto_resumo(dados, "defeito")
    linhas = resultado.itens[0][2]
    assert linhas[0][0] == "risco"
    assert _num(linhas[0][3]) == pytest.approx(50.0, abs=0.1)
    assert _num(linhas[-1][3]) == pytest.approx(100.0, abs=0.1)


def test_tolerancia_normal_k_razoavel():
    x = rng.normal(50, 5, 50)
    resultado = mod_qual.intervalo_tolerancia(x, "x", 0.95, 0.95)
    tabela_k = [i for i in resultado.itens if i[0] == "tabela"][1]
    k = _num(tabela_k[2][0][1])
    # k para n=50, 95/95 bilateral ≈ 2,38 (Howe); e sempre > z_0,975
    assert 2.2 < k < 2.6
    assert k > 1.96


def test_tolerancia_k_converge_para_z():
    x = rng.normal(0, 1, 100_000)
    resultado = mod_qual.intervalo_tolerancia(x, "x", 0.95, 0.95)
    tabela_k = [i for i in resultado.itens if i[0] == "tabela"][1]
    assert _num(tabela_k[2][0][1]) == pytest.approx(1.96, abs=0.02)


def test_tolerancia_nao_parametrica_regra_93():
    # clássico: n = 93 atinge 95% de confiança para 95% de cobertura (mín/máx)
    conf = lambda n: 1 - n * 0.95 ** (n - 1) + (n - 1) * 0.95**n
    assert conf(93) >= 0.95 > conf(92)
    x = rng.normal(0, 1, 93)
    resultado = mod_qual.intervalo_tolerancia(x, "x", 0.95, 0.95)
    tabela_np = [i for i in resultado.itens if i[0] == "tabela"][2]
    assert _num(tabela_np[2][1][1]) == pytest.approx(100 * conf(93), abs=0.1)


# ------------------------------------------------------------------- MSA

def _dados_grr(sigma_oper=0.5, sigma_rep=0.3, sigma_peca=2.0, semente=5):
    r = np.random.default_rng(semente)
    pecas = r.normal(0, sigma_peca, 10)
    operadores = r.normal(0, sigma_oper, 3)
    linhas = []
    for i, vp in enumerate(pecas):
        for j, vo in enumerate(operadores):
            for _ in range(3):
                linhas.append({"peça": f"P{i + 1}", "operador": f"O{j + 1}",
                               "medição": 50 + vp + vo + r.normal(0, sigma_rep)})
    return pd.DataFrame(linhas)


def test_gage_rr_cruzado_componentes():
    df = _dados_grr()
    resultado = mod_msa.gage_rr_cruzado(df, "medição", "peça", "operador")
    # repetitividade estimada ≈ σ_rep² = 0,09
    tabela_var = [i for i in resultado.itens if i[0] == "tabela"][1]
    var_rep = _num(tabela_var[2][1][1])
    assert 0.04 < var_rep < 0.2
    # peça-a-peça deve dominar (%contribuição > 80%)
    pct_peca = _num(tabela_var[2][3][2])
    assert pct_peca > 70
    assert resultado.dados["ndc"] >= 4


def test_gage_rr_cruzado_soma_contribuicoes():
    df = _dados_grr()
    resultado = mod_msa.gage_rr_cruzado(df, "medição", "peça", "operador")
    tabela_var = [i for i in resultado.itens if i[0] == "tabela"][1]
    # GRR% + peça% = 100%
    pct_grr = _num(tabela_var[2][0][2])
    pct_peca = _num(tabela_var[2][3][2])
    assert pct_grr + pct_peca == pytest.approx(100.0, abs=0.1)


def test_gage_rr_exige_balanceado():
    df = _dados_grr().iloc[:-1]  # remove uma medição
    with pytest.raises(ErroAnalise, match="balanceado"):
        mod_msa.gage_rr_cruzado(df, "medição", "peça", "operador")


def test_gage_rr_aninhado():
    r = np.random.default_rng(9)
    linhas = []
    for j in range(3):  # 3 operadores, 5 peças exclusivas cada
        for i in range(5):
            valor_peca = r.normal(0, 2)
            for _ in range(2):
                linhas.append({"peça": f"O{j}P{i}", "operador": f"O{j + 1}",
                               "medição": 30 + valor_peca + r.normal(0, 0.3)})
    df = pd.DataFrame(linhas)
    resultado = mod_msa.gage_rr_aninhado(df, "medição", "peça", "operador")
    assert resultado.dados["pct_grr"] < 40


def test_linearidade_vies_detecta_inclinacao():
    r = np.random.default_rng(11)
    referencias = np.repeat([2.0, 4, 6, 8, 10], 12)
    medicoes = referencias + 0.05 * referencias + r.normal(0, 0.05, 60)
    df = pd.DataFrame({"ref": referencias, "med": medicoes})
    resultado = mod_msa.linearidade_vies(df, "med", "ref")
    tabela_lin = [i for i in resultado.itens if i[0] == "tabela"][1]
    inclinacao = _num(tabela_lin[2][0][1])
    assert inclinacao == pytest.approx(0.05, abs=0.02)
    assert "problema de linearidade" in resultado.itens[-1][1]


def test_estudo_tipo1_cg():
    r = np.random.default_rng(13)
    x = r.normal(10.02, 0.01, 30)
    resultado = mod_msa.estudo_tipo1(x, "med", referencia=10.0, tolerancia=1.0)
    s = x.std(ddof=1)
    cg_esperado = (0.2 * 1.0) / (6 * s)
    tabela = resultado.itens[0][2]
    cg = _num([linha for linha in tabela if linha[0] == "Cg"][0][1])
    assert cg == pytest.approx(cg_esperado, abs=0.05)


def test_concordancia_atributos_kappa():
    # 2 avaliadores × 20 peças × 2 tentativas, alta concordância com padrão
    r = np.random.default_rng(17)
    padrao = r.choice(["ok", "defeito"], 20)
    linhas = []
    for i, verdadeiro in enumerate(padrao):
        for aval in ("A", "B"):
            for _ in range(2):
                resposta = verdadeiro if r.uniform() < 0.9 else (
                    "ok" if verdadeiro == "defeito" else "defeito")
                linhas.append({"peça": f"P{i}", "avaliador": aval,
                               "resultado": resposta, "padrão": verdadeiro})
    df = pd.DataFrame(linhas)
    resultado = mod_msa.concordancia_atributos(df, "peça", "avaliador",
                                               "resultado", "padrão")
    assert any("kappa" in str(item).lower() for item in resultado.itens)


def test_kappa_cohen_perfeito_e_aleatorio():
    a = pd.Series(["x", "y"] * 20)
    assert mod_msa._kappa_cohen(a, a) == pytest.approx(1.0)
    b = pd.Series(["x"] * 20 + ["y"] * 20)
    c = pd.Series((["x", "y"] * 10) * 2)
    assert abs(mod_msa._kappa_cohen(b, c)) < 0.3  # próximo do acaso


def test_planilha_coleta_grr():
    tabela = mod_msa.planilha_coleta_grr(5, 3, 2, semente=1)
    assert len(tabela) == 30
    assert set(tabela["peça"].unique()) == {f"P{i}" for i in range(1, 6)}
    contagem = tabela.groupby(["peça", "operador"]).size()
    assert (contagem == 2).all()  # balanceada


# ------------------------------------------------------------- Gráficos

def test_graficos_fase3_geram_figuras():
    from app.plots import qualidade_plots as qp

    x = rng.normal(10, 1, 40)
    cartas, _ = mod_cep.carta_i_mr(x, "x")
    assert qp.carta_controle(cartas, "I-MR").axes

    contagens = mod_qual.pareto_contagens(["a"] * 5 + ["b"] * 3, "defeito")
    assert qp.pareto(contagens, "defeito").axes

    assert qp.ishikawa("Peça fora de especificação",
                       {"Máquina": ["desgaste"], "Método": ["setup"]}).axes

    df = _dados_grr()
    dados = mod_msa._dados_msa(df, "medição", "peça", "operador")
    assert qp.gage_run(dados, "medição").axes

    resultado = mod_cap.capabilidade_normal(rng.normal(50, 2, 100), "x",
                                            lie=44, lse=56)
    assert qp.capabilidade_histograma(resultado.dados).axes
    assert len(qp.relatorio_capabilidade(resultado.dados).axes) == 6

    dfm = pd.DataFrame({"y": rng.normal(10, 1, 24),
                        "t": np.tile(["t1", "t2"], 12),
                        "m": np.repeat(["m1", "m2"], 12)})
    dados_mv = mod_qual.multivari_dados(dfm, "y", ["t", "m"])
    assert qp.multivari(dados_mv, "y", ["t", "m"]).axes
