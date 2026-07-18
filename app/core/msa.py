"""Análise de sistemas de medição (MSA).

- Gage R&R cruzado: método ANOVA de 2 fatores (peça × operador), com a
  interação removida quando p > 0,25 (procedimento padrão AIAG). Componentes
  de variância, %Contribuição, %Variação do Estudo e ndc.
- Gage R&R aninhado: peças distintas por operador (peça aninhada em operador).
- Linearidade e viés, estudo Tipo 1 (Cg/Cgk) e concordância por atributos
  (kappa de Fleiss e de Cohen).

Planos balanceados são exigidos nas análises de Gage R&R (Fase 3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from .resultados import ErroAnalise, ResultadoComposto

K_ESTUDO = 6.0  # nº de desvios-padrão da variação do estudo (padrão AIAG)


def _dados_msa(df: pd.DataFrame, medicao: str, peca: str,
               operador: str) -> pd.DataFrame:
    dados = pd.DataFrame({
        "y": pd.to_numeric(df[medicao], errors="coerce"),
        "p": df[peca].astype(object),
        "o": df[operador].astype(object),
    })
    for c in ("p", "o"):
        vazio = dados[c].isna() | (dados[c].astype(str).str.strip() == "")
        dados[c] = dados[c].astype(str).where(~vazio, np.nan)
    dados = dados.dropna()
    if len(dados) < 8:
        raise ErroAnalise("Dados insuficientes para o estudo de MSA.")
    return dados


def _exigir_balanceado(dados: pd.DataFrame, grupos: list[str]) -> int:
    contagem = dados.groupby(grupos, observed=True).size()
    if contagem.nunique() != 1:
        raise ErroAnalise(
            "O plano precisa ser balanceado (mesmo número de medições por "
            "combinação). Verifique dados faltantes ou repetições desiguais.")
    return int(contagem.iloc[0])


def _tabela_grr(componentes: dict[str, float], sigma_total: float,
                variacao_processo: float | None) -> list[tuple]:
    from app.reports.formatacao import fmt

    var_total = sum(componentes.values())
    linhas_contrib = []
    linhas_estudo = []
    grr = componentes.get("repetitividade", 0) + componentes.get(
        "reprodutibilidade", 0)
    ordem = [("Gage R&R total", grr),
             ("  repetitividade (equipamento)", componentes.get("repetitividade", 0)),
             ("  reprodutibilidade (operadores)",
              componentes.get("reprodutibilidade", 0)),
             ("peça a peça", componentes.get("peca", 0)),
             ("variação total", var_total)]
    base_estudo = variacao_processo or np.sqrt(var_total)
    for nome, var in ordem:
        sigma = np.sqrt(max(var, 0))
        linhas_contrib.append([nome, fmt(var, 6), fmt(100 * var / var_total, 2) + "%"])
        linhas_estudo.append([nome, fmt(sigma, 5),
                              fmt(K_ESTUDO * sigma, 5),
                              fmt(100 * sigma / base_estudo, 2) + "%"])
    return [
        ("subtitulo", "Componentes de variância"),
        ("tabela", ["fonte", "variância", "%contribuição"], linhas_contrib),
        ("subtitulo", f"Variação do estudo ({K_ESTUDO:g}·σ)"),
        ("tabela", ["fonte", "σ", f"variação ({K_ESTUDO:g}σ)", "%variação do estudo"],
         linhas_estudo),
    ]


def _avaliacao_grr(pct_grr: float, ndc: int) -> str:
    from app.reports.formatacao import fmt

    if pct_grr < 10:
        situacao = "aceitável (< 10%)"
    elif pct_grr <= 30:
        situacao = "condicionalmente aceitável (10% a 30%) — pode servir conforme "
        situacao += "a criticidade da aplicação"
    else:
        situacao = "inaceitável (> 30%) — o sistema de medição precisa ser melhorado"
    return (f"%Gage R&R = {fmt(pct_grr, 1)}% da variação do estudo: {situacao}. "
            f"ndc = {ndc} (número de categorias distintas; desejável ≥ 5).")


def gage_rr_cruzado(df: pd.DataFrame, medicao: str, peca: str, operador: str,
                    variacao_processo: float | None = None) -> ResultadoComposto:
    from app.reports.formatacao import fmt, fmt_p

    dados = _dados_msa(df, medicao, peca, operador)
    r = _exigir_balanceado(dados, ["p", "o"])
    n_pecas = dados["p"].nunique()
    n_oper = dados["o"].nunique()
    if n_pecas < 2 or n_oper < 2 or r < 2:
        raise ErroAnalise("O estudo cruzado exige ≥ 2 peças, ≥ 2 operadores e "
                          "≥ 2 repetições.")

    modelo = smf.ols("y ~ C(p) * C(o)", data=dados).fit()
    tabela = sm.stats.anova_lm(modelo, typ=2)
    ms = tabela["sum_sq"] / tabela["df"]
    ms_p, ms_o = float(ms["C(p)"]), float(ms["C(o)"])
    ms_po, ms_e = float(ms["C(p):C(o)"]), float(ms["Residual"])
    p_interacao = float(tabela.loc["C(p):C(o)", "PR(>F)"])

    com_interacao = p_interacao <= 0.25
    if com_interacao:
        var_e = ms_e
        var_po = max(0.0, (ms_po - ms_e) / r)
        var_o = max(0.0, (ms_o - ms_po) / (n_pecas * r))
        var_p = max(0.0, (ms_p - ms_po) / (n_oper * r))
    else:
        # interação removida: repete a ANOVA sem o termo (procedimento AIAG)
        modelo = smf.ols("y ~ C(p) + C(o)", data=dados).fit()
        tabela2 = sm.stats.anova_lm(modelo, typ=2)
        ms2 = tabela2["sum_sq"] / tabela2["df"]
        ms_e2 = float(ms2["Residual"])
        var_e = ms_e2
        var_po = 0.0
        var_o = max(0.0, (float(ms2["C(o)"]) - ms_e2) / (n_pecas * r))
        var_p = max(0.0, (float(ms2["C(p)"]) - ms_e2) / (n_oper * r))

    componentes = {"repetitividade": var_e,
                   "reprodutibilidade": var_o + var_po,
                   "peca": var_p}
    sigma_grr = np.sqrt(var_e + var_o + var_po)
    sigma_total = np.sqrt(sum(componentes.values()))
    pct_grr = 100 * sigma_grr / (variacao_processo or sigma_total)
    ndc = max(1, int(np.floor(1.41 * np.sqrt(var_p) / sigma_grr))) \
        if sigma_grr > 0 else 99

    linhas_anova = []
    for indice in tabela.index:
        rotulo = {"C(p)": "peça", "C(o)": "operador",
                  "C(p):C(o)": "peça × operador", "Residual": "repetitividade"}[str(indice)]
        linha = tabela.loc[indice]
        linhas_anova.append([
            rotulo, int(linha["df"]), fmt(float(linha["sum_sq"])),
            fmt(float(linha["sum_sq"] / linha["df"])),
            fmt(float(linha["F"]), 2) if np.isfinite(linha.get("F", np.nan)) else "",
            fmt_p(float(linha["PR(>F)"])) if np.isfinite(linha.get("PR(>F)", np.nan)) else ""])

    itens: list[tuple] = [
        ("nota", f"Plano: {n_pecas} peças × {n_oper} operadores × {r} repetições "
                 f"= {len(dados)} medições."),
        ("subtitulo", "ANOVA (com interação)"),
        ("tabela", ["fonte", "GL", "SQ", "QM", "F", "p-valor"], linhas_anova),
        ("nota", ("Interação peça × operador mantida (p ≤ 0,25)."
                  if com_interacao else
                  f"Interação removida (p = {fmt_p(p_interacao)} > 0,25) e ANOVA "
                  "reajustada — procedimento padrão.")),
    ]
    itens += _tabela_grr(componentes, sigma_total, variacao_processo)
    itens.append(("interpretacao", _avaliacao_grr(pct_grr, ndc)))
    if variacao_processo:
        itens.append(("nota", f"%variação calculada sobre a variação de processo "
                              f"informada (σ = {fmt(variacao_processo)})."))
    return ResultadoComposto(
        titulo=f"Gage R&R cruzado (ANOVA): {medicao}",
        itens=itens,
        dados={"dados": dados, "medicao": medicao, "peca": peca,
               "operador": operador, "pct_grr": pct_grr, "ndc": ndc},
    )


def gage_rr_aninhado(df: pd.DataFrame, medicao: str, peca: str,
                     operador: str) -> ResultadoComposto:
    from app.reports.formatacao import fmt, fmt_p

    dados = _dados_msa(df, medicao, peca, operador)
    r = _exigir_balanceado(dados, ["o", "p"])
    n_oper = dados["o"].nunique()
    pecas_por_oper = dados.groupby("o", observed=True)["p"].nunique()
    if pecas_por_oper.nunique() != 1:
        raise ErroAnalise("O plano aninhado exige o mesmo número de peças por "
                          "operador.")
    p_por_o = int(pecas_por_oper.iloc[0])
    if r < 2 or p_por_o < 2 or n_oper < 2:
        raise ErroAnalise("O estudo aninhado exige ≥ 2 operadores, ≥ 2 peças por "
                          "operador e ≥ 2 repetições.")

    # Somas de quadrados do plano aninhado balanceado (fórmulas clássicas):
    # SS_O = p·r·Σ(ȳ_o − ȳ)²; SS_P(O) = r·ΣΣ(ȳ_op − ȳ_o)²; SS_E por diferença.
    media_geral = float(dados["y"].mean())
    medias_oper = dados.groupby("o", observed=True)["y"].mean()
    medias_cel = dados.groupby(["o", "p"], observed=True)["y"].mean()

    ss_o = p_por_o * r * float(((medias_oper - media_geral) ** 2).sum())
    ss_po = r * float(sum(
        (medias_cel[oper] - medias_oper[oper]).pow(2).sum()
        for oper in medias_oper.index))
    ss_total = float(((dados["y"] - media_geral) ** 2).sum())
    ss_e = ss_total - ss_o - ss_po

    gl_o = n_oper - 1
    gl_po = n_oper * (p_por_o - 1)
    gl_e = n_oper * p_por_o * (r - 1)
    ms_o = ss_o / gl_o
    ms_po = ss_po / gl_po
    ms_e = ss_e / gl_e
    tabela = pd.DataFrame(
        {"df": [gl_o, gl_po, gl_e], "sum_sq": [ss_o, ss_po, ss_e],
         "F": [ms_o / ms_po, ms_po / ms_e, np.nan],
         "PR(>F)": [float(stats.f.sf(ms_o / ms_po, gl_o, gl_po)),
                    float(stats.f.sf(ms_po / ms_e, gl_po, gl_e)), np.nan]},
        index=["C(o)", "C(o):C(p)", "Residual"])

    var_e = ms_e
    var_p = max(0.0, (ms_po - ms_e) / r)
    var_o = max(0.0, (ms_o - ms_po) / (p_por_o * r))
    componentes = {"repetitividade": var_e, "reprodutibilidade": var_o,
                   "peca": var_p}
    sigma_grr = np.sqrt(var_e + var_o)
    sigma_total = np.sqrt(sum(componentes.values()))
    pct_grr = 100 * sigma_grr / sigma_total
    ndc = max(1, int(np.floor(1.41 * np.sqrt(var_p) / sigma_grr))) \
        if sigma_grr > 0 else 99

    linhas_anova = []
    for indice in tabela.index:
        rotulo = {"C(o)": "operador", "C(o):C(p)": "peça (dentro de operador)",
                  "Residual": "repetitividade"}[str(indice)]
        linha = tabela.loc[indice]
        linhas_anova.append([
            rotulo, int(linha["df"]), fmt(float(linha["sum_sq"])),
            fmt(float(linha["sum_sq"] / linha["df"])),
            fmt(float(linha["F"]), 2) if np.isfinite(linha.get("F", np.nan)) else "",
            fmt_p(float(linha["PR(>F)"])) if np.isfinite(linha.get("PR(>F)", np.nan)) else ""])

    itens = [
        ("nota", f"Plano aninhado: {n_oper} operadores × {p_por_o} peças cada × "
                 f"{r} repetições (peças diferentes por operador)."),
        ("subtitulo", "ANOVA (aninhada)"),
        ("tabela", ["fonte", "GL", "SQ", "QM", "F", "p-valor"], linhas_anova),
    ]
    itens += _tabela_grr(componentes, sigma_total, None)
    itens.append(("interpretacao", _avaliacao_grr(pct_grr, ndc)))
    return ResultadoComposto(
        titulo=f"Gage R&R aninhado (ANOVA): {medicao}",
        itens=itens,
        dados={"dados": dados, "pct_grr": pct_grr, "ndc": ndc},
    )


def linearidade_vies(df: pd.DataFrame, medicao: str, referencia: str,
                     variacao_processo: float | None = None,
                     alfa: float = 0.05) -> ResultadoComposto:
    from app.reports.formatacao import fmt, fmt_p

    dados = pd.DataFrame({
        "y": pd.to_numeric(df[medicao], errors="coerce"),
        "ref": pd.to_numeric(df[referencia], errors="coerce"),
    }).dropna()
    if dados["ref"].nunique() < 2:
        raise ErroAnalise("A linearidade exige pelo menos 2 valores de referência "
                          "distintos.")
    dados["vies"] = dados["y"] - dados["ref"]

    linhas_vies = []
    for ref, grupo in dados.groupby("ref"):
        vies = grupo["vies"]
        if len(vies) >= 2 and vies.std(ddof=1) > 0:
            t, p = stats.ttest_1samp(vies, 0)
            p_txt = fmt_p(float(p))
        else:
            p_txt = "—"
        linhas_vies.append([fmt(float(ref)), len(vies),
                            fmt(float(vies.mean())), p_txt])

    modelo = sm.OLS(dados["vies"],
                    sm.add_constant(dados["ref"].astype(float))).fit()
    inclinacao = float(modelo.params.iloc[1])
    p_inclinacao = float(modelo.pvalues.iloc[1])
    vies_medio = float(dados["vies"].mean())
    _, p_vies_geral = stats.ttest_1samp(dados["vies"], 0)

    itens: list[tuple] = [
        ("subtitulo", "Viés por valor de referência"),
        ("tabela", ["referência", "n", "viés médio", "p-valor (viés = 0)"],
         linhas_vies),
        ("subtitulo", "Linearidade (regressão: viés ~ referência)"),
        ("tabela", ["item", "valor"],
         [["inclinação", fmt(inclinacao, 5)],
          ["p-valor da inclinação", fmt_p(p_inclinacao)],
          ["viés médio geral", fmt(vies_medio, 5)],
          ["p-valor do viés geral", fmt_p(float(p_vies_geral))]]
         + ([["%linearidade (|inclinação|·σproc/σproc)",
              fmt(100 * abs(inclinacao), 2) + "%"],
             ["%viés (|viés|/variação do processo)",
              fmt(100 * abs(vies_medio) / variacao_processo, 2) + "%"]]
            if variacao_processo else [])),
        ("interpretacao",
         ("A inclinação é significativa: o viés muda com o valor medido — o "
          "instrumento tem problema de linearidade."
          if p_inclinacao < alfa else
          "A inclinação não é significativa: não há evidência de problema de "
          "linearidade.")
         + (" Há viés sistemático significativo."
            if p_vies_geral < alfa else " Não há evidência de viés sistemático.")),
    ]
    return ResultadoComposto(
        titulo=f"Linearidade e viés do sistema de medição: {medicao}",
        itens=itens,
        dados={"dados": dados},
    )


def estudo_tipo1(dados, coluna: str, referencia: float, tolerancia: float,
                 percentual_k: float = 20.0) -> ResultadoComposto:
    from app.reports.formatacao import fmt, fmt_p
    from .util import limpar_numerica

    x = limpar_numerica(dados, coluna, n_minimo=10)
    if tolerancia <= 0:
        raise ErroAnalise("A tolerância (LSE − LIE) deve ser maior que zero.")
    media = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    vies = media - referencia
    if s == 0:
        raise ErroAnalise("Desvio-padrão nulo nas medições repetidas.")
    cg = (percentual_k / 100 * tolerancia) / (K_ESTUDO * s)
    cgk = (percentual_k / 200 * tolerancia - abs(vies)) / (K_ESTUDO / 2 * s)
    t, p_vies = stats.ttest_1samp(x, referencia)

    if x.size < 25:
        aviso = [("aviso", f"n = {x.size}: recomenda-se ≥ 25 medições repetidas "
                           "para o estudo Tipo 1.")]
    else:
        aviso = []
    return ResultadoComposto(
        titulo=f"Estudo Tipo 1 do sistema de medição: {coluna}",
        itens=[
            ("tabela", ["item", "valor"],
             [["n (medições da mesma peça)", str(x.size)],
              ["referência", fmt(referencia)], ["média das medições", fmt(media)],
              ["viés", fmt(vies, 5)], ["desvio-padrão", fmt(s, 5)],
              ["tolerância", fmt(tolerancia)],
              ["Cg", fmt(float(cg), 2)], ["Cgk", fmt(float(cgk), 2)],
              ["p-valor (viés = 0)", fmt_p(float(p_vies))]]),
            ("interpretacao",
             f"Cg = {fmt(float(cg), 2)} e Cgk = {fmt(float(cgk), 2)}: valores "
             "≥ 1,33 indicam que a variação (e o viés, no Cgk) do instrumento "
             f"ocupa fração aceitável da tolerância (critério: {fmt(percentual_k, 0)}% "
             "da tolerância)."),
        ] + aviso,
        dados={"x": x, "referencia": referencia, "tolerancia": tolerancia,
               "percentual_k": percentual_k},
    )


def _kappa_cohen(a: pd.Series, b: pd.Series) -> float:
    tabela = pd.crosstab(a, b)
    categorias = sorted(set(tabela.index) | set(tabela.columns))
    tabela = tabela.reindex(index=categorias, columns=categorias, fill_value=0)
    matriz = tabela.to_numpy(dtype=float)
    n = matriz.sum()
    po = np.trace(matriz) / n
    pe = float((matriz.sum(axis=0) * matriz.sum(axis=1)).sum() / n**2)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def concordancia_atributos(df: pd.DataFrame, peca: str, avaliador: str,
                           resultado: str,
                           padrao: str | None = None) -> ResultadoComposto:
    from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

    from app.reports.formatacao import fmt

    dados = pd.DataFrame({
        "p": df[peca].astype(object), "a": df[avaliador].astype(object),
        "r": df[resultado].astype(object),
    })
    if padrao:
        dados["s"] = df[padrao].astype(object)
    for c in dados.columns:
        vazio = dados[c].isna() | (dados[c].astype(str).str.strip() == "")
        dados[c] = dados[c].astype(str).where(~vazio, np.nan)
    dados = dados.dropna()
    if dados.empty:
        raise ErroAnalise("Sem dados completos para a análise de concordância.")

    # dentro de cada avaliador (todas as tentativas da mesma peça iguais)
    linhas_dentro = []
    for aval, grupo in dados.groupby("a"):
        por_peca = grupo.groupby("p")["r"].nunique()
        repetidas = grupo.groupby("p")["r"].size()
        pecas_repetidas = por_peca[repetidas >= 2]
        if len(pecas_repetidas):
            pct = 100 * float((pecas_repetidas == 1).mean())
            linhas_dentro.append([aval, len(pecas_repetidas), fmt(pct, 1) + "%"])
    itens: list[tuple] = []
    if linhas_dentro:
        itens += [("subtitulo", "Concordância dentro de cada avaliador "
                                "(repetitividade)"),
                  ("tabela", ["avaliador", "peças avaliadas 2+ vezes",
                              "% concordância"], linhas_dentro)]

    # entre avaliadores (kappa de Fleiss usando a 1ª avaliação de cada um)
    primeira = (dados.groupby(["p", "a"], observed=True)["r"].first()
                .unstack())
    completas = primeira.dropna()
    if completas.shape[0] >= 2 and completas.shape[1] >= 2:
        todas_iguais = completas.nunique(axis=1) == 1
        pct_entre = 100 * float(todas_iguais.mean())
        matriz, _ = aggregate_raters(completas.to_numpy())
        kappa_f = float(fleiss_kappa(matriz))
        itens += [("subtitulo", "Concordância entre avaliadores"),
                  ("tabela", ["item", "valor"],
                   [["peças com todos concordando", fmt(pct_entre, 1) + "%"],
                    ["kappa de Fleiss", fmt(kappa_f, 3)]])]
    else:
        kappa_f = None

    # contra o padrão
    if padrao:
        linhas_padrao = []
        for aval, grupo in dados.groupby("a"):
            acerto = 100 * float((grupo["r"] == grupo["s"]).mean())
            kappa_c = _kappa_cohen(grupo["r"], grupo["s"])
            linhas_padrao.append([aval, fmt(acerto, 1) + "%", fmt(kappa_c, 3)])
        itens += [("subtitulo", "Concordância com o padrão"),
                  ("tabela", ["avaliador", "% de acerto", "kappa de Cohen"],
                   linhas_padrao)]

    interpretacao = ("Kappa mede concordância além do acaso: < 0,4 fraca; "
                     "0,4–0,75 moderada a boa; > 0,75 excelente (Landis & Koch/"
                     "AIAG: desejável ≥ 0,75).")
    if kappa_f is not None:
        interpretacao = (f"Kappa de Fleiss entre avaliadores = {fmt(kappa_f, 2)}. "
                         + interpretacao)
    itens.append(("interpretacao", interpretacao))
    return ResultadoComposto(titulo="Análise de concordância por atributos",
                             itens=itens)


def planilha_coleta_grr(n_pecas: int, n_operadores: int, n_replicas: int,
                        semente: int | None = None) -> pd.DataFrame:
    """Planilha de coleta para Gage R&R cruzado, em ordem aleatorizada."""
    if not (2 <= n_pecas <= 50 and 2 <= n_operadores <= 10
            and 2 <= n_replicas <= 10):
        raise ErroAnalise("Use 2–50 peças, 2–10 operadores e 2–10 repetições.")
    linhas = [(f"P{p + 1}", f"Operador {o + 1}")
              for _ in range(n_replicas)
              for o in range(n_operadores)
              for p in range(n_pecas)]
    rng = np.random.default_rng(semente)
    ordem = rng.permutation(len(linhas))
    tabela = pd.DataFrame([linhas[i] for i in ordem],
                          columns=["peça", "operador"])
    tabela.insert(0, "ordem", np.arange(1, len(linhas) + 1))
    tabela["medição"] = np.nan
    return tabela
