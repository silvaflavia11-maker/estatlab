"""Regressão: linear (com stepwise e melhores subconjuntos), logística
binária e Poisson. Ajustes via statsmodels."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .resultados import ErroAnalise, ResultadoComposto


def _preparar(df: pd.DataFrame, resposta: str, preditores: list[str],
              binaria: bool = False) -> tuple[pd.Series, pd.DataFrame, list[str]]:
    if not preditores:
        raise ErroAnalise("Selecione pelo menos um preditor.")
    if resposta in preditores:
        raise ErroAnalise("A resposta não pode estar entre os preditores.")
    x = df[preditores].apply(pd.to_numeric, errors="coerce")
    if binaria:
        y_bruto = df[resposta]
        vazio = y_bruto.isna() | (y_bruto.astype(str).str.strip() == "")
        y_bruto = y_bruto.where(~vazio, np.nan)
        niveis = sorted(y_bruto.dropna().astype(str).unique())
        if len(niveis) != 2:
            raise ErroAnalise(
                f"A resposta '{resposta}' precisa ter exatamente 2 valores distintos "
                f"(encontrados: {len(niveis)})."
            )
        y = y_bruto.astype(str).map({niveis[0]: 0.0, niveis[1]: 1.0})
        y.loc[y_bruto.isna()] = np.nan
        rotulo_evento = niveis[1]
    else:
        y = pd.to_numeric(df[resposta], errors="coerce")
        rotulo_evento = None
    completos = pd.concat([y.rename("__y"), x], axis=1).dropna()
    minimo = len(preditores) + 2
    if len(completos) < minimo:
        raise ErroAnalise(
            f"Observações completas insuficientes ({len(completos)}); esta análise "
            f"exige pelo menos {minimo} com os preditores selecionados."
        )
    return completos["__y"], completos[preditores], ([rotulo_evento, niveis[0]]
                                                     if binaria else [])


def _aicc(modelo) -> float:
    k = modelo.df_model + 2  # preditores + intercepto + variância do erro
    n = modelo.nobs
    if n - k - 1 <= 0:
        return float("inf")
    return float(modelo.aic + 2 * k * (k + 1) / (n - k - 1))


def _equacao(resposta: str, params: pd.Series) -> str:
    from app.reports.formatacao import fmt

    termos = [fmt(float(params.iloc[0]))]
    for nome, coef in params.iloc[1:].items():
        sinal = "+" if coef >= 0 else "−"
        termos.append(f" {sinal} {fmt(abs(float(coef)))}·{nome}")
    return f"{resposta} = " + "".join(termos)


def regressao_linear(df: pd.DataFrame, resposta: str, preditores: list[str],
                     alfa: float = 0.05,
                     predicao: dict[str, float] | None = None) -> ResultadoComposto:
    from app.reports.formatacao import fmt, fmt_p

    y, x, _ = _preparar(df, resposta, preditores)
    modelo = sm.OLS(y, sm.add_constant(x)).fit()

    conf = modelo.conf_int(alpha=alfa)
    linhas_coef = []
    for nome in modelo.params.index:
        rotulo = "constante" if nome == "const" else nome
        linhas_coef.append([rotulo, fmt(float(modelo.params[nome])),
                            fmt(float(modelo.bse[nome])),
                            fmt(float(modelo.tvalues[nome]), 2),
                            fmt_p(float(modelo.pvalues[nome])),
                            f"({fmt(float(conf.loc[nome, 0]))}; {fmt(float(conf.loc[nome, 1]))})"])

    gl_reg = int(modelo.df_model)
    gl_res = int(modelo.df_resid)
    sq_reg = float(modelo.ess)
    sq_res = float(modelo.ssr)
    itens: list[tuple] = [
        ("nota", "Equação ajustada: " + _equacao(resposta, modelo.params)),
        ("subtitulo", "Coeficientes"),
        ("tabela", ["termo", "coeficiente", "EP", "t", "p-valor",
                    f"IC {fmt(100 * (1 - alfa), 0)}%"], linhas_coef),
        ("subtitulo", "Análise de variância da regressão"),
        ("tabela", ["fonte", "GL", "SQ", "QM", "F", "p-valor"],
         [["regressão", gl_reg, fmt(sq_reg), fmt(sq_reg / max(gl_reg, 1)),
           fmt(float(modelo.fvalue), 2), fmt_p(float(modelo.f_pvalue))],
          ["erro", gl_res, fmt(sq_res), fmt(sq_res / gl_res), "", ""],
          ["total", gl_reg + gl_res, fmt(sq_reg + sq_res), "", "", ""]]),
        ("nota", f"R² = {fmt(100 * modelo.rsquared, 2)}%  •  "
                 f"R² ajustado = {fmt(100 * modelo.rsquared_adj, 2)}%  •  "
                 f"s = {fmt(float(np.sqrt(modelo.mse_resid)))}  •  n = {int(modelo.nobs)}  •  "
                 f"AICc = {fmt(_aicc(modelo), 2)}  •  BIC = {fmt(float(modelo.bic), 2)}"),
    ]

    significativos = [str(nome) for nome in modelo.params.index[1:]
                      if modelo.pvalues[nome] < alfa]
    frase_global = (f"O modelo é significativo (F: p = {fmt_p(float(modelo.f_pvalue))}) "
                    if modelo.f_pvalue < alfa else
                    f"O modelo não é significativo (F: p = {fmt_p(float(modelo.f_pvalue))}) ")
    frase_pred = (f"e explica {fmt(100 * modelo.rsquared, 1)}% da variação de "
                  f"'{resposta}'. Preditores significativos ao nível α = {fmt(alfa, 2)}: "
                  + (", ".join(significativos) if significativos else "nenhum") + ".")
    itens.append(("interpretacao", frase_global + frase_pred))

    if len(preditores) >= 2:
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        matriz = sm.add_constant(x).to_numpy(dtype=float)
        vifs = [(preditores[i], variance_inflation_factor(matriz, i + 1))
                for i in range(len(preditores))]
        piores = [f"{nome} (VIF = {fmt(v, 1)})" for nome, v in vifs if v > 10]
        if piores:
            itens.append(("aviso", "Multicolinearidade alta em: " + ", ".join(piores)
                          + ". Os coeficientes individuais podem ser instáveis."))

    if predicao:
        faltando = [p for p in preditores if p not in predicao]
        if faltando:
            raise ErroAnalise("Informe o valor de todos os preditores para a "
                              f"predição (faltam: {', '.join(faltando)}).")
        novo = pd.DataFrame([{**{"const": 1.0}, **predicao}])[["const", *preditores]]
        prev = modelo.get_prediction(novo)
        media = prev.summary_frame(alpha=alfa)
        itens += [
            ("subtitulo", "Predição para " + ", ".join(f"{k} = {fmt(float(v))}"
                                                       for k, v in predicao.items())),
            ("tabela", ["valor previsto", f"IC {fmt(100 * (1 - alfa), 0)}% (média)",
                        f"IP {fmt(100 * (1 - alfa), 0)}% (observação individual)"],
             [[fmt(float(media["mean"].iloc[0])),
               f"({fmt(float(media['mean_ci_lower'].iloc[0]))}; {fmt(float(media['mean_ci_upper'].iloc[0]))})",
               f"({fmt(float(media['obs_ci_lower'].iloc[0]))}; {fmt(float(media['obs_ci_upper'].iloc[0]))})"]]),
            ("nota", "IC: intervalo para a média de Y nesses valores; IP: intervalo "
                     "para uma nova observação individual (sempre mais largo)."),
        ]

    itens.append(("aviso", "Verifique os pressupostos nos gráficos de resíduos: "
                           "linearidade, variância constante e normalidade."))
    return ResultadoComposto(
        titulo=f"Regressão linear: {resposta} ~ " + " + ".join(preditores),
        itens=itens,
        dados={"residuos": np.asarray(modelo.resid),
               "ajustados": np.asarray(modelo.fittedvalues),
               "resposta": resposta, "x": x, "y": np.asarray(y)},
    )


def stepwise(df: pd.DataFrame, resposta: str, preditores: list[str],
             criterio: str = "p", alfa_entrada: float = 0.15,
             alfa: float = 0.05) -> ResultadoComposto:
    """Seleção progressiva (forward) por p-valor, AICc ou BIC."""
    from app.reports.formatacao import fmt, fmt_p

    y, x, _ = _preparar(df, resposta, preditores)

    def ajustar(colunas: list[str]):
        matriz = sm.add_constant(x[colunas]) if colunas else pd.DataFrame(
            {"const": np.ones(len(y))}, index=x.index)
        return sm.OLS(y, matriz).fit()

    def valor_criterio(modelo) -> float:
        return _aicc(modelo) if criterio == "aicc" else float(modelo.bic)

    selecionados: list[str] = []
    passos: list[list] = []
    while True:
        restantes = [p for p in preditores if p not in selecionados]
        if not restantes:
            break
        if criterio == "p":
            candidatos = []
            for cand in restantes:
                m = ajustar(selecionados + [cand])
                candidatos.append((float(m.pvalues[cand]), cand))
            melhor_p, melhor = min(candidatos)
            if melhor_p >= alfa_entrada:
                break
            selecionados.append(melhor)
            passos.append([len(passos) + 1, f"entra {melhor}",
                           f"p = {fmt_p(melhor_p)}"])
        else:
            atual = valor_criterio(ajustar(selecionados))
            candidatos = [(valor_criterio(ajustar(selecionados + [c])), c)
                          for c in restantes]
            melhor_valor, melhor = min(candidatos)
            if melhor_valor >= atual:
                break
            selecionados.append(melhor)
            rotulo = "AICc" if criterio == "aicc" else "BIC"
            passos.append([len(passos) + 1, f"entra {melhor}",
                           f"{rotulo} = {fmt(melhor_valor, 2)}"])

    nomes = {"p": f"p-valor (α de entrada = {fmt(alfa_entrada, 2)})",
             "aicc": "AICc", "bic": "BIC"}
    if not selecionados:
        return ResultadoComposto(
            titulo=f"Stepwise ({nomes[criterio]}): {resposta}",
            itens=[("interpretacao",
                    "Nenhum preditor atendeu ao critério de entrada — nenhum modelo "
                    "foi selecionado. Considere relaxar o critério ou revisar os "
                    "preditores candidatos.")],
        )

    resultado = regressao_linear(df, resposta, selecionados, alfa)
    resultado.titulo = f"Stepwise ({nomes[criterio]}): " + resultado.titulo
    resultado.itens = ([("subtitulo", "Passos da seleção (forward)"),
                        ("tabela", ["passo", "ação", "critério"], passos)]
                       + resultado.itens)
    return resultado


def melhores_subconjuntos(df: pd.DataFrame, resposta: str,
                          preditores: list[str]) -> ResultadoComposto:
    from app.reports.formatacao import fmt

    if len(preditores) > 12:
        raise ErroAnalise("Melhores subconjuntos aceita até 12 preditores "
                          f"(selecionados: {len(preditores)}).")
    y, x, _ = _preparar(df, resposta, preditores)

    linhas = []
    for tamanho in range(1, len(preditores) + 1):
        modelos = []
        for combo in combinations(preditores, tamanho):
            m = sm.OLS(y, sm.add_constant(x[list(combo)])).fit()
            modelos.append((m.rsquared_adj, m, combo))
        modelos.sort(key=lambda item: -item[0])
        for r2adj, m, combo in modelos[:3]:  # 3 melhores por tamanho
            linhas.append([tamanho, fmt(100 * m.rsquared, 1),
                           fmt(100 * r2adj, 1), fmt(_aicc(m), 2),
                           fmt(float(m.bic), 2),
                           fmt(float(np.sqrt(m.mse_resid))), ", ".join(combo)])
    return ResultadoComposto(
        titulo=f"Melhores subconjuntos: {resposta}",
        itens=[
            ("tabela", ["nº preditores", "R² (%)", "R² aj. (%)", "AICc", "BIC",
                        "s", "preditores"], linhas),
            ("nota", "São mostrados os 3 melhores modelos de cada tamanho, por R² "
                     "ajustado."),
            ("interpretacao", "Procure o menor modelo com R² ajustado alto e "
                              "AICc/BIC baixos — critérios de informação penalizam "
                              "complexidade e ajudam a evitar sobreajuste."),
        ],
    )


def regressao_logistica(df: pd.DataFrame, resposta: str, preditores: list[str],
                        alfa: float = 0.05) -> ResultadoComposto:
    from app.reports.formatacao import fmt, fmt_p

    y, x, niveis = _preparar(df, resposta, preditores, binaria=True)
    evento, referencia = niveis
    try:
        modelo = sm.Logit(y, sm.add_constant(x)).fit(disp=False)
    except Exception as erro:
        raise ErroAnalise(f"O ajuste da regressão logística falhou ({erro}). "
                          "Verifique separação perfeita ou preditores redundantes.")

    conf = modelo.conf_int(alpha=alfa)
    linhas = []
    for nome in modelo.params.index:
        rotulo = "constante" if nome == "const" else nome
        rc = "—" if nome == "const" else fmt(float(np.exp(modelo.params[nome])), 3)
        ic_rc = ("—" if nome == "const" else
                 f"({fmt(float(np.exp(conf.loc[nome, 0])), 3)}; "
                 f"{fmt(float(np.exp(conf.loc[nome, 1])), 3)})")
        linhas.append([rotulo, fmt(float(modelo.params[nome])),
                       fmt(float(modelo.bse[nome])),
                       fmt(float(modelo.tvalues[nome]), 2),
                       fmt_p(float(modelo.pvalues[nome])), rc, ic_rc])

    significativos = [str(n) for n in modelo.params.index[1:]
                      if modelo.pvalues[n] < alfa]
    return ResultadoComposto(
        titulo=f"Regressão logística binária: {resposta} ~ " + " + ".join(preditores),
        itens=[
            ("nota", f"Evento modelado: '{resposta}' = \"{evento}\" "
                     f"(referência: \"{referencia}\"). n = {int(modelo.nobs)}."),
            ("subtitulo", "Coeficientes"),
            ("tabela", ["termo", "coeficiente", "EP", "z", "p-valor",
                        "razão de chances", f"IC {fmt(100 * (1 - alfa), 0)}% da RC"],
             linhas),
            ("nota", f"Teste da razão de verossimilhança do modelo: "
                     f"p = {fmt_p(float(modelo.llr_pvalue))}  •  "
                     f"pseudo-R² (McFadden) = {fmt(float(modelo.prsquared), 3)}"),
            ("interpretacao",
             ("O modelo é significativo. " if modelo.llr_pvalue < alfa
              else "O modelo não é significativo. ")
             + "Razão de chances > 1 indica que aumentos no preditor elevam a "
               f"chance do evento \"{evento}\"; < 1, que reduzem. Significativos: "
             + (", ".join(significativos) if significativos else "nenhum") + "."),
        ],
        dados={"ajustados": np.asarray(modelo.predict()), "y": np.asarray(y)},
    )


def regressao_poisson(df: pd.DataFrame, resposta: str, preditores: list[str],
                      alfa: float = 0.05) -> ResultadoComposto:
    from app.reports.formatacao import fmt, fmt_p

    y, x, _ = _preparar(df, resposta, preditores)
    if (y < 0).any() or not np.allclose(y, np.round(y)):
        raise ErroAnalise(f"A resposta '{resposta}' deve conter contagens "
                          "(inteiros ≥ 0) para regressão de Poisson.")
    modelo = sm.GLM(y, sm.add_constant(x), family=sm.families.Poisson()).fit()

    conf = modelo.conf_int(alpha=alfa)
    linhas = []
    for nome in modelo.params.index:
        rotulo = "constante" if nome == "const" else nome
        rt = "—" if nome == "const" else fmt(float(np.exp(modelo.params[nome])), 3)
        ic_rt = ("—" if nome == "const" else
                 f"({fmt(float(np.exp(conf.loc[nome, 0])), 3)}; "
                 f"{fmt(float(np.exp(conf.loc[nome, 1])), 3)})")
        linhas.append([rotulo, fmt(float(modelo.params[nome])),
                       fmt(float(modelo.bse[nome])),
                       fmt(float(modelo.tvalues[nome]), 2),
                       fmt_p(float(modelo.pvalues[nome])), rt, ic_rt])

    dispersao = float(modelo.pearson_chi2 / modelo.df_resid)
    itens: list[tuple] = [
        ("subtitulo", "Coeficientes"),
        ("tabela", ["termo", "coeficiente", "EP", "z", "p-valor",
                    "razão de taxas", f"IC {fmt(100 * (1 - alfa), 0)}% da RT"], linhas),
        ("nota", f"Deviance = {fmt(float(modelo.deviance), 2)} com "
                 f"{int(modelo.df_resid)} GL  •  dispersão (χ²/GL) = {fmt(dispersao, 2)}"
                 f"  •  n = {int(modelo.nobs)}"),
        ("interpretacao", "A razão de taxas indica o fator multiplicativo na taxa "
                          "esperada de eventos para +1 unidade do preditor."),
    ]
    if dispersao > 1.5:
        itens.append(("aviso", f"Superdispersão detectada (χ²/GL = {fmt(dispersao, 2)} "
                               "> 1,5): os EPs podem estar subestimados. Interprete "
                               "com cautela (alternativas: binomial negativa, na "
                               "Fase 4)."))
    return ResultadoComposto(
        titulo=f"Regressão de Poisson: {resposta} ~ " + " + ".join(preditores),
        itens=itens,
        dados={"ajustados": np.asarray(modelo.predict()), "y": np.asarray(y)},
    )
