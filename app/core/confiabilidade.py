"""Confiabilidade/sobrevivência (lifelines).

- Não paramétrica: Kaplan-Meier (censura à direita).
- Paramétrica: Weibull, lognormal e exponencial, com censura à direita,
  à esquerda ou por intervalo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .resultados import ErroAnalise, ResultadoComposto

FAMILIAS = ("Weibull", "Lognormal", "Exponencial")


def _tempos_censura(df: pd.DataFrame, tempo: str,
                    censura: str | None) -> tuple[np.ndarray, np.ndarray]:
    t = pd.to_numeric(df[tempo], errors="coerce")
    if censura:
        c = pd.to_numeric(df[censura], errors="coerce")
        dados = pd.DataFrame({"t": t, "c": c}).dropna()
        eventos = dados["c"].to_numpy()
        if not set(np.unique(eventos)) <= {0.0, 1.0}:
            raise ErroAnalise("A coluna de censura deve conter apenas 0 "
                              "(censurado) e 1 (falha observada).")
    else:
        dados = pd.DataFrame({"t": t}).dropna()
        eventos = np.ones(len(dados))
    tempos = dados["t"].to_numpy(dtype=float)
    if len(tempos) < 5:
        raise ErroAnalise("São necessários pelo menos 5 tempos válidos.")
    if np.any(tempos <= 0):
        raise ErroAnalise("Todos os tempos devem ser positivos.")
    return tempos, eventos.astype(float)


def kaplan_meier(df: pd.DataFrame, tempo: str, censura: str | None = None,
                 alfa: float = 0.05) -> ResultadoComposto:
    from lifelines import KaplanMeierFitter

    from app.reports.formatacao import fmt

    tempos, eventos = _tempos_censura(df, tempo, censura)
    km = KaplanMeierFitter(alpha=alfa)
    km.fit(tempos, event_observed=eventos)

    mediana = float(km.median_survival_time_)
    n_falhas = int(eventos.sum())
    n_censurados = int(len(eventos) - n_falhas)

    tabela_sf = km.survival_function_
    ic = km.confidence_interval_survival_function_
    passos = np.unique(np.round(
        np.quantile(tempos, np.linspace(0.1, 1.0, 8)), 6))
    linhas = []
    for ponto in passos:
        s = float(tabela_sf.iloc[
            tabela_sf.index.get_indexer([ponto], method="ffill")[0], 0])
        lo = float(ic.iloc[ic.index.get_indexer([ponto], method="ffill")[0], 0])
        hi = float(ic.iloc[ic.index.get_indexer([ponto], method="ffill")[0], 1])
        linhas.append([fmt(float(ponto)), fmt(s, 4),
                       f"({fmt(lo, 4)}; {fmt(hi, 4)})"])
    return ResultadoComposto(
        titulo=f"Análise não paramétrica (Kaplan-Meier): {tempo}",
        itens=[
            ("tabela", ["item", "valor"],
             [["n", str(len(tempos))], ["falhas observadas", str(n_falhas)],
              ["censurados (à direita)", str(n_censurados)],
              ["tempo mediano de sobrevivência",
               fmt(mediana) if np.isfinite(mediana) else "não atingido"]]),
            ("subtitulo", "Função de sobrevivência estimada"),
            ("tabela", ["tempo", "S(t)",
                        f"IC {fmt(100 * (1 - alfa), 0)}%"], linhas),
            ("interpretacao",
             "S(t) é a probabilidade de sobreviver além do tempo t, estimada sem "
             "supor distribuição. Unidades censuradas contribuem enquanto "
             "observadas — não descarte censuras."),
        ],
        dados={"km": km, "tempos": tempos, "eventos": eventos, "coluna": tempo},
    )


def _ajustar_familia(nome: str, alfa: float):
    from lifelines import (
        ExponentialFitter,
        LogNormalFitter,
        WeibullFitter,
    )

    return {"Weibull": WeibullFitter, "Lognormal": LogNormalFitter,
            "Exponencial": ExponentialFitter}[nome](alpha=alfa)


def analise_parametrica(df: pd.DataFrame, tempo: str,
                        censura: str | None = None,
                        familia: str = "Weibull",
                        tipo_censura: str = "direita",
                        tempo_fim: str | None = None,
                        alfa: float = 0.05) -> ResultadoComposto:
    """``tipo_censura``: 'direita' (coluna 0/1), 'esquerda' (coluna 0/1) ou
    'intervalo' (colunas tempo inicial/final)."""
    from app.reports.formatacao import fmt

    if familia not in FAMILIAS:
        raise ErroAnalise(f"Família inválida: use {', '.join(FAMILIAS)}.")
    ajuste = _ajustar_familia(familia, alfa)

    if tipo_censura == "intervalo":
        if not tempo_fim:
            raise ErroAnalise("Para censura por intervalo, informe a coluna do "
                              "fim do intervalo.")
        t_ini = pd.to_numeric(df[tempo], errors="coerce")
        t_fim = pd.to_numeric(df[tempo_fim], errors="coerce")
        dados = pd.DataFrame({"a": t_ini, "b": t_fim}).dropna()
        if len(dados) < 5:
            raise ErroAnalise("São necessários pelo menos 5 intervalos válidos.")
        if np.any(dados["a"] <= 0) or np.any(dados["b"] < dados["a"]):
            raise ErroAnalise("Exige 0 < início ≤ fim em todos os intervalos.")
        ajuste.fit_interval_censoring(dados["a"], dados["b"])
        n, n_eventos = len(dados), int((dados["a"] == dados["b"]).sum())
        tempos_plot = dados["b"].to_numpy()
    else:
        tempos, eventos = _tempos_censura(df, tempo, censura)
        if tipo_censura == "direita":
            ajuste.fit(tempos, event_observed=eventos)
        elif tipo_censura == "esquerda":
            ajuste.fit_left_censoring(tempos, event_observed=eventos)
        else:
            raise ErroAnalise("Tipo de censura inválido.")
        n, n_eventos = len(tempos), int(eventos.sum())
        tempos_plot = tempos

    linhas_param = []
    resumo = ajuste.summary
    for nome_parametro in resumo.index:
        linha = resumo.loc[nome_parametro]
        rotulo = {"lambda_": "escala (λ)", "rho_": "forma (β)",
                  "mu_": "μ (escala log)", "sigma_": "σ (forma log)"}.get(
            str(nome_parametro), str(nome_parametro))
        linhas_param.append([
            rotulo, fmt(float(linha["coef"]), 4),
            f"({fmt(float(linha.iloc[2]), 4)}; {fmt(float(linha.iloc[3]), 4)})"])

    quantis = [0.01, 0.05, 0.10, 0.50, 0.90]
    linhas_quantis = [[fmt(100 * q, 0) + "%",
                       fmt(float(ajuste.percentile(q)))] for q in quantis]

    itens: list[tuple] = [
        ("tabela", ["item", "valor"],
         [["família", familia], ["n", str(n)],
          ["falhas observadas", str(n_eventos)],
          ["censura", tipo_censura],
          ["log-verossimilhança", fmt(float(ajuste.log_likelihood_), 2)],
          ["AIC", fmt(float(ajuste.AIC_), 2)]]),
        ("subtitulo", "Parâmetros estimados"),
        ("tabela", ["parâmetro", "estimativa",
                    f"IC {fmt(100 * (1 - alfa), 0)}%"], linhas_param),
        ("subtitulo", "Percentis do tempo de falha (B-lives)"),
        ("tabela", ["percentil", "tempo"], linhas_quantis),
    ]
    if familia == "Weibull":
        forma = float(ajuste.rho_)
        if forma < 1:
            fase = "falhas precoces (mortalidade infantil): β < 1"
        elif forma <= 1.2:
            fase = "taxa de falha aproximadamente constante: β ≈ 1"
        else:
            fase = "desgaste (taxa de falha crescente): β > 1"
        itens.append(("interpretacao",
                      f"Weibull com forma β = {fmt(forma, 2)} — {fase}. "
                      f"B10 = {fmt(float(ajuste.percentile(0.10)))}: 10% das "
                      "unidades falham até esse tempo. Compare famílias pelo AIC "
                      "(menor = melhor)."))
    else:
        itens.append(("interpretacao",
                      "Compare famílias pelo AIC (menor = melhor) e confira o "
                      "gráfico de probabilidade antes de usar os percentis."))
    return ResultadoComposto(
        titulo=f"Análise de distribuição ({familia}): {tempo}",
        itens=itens,
        dados={"ajuste": ajuste, "tempos": tempos_plot, "coluna": tempo,
               "familia": familia},
    )
