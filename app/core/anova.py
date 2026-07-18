"""ANOVA de 1 e 2 fatores, comparações múltiplas e igualdade de variâncias.

Dados no formato empilhado: uma coluna de resposta numérica e coluna(s) de
fator (texto ou numérica tratada como categórica).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from .resultados import ErroAnalise, ResultadoComposto, ResultadoTeste


def _empilhar(df: pd.DataFrame, resposta: str, fatores: list[str]) -> pd.DataFrame:
    dados = pd.DataFrame({"y": pd.to_numeric(df[resposta], errors="coerce")})
    for i, fator in enumerate(fatores):
        serie = df[fator].astype(object)
        vazio = serie.isna() | (serie.astype(str).str.strip() == "")
        dados[f"f{i}"] = serie.astype(str).where(~vazio, np.nan)
    dados = dados.dropna()
    if len(dados) < 3:
        raise ErroAnalise("Menos de 3 observações completas (resposta + fatores).")
    return dados


def _grupos(dados: pd.DataFrame, fator: str = "f0") -> dict[str, np.ndarray]:
    return {str(nivel): grupo["y"].to_numpy()
            for nivel, grupo in dados.groupby(fator, observed=True)}


def _tabela_grupos(grupos: dict[str, np.ndarray], fator: str) -> tuple:
    from app.reports.formatacao import fmt

    linhas = [[nivel, len(x), fmt(float(np.mean(x))), fmt(float(np.std(x, ddof=1)))
               if len(x) > 1 else "—"] for nivel, x in grupos.items()]
    return ("tabela", [fator, "n", "média", "desvio-padrão"], linhas)


def anova_1via(df: pd.DataFrame, resposta: str, fator: str, alfa: float = 0.05,
               comparacao: str | None = None,
               nivel_controle: str | None = None) -> ResultadoComposto:
    """ANOVA de 1 fator; ``comparacao``: None, 'tukey', 'fisher' ou 'dunnett'."""
    from app.reports.formatacao import fmt, fmt_p

    dados = _empilhar(df, resposta, [fator])
    grupos = _grupos(dados)
    if len(grupos) < 2:
        raise ErroAnalise(f"O fator '{fator}' precisa de pelo menos 2 níveis "
                          f"(encontrado: {len(grupos)}).")
    if any(len(x) < 2 for x in grupos.values()):
        raise ErroAnalise("Cada nível do fator precisa de pelo menos 2 observações.")

    modelo = smf.ols("y ~ C(f0)", data=dados).fit()
    tabela_anova = sm.stats.anova_lm(modelo, typ=2)
    ss_fator = float(tabela_anova.loc["C(f0)", "sum_sq"])
    ss_erro = float(tabela_anova.loc["Residual", "sum_sq"])
    gl_fator = int(tabela_anova.loc["C(f0)", "df"])
    gl_erro = int(tabela_anova.loc["Residual", "df"])
    f_valor = float(tabela_anova.loc["C(f0)", "F"])
    p_valor = float(tabela_anova.loc["C(f0)", "PR(>F)"])
    mse = ss_erro / gl_erro

    itens: list[tuple] = [
        ("nota", f"H₀: todas as médias dos níveis de '{fator}' são iguais.  "
                 "H₁: pelo menos uma média difere das demais."),
        ("tabela",
         ["fonte", "GL", "SQ", "QM", "F", "p-valor"],
         [[f"{fator}", gl_fator, fmt(ss_fator), fmt(ss_fator / gl_fator),
           fmt(f_valor), fmt_p(p_valor)],
          ["erro", gl_erro, fmt(ss_erro), fmt(mse), "", ""],
          ["total", gl_fator + gl_erro, fmt(ss_fator + ss_erro), "", "", ""]]),
        _tabela_grupos(grupos, fator),
        ("nota", f"R² = {fmt(100 * modelo.rsquared, 2)}%  •  "
                 f"s (desvio-padrão combinado) = {fmt(np.sqrt(mse))}"),
    ]
    if p_valor < alfa:
        itens.append(("interpretacao",
                      f"Como p = {fmt_p(p_valor)} < α = {fmt(alfa, 2)}, rejeita-se H₀: "
                      f"há evidência de que pelo menos um nível de '{fator}' tem média "
                      f"de '{resposta}' diferente dos demais. Use as comparações "
                      "múltiplas para identificar quais níveis diferem."))
    else:
        itens.append(("interpretacao",
                      f"Como p = {fmt_p(p_valor)} ≥ α = {fmt(alfa, 2)}, não se rejeita "
                      f"H₀: não há evidência de diferença entre as médias dos níveis "
                      f"de '{fator}'."))
    itens.append(("aviso", "Pressupostos da ANOVA: independência, normalidade dos "
                           "resíduos e igualdade de variâncias entre os níveis "
                           "(verifique com o teste de igualdade de variâncias e o "
                           "gráfico de resíduos)."))

    if comparacao == "tukey":
        itens += _tukey(dados, alfa)
    elif comparacao == "fisher":
        itens += _fisher_lsd(grupos, mse, gl_erro, alfa)
    elif comparacao == "dunnett":
        itens += _dunnett(grupos, nivel_controle, alfa)

    return ResultadoComposto(
        titulo=f"ANOVA de 1 fator: {resposta} × {fator}",
        itens=itens,
        dados={"residuos": np.asarray(modelo.resid),
               "ajustados": np.asarray(modelo.fittedvalues),
               "grupos": grupos, "fator": fator, "resposta": resposta},
    )


def _tukey(dados: pd.DataFrame, alfa: float) -> list[tuple]:
    from app.reports.formatacao import fmt

    res = pairwise_tukeyhsd(dados["y"], dados["f0"], alpha=alfa)
    linhas = []
    for (g1, g2, dif, p_adj, lo, hi, rejeita) in res.summary().data[1:]:
        linhas.append([f"{g1} − {g2}", fmt(float(dif)), fmt(float(lo)),
                       fmt(float(hi)), fmt(float(p_adj), 3),
                       "difere" if rejeita else "não difere"])
    return [
        ("subtitulo", f"Comparações múltiplas de Tukey (confiança conjunta "
                      f"{fmt(100 * (1 - alfa), 0)}%)"),
        ("tabela", ["par de níveis", "diferença", "IC inferior", "IC superior",
                    "p ajustado", "conclusão"], linhas),
        ("nota", "Pares cujo IC não contém 0 (p ajustado < α) têm médias "
                 "significativamente diferentes, controlando o erro conjunto."),
    ]


def _fisher_lsd(grupos: dict[str, np.ndarray], mse: float, gl_erro: int,
                alfa: float) -> list[tuple]:
    from app.reports.formatacao import fmt, fmt_p

    niveis = list(grupos)
    linhas = []
    t_crit = stats.t.ppf(1 - alfa / 2, gl_erro)
    for i in range(len(niveis)):
        for j in range(i + 1, len(niveis)):
            xi, xj = grupos[niveis[i]], grupos[niveis[j]]
            dif = float(np.mean(xi) - np.mean(xj))
            ep = np.sqrt(mse * (1 / len(xi) + 1 / len(xj)))
            t = dif / ep
            p = 2 * stats.t.sf(abs(t), gl_erro)
            linhas.append([f"{niveis[i]} − {niveis[j]}", fmt(dif),
                           fmt(dif - t_crit * ep), fmt(dif + t_crit * ep),
                           fmt_p(p), "difere" if p < alfa else "não difere"])
    return [
        ("subtitulo", "Comparações de Fisher (LSD, sem ajuste de erro conjunto)"),
        ("tabela", ["par de níveis", "diferença", "IC inferior", "IC superior",
                    "p-valor", "conclusão"], linhas),
        ("aviso", "O método de Fisher não controla a taxa de erro conjunta; com "
                  "muitos pares, prefira Tukey."),
    ]


def _dunnett(grupos: dict[str, np.ndarray], controle: str | None,
             alfa: float) -> list[tuple]:
    from app.reports.formatacao import fmt, fmt_p

    if controle is None or controle not in grupos:
        raise ErroAnalise("Para o método de Dunnett, informe o nível de controle "
                          f"(níveis disponíveis: {', '.join(grupos)}).")
    tratamentos = [nivel for nivel in grupos if nivel != controle]
    res = stats.dunnett(*[grupos[t] for t in tratamentos], control=grupos[controle])
    ic = res.confidence_interval(confidence_level=1 - alfa)
    linhas = []
    for k, trat in enumerate(tratamentos):
        dif = float(np.mean(grupos[trat]) - np.mean(grupos[controle]))
        linhas.append([f"{trat} − {controle}", fmt(dif), fmt(float(ic.low[k])),
                       fmt(float(ic.high[k])), fmt_p(float(res.pvalue[k])),
                       "difere" if res.pvalue[k] < alfa else "não difere"])
    return [
        ("subtitulo", f"Comparações de Dunnett contra o controle '{controle}'"),
        ("tabela", ["tratamento − controle", "diferença", "IC inferior",
                    "IC superior", "p ajustado", "conclusão"], linhas),
    ]


def anova_2vias(df: pd.DataFrame, resposta: str, fator_a: str, fator_b: str,
                interacao: bool = True, alfa: float = 0.05) -> ResultadoComposto:
    from app.reports.formatacao import fmt, fmt_p

    dados = _empilhar(df, resposta, [fator_a, fator_b])
    formula = "y ~ C(f0) * C(f1)" if interacao else "y ~ C(f0) + C(f1)"
    modelo = smf.ols(formula, data=dados).fit()
    tabela = sm.stats.anova_lm(modelo, typ=2)

    rotulos = {"C(f0)": fator_a, "C(f1)": fator_b,
               "C(f0):C(f1)": f"{fator_a} × {fator_b} (interação)",
               "Residual": "erro"}
    linhas = []
    for indice, linha in tabela.iterrows():
        linhas.append([
            rotulos.get(str(indice), str(indice)), int(linha["df"]),
            fmt(float(linha["sum_sq"])),
            fmt(float(linha["sum_sq"]) / linha["df"]),
            fmt(float(linha["F"])) if np.isfinite(linha.get("F", np.nan)) else "",
            fmt_p(float(linha["PR(>F)"])) if np.isfinite(linha.get("PR(>F)", np.nan)) else "",
        ])

    itens: list[tuple] = [
        ("nota", "Para cada fonte, H₀: o efeito é nulo; H₁: o efeito existe."),
        ("tabela", ["fonte", "GL", "SQ", "QM", "F", "p-valor"], linhas),
        ("nota", f"R² = {fmt(100 * modelo.rsquared, 2)}%  •  n = {len(dados)}"),
    ]
    conclusoes = []
    for indice in tabela.index:
        if str(indice) == "Residual":
            continue
        p = float(tabela.loc[indice, "PR(>F)"])
        nome = rotulos.get(str(indice), str(indice))
        efeito = "significativo" if p < alfa else "não significativo"
        conclusoes.append(f"{nome}: p = {fmt_p(p)} → {efeito}")
    itens.append(("interpretacao",
                  f"Ao nível α = {fmt(alfa, 2)}: " + "; ".join(conclusoes) + ". "
                  + ("Se a interação for significativa, interprete os fatores "
                     "principais com cautela e examine o gráfico de interação."
                     if interacao else "")))

    # contagem por célula para verificação de balanceamento
    contagem = dados.groupby(["f0", "f1"], observed=True).size()
    if contagem.nunique() > 1:
        itens.append(("aviso", "O plano é desbalanceado (células com tamanhos "
                               "diferentes); as somas de quadrados usadas são do "
                               "tipo II."))

    medias_int = dados.groupby(["f0", "f1"], observed=True)["y"].mean()
    return ResultadoComposto(
        titulo=f"ANOVA de 2 fatores: {resposta} × {fator_a}, {fator_b}",
        itens=itens,
        dados={"residuos": np.asarray(modelo.resid),
               "ajustados": np.asarray(modelo.fittedvalues),
               "medias_interacao": medias_int, "fator_a": fator_a,
               "fator_b": fator_b, "resposta": resposta,
               "grupos_a": _grupos(dados, "f0"), "grupos_b": _grupos(dados, "f1")},
    )


def variancias_grupos(df: pd.DataFrame, resposta: str, fator: str,
                      metodo: str = "levene", alfa: float = 0.05) -> ResultadoTeste:
    """Igualdade de variâncias entre níveis: Levene (robusto) ou Bartlett."""
    dados = _empilhar(df, resposta, [fator])
    grupos = _grupos(dados)
    if len(grupos) < 2:
        raise ErroAnalise(f"O fator '{fator}' precisa de pelo menos 2 níveis.")
    amostras = list(grupos.values())
    if metodo == "levene":
        estat, p = stats.levene(*amostras, center="median")
        nome, rotulo = "W", "Levene (robusto, centrado na mediana)"
        avisos = []
    elif metodo == "bartlett":
        estat, p = stats.bartlett(*amostras)
        nome, rotulo = "χ²", "Bartlett"
        avisos = ["O teste de Bartlett é sensível à não normalidade; com dúvida "
                  "sobre normalidade, prefira Levene."]
    else:
        raise ErroAnalise("Método inválido: use 'levene' ou 'bartlett'.")

    return ResultadoTeste(
        titulo=f"Igualdade de variâncias ({rotulo}): {resposta} × {fator}",
        h0=f"H₀: as variâncias de '{resposta}' são iguais em todos os níveis de '{fator}'",
        h1="H₁: pelo menos um nível tem variância diferente",
        nome_estatistica=nome,
        estatistica=float(estat),
        p_valor=float(p),
        alfa=alfa,
        conclusao_h1=f"as variâncias de '{resposta}' diferem entre os níveis de '{fator}'",
        amostras=[{"amostra": nivel, "n": len(x),
                   "desvio-padrão": float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"),
                   "variância": float(np.var(x, ddof=1)) if len(x) > 1 else float("nan")}
                  for nivel, x in grupos.items()],
        avisos=avisos,
    )
