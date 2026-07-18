"""Capabilidade de processo.

- Normal: σ dentro (MR-barra/d₂ para dados individuais; R-barra/d₂ para
  subgrupos) e σ geral (desvio-padrão amostral) → Cp/Cpk e Pp/Ppk.
- Não normal: transformações Box-Cox e Johnson (família SU) aplicadas aos
  dados e aos limites de especificação, seguidas de capabilidade normal.
- Identificação de distribuição: ajusta candidatas e ordena pela estatística
  de Anderson-Darling (menor = melhor ajuste).
- Atributos (binomial): % defeituosa, PPM e IC exato.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .cep import D2
from .resultados import ErroAnalise, ResultadoComposto
from .util import limpar_numerica

DISTRIBUICOES_ID = {
    "Normal": stats.norm,
    "Lognormal": stats.lognorm,
    "Exponencial": stats.expon,
    "Weibull": stats.weibull_min,
    "Gama": stats.gamma,
    "Logística": stats.logistic,
    "Menor valor extremo": stats.gumbel_l,
    "Maior valor extremo": stats.gumbel_r,
}


def _sigma_dentro(x: np.ndarray, tamanho_subgrupo: int) -> tuple[float, str]:
    if tamanho_subgrupo <= 1:
        return (float(np.mean(np.abs(np.diff(x))) / D2[2]),
                "MR-barra/d₂ (amplitudes móveis)")
    if tamanho_subgrupo > 10:
        raise ErroAnalise("Para σ dentro com subgrupos, use tamanho ≤ 10.")
    completos = (x.size // tamanho_subgrupo) * tamanho_subgrupo
    matriz = x[:completos].reshape(-1, tamanho_subgrupo)
    if matriz.shape[0] < 2:
        raise ErroAnalise("Menos de 2 subgrupos completos para estimar σ dentro.")
    return (float(np.mean(np.ptp(matriz, axis=1)) / D2[tamanho_subgrupo]),
            f"R-barra/d₂ (subgrupos de {tamanho_subgrupo})")


def capabilidade_normal(dados, coluna: str, lie: float | None, lse: float | None,
                        alvo: float | None = None, tamanho_subgrupo: int = 1,
                        transformacao: str | None = None,
                        lambda_boxcox: float | None = None) -> ResultadoComposto:
    """``transformacao``: None, 'boxcox' ou 'johnson'."""
    from app.reports.formatacao import fmt

    x = limpar_numerica(dados, coluna, n_minimo=10)
    if lie is None and lse is None:
        raise ErroAnalise("Informe pelo menos um limite de especificação (LIE/LSE).")
    if lie is not None and lse is not None and lie >= lse:
        raise ErroAnalise("O LIE deve ser menor que o LSE.")

    notas_transf: list[tuple] = []
    x_orig, lie_orig, lse_orig = x.copy(), lie, lse
    if transformacao == "boxcox":
        if np.any(x <= 0):
            raise ErroAnalise("Box-Cox exige dados positivos.")
        if lambda_boxcox is None:
            x, lambda_boxcox = stats.boxcox(x)
        else:
            x = stats.boxcox(x_orig, lmbda=lambda_boxcox)
        transf = (lambda v: (v**lambda_boxcox - 1) / lambda_boxcox
                  if abs(lambda_boxcox) > 1e-12 else np.log(v))
        lie = transf(lie) if lie is not None and lie > 0 else (
            None if lie is None else transf(max(lie, 1e-12)))
        lse = transf(lse) if lse is not None else None
        notas_transf.append(("nota", f"Transformação Box-Cox aplicada com "
                                     f"λ = {fmt(float(lambda_boxcox), 3)}; dados e "
                                     "especificações transformados."))
    elif transformacao == "johnson":
        parametros = stats.johnsonsu.fit(x)
        dist = stats.johnsonsu(*parametros)
        transf = lambda v: stats.norm.ppf(np.clip(dist.cdf(v), 1e-12, 1 - 1e-12))
        x = transf(x)
        lie = float(transf(lie)) if lie is not None else None
        lse = float(transf(lse)) if lse is not None else None
        notas_transf.append(("nota", "Transformação de Johnson (família SU) "
                                     "aplicada; dados e especificações "
                                     "transformados para a escala normal."))

    mu = float(np.mean(x))
    sigma_g = float(np.std(x, ddof=1))
    sigma_d, metodo_dentro = _sigma_dentro(x, tamanho_subgrupo)
    if sigma_d <= 0 or sigma_g <= 0:
        raise ErroAnalise("Variação nula nos dados; capabilidade não definida.")

    def indices(sigma: float) -> dict[str, float | None]:
        cpu = (lse - mu) / (3 * sigma) if lse is not None else None
        cpl = (mu - lie) / (3 * sigma) if lie is not None else None
        cp = ((lse - lie) / (6 * sigma)
              if lie is not None and lse is not None else None)
        cpk = min(v for v in (cpu, cpl) if v is not None)
        return {"cp": cp, "cpu": cpu, "cpl": cpl, "cpk": cpk}

    dentro = indices(sigma_d)
    geral = indices(sigma_g)
    cpm = None
    if alvo is not None and lie is not None and lse is not None:
        tau = np.sqrt(sigma_g**2 + (mu - alvo) ** 2)
        cpm = float((lse - lie) / (6 * tau))

    def ppm_esperado(sigma: float) -> float:
        p = 0.0
        if lse is not None:
            p += stats.norm.sf(lse, mu, sigma)
        if lie is not None:
            p += stats.norm.cdf(lie, mu, sigma)
        return float(p * 1e6)

    fora = 0
    if lse is not None:
        fora += int(np.sum(x > lse))
    if lie is not None:
        fora += int(np.sum(x < lie))
    ppm_obs = 1e6 * fora / x.size

    def linha(nome, d, g):
        return [nome, fmt(d, 2) if d is not None else "—",
                fmt(g, 2) if g is not None else "—"]

    itens: list[tuple] = list(notas_transf)
    itens += [
        ("tabela", ["dado do processo", "valor"],
         [["n", str(x.size)], ["média", fmt(mu)],
          ["σ dentro (curto prazo)", fmt(sigma_d)],
          ["σ geral (longo prazo)", fmt(sigma_g)],
          ["LIE", fmt(lie_orig) if lie_orig is not None else "—"],
          ["LSE", fmt(lse_orig) if lse_orig is not None else "—"],
          ["alvo", fmt(alvo) if alvo is not None else "—"]]),
        ("subtitulo", "Índices de capabilidade"),
        ("tabela", ["índice", "dentro (Cp…)", "geral (Pp…)"],
         [linha("Cp / Pp", dentro["cp"], geral["cp"]),
          linha("CPU", dentro["cpu"], geral["cpu"]),
          linha("CPL", dentro["cpl"], geral["cpl"]),
          linha("Cpk / Ppk", dentro["cpk"], geral["cpk"])]
         + ([["Cpm", "—", fmt(cpm, 2)]] if cpm is not None else [])),
        ("subtitulo", "Desempenho (partes por milhão)"),
        ("tabela", ["medida", "valor"],
         [["PPM observado (fora da especificação)", fmt(ppm_obs, 1)],
          ["PPM esperado (dentro)", fmt(ppm_esperado(sigma_d), 1)],
          ["PPM esperado (geral)", fmt(ppm_esperado(sigma_g), 1)]]),
        ("nota", f"σ dentro estimado por {metodo_dentro}."),
    ]

    cpk = dentro["cpk"]
    if cpk >= 1.33:
        avaliacao = "capaz (referência usual: Cpk ≥ 1,33)"
    elif cpk >= 1.0:
        avaliacao = "marginalmente capaz (1,00 ≤ Cpk < 1,33)"
    else:
        avaliacao = "não capaz (Cpk < 1,00)"
    itens.append(("interpretacao",
                  f"Cpk = {fmt(cpk, 2)}: o processo é {avaliacao}. "
                  "Cp mede o potencial (dispersão); Cpk desconta o descentramento; "
                  "Pp/Ppk usam a variação total de longo prazo."))
    if abs(dentro["cpk"] - geral["cpk"]) > 0.2:
        itens.append(("aviso", "Diferença relevante entre Cpk e Ppk: há variação "
                               "entre subgrupos/ao longo do tempo além da variação "
                               "de curto prazo (processo instável ou com desvios)."))
    if transformacao is None:
        ad, p_ad = _ad_normal(x)
        if p_ad < 0.05:
            itens.append(("aviso", "Os dados rejeitam normalidade (AD, p < 0,05). "
                                   "Considere a capabilidade não normal "
                                   "(transformação Box-Cox/Johnson ou identificação "
                                   "de distribuição)."))

    return ResultadoComposto(
        titulo=f"Capabilidade de processo: {coluna}",
        itens=itens,
        dados={"x": x, "x_original": x_orig, "mu": mu, "sigma_d": sigma_d,
               "sigma_g": sigma_g, "lie": lie, "lse": lse,
               "lie_original": lie_orig, "lse_original": lse_orig,
               "coluna": coluna, "tamanho_subgrupo": tamanho_subgrupo},
    )


def _ad_normal(x: np.ndarray) -> tuple[float, float]:
    from statsmodels.stats.diagnostic import normal_ad

    estat, p = normal_ad(x)
    return float(estat), float(p)


def _estatistica_ad(x: np.ndarray, dist) -> float:
    """Estatística de Anderson-Darling para uma distribuição ajustada."""
    n = x.size
    u = np.clip(dist.cdf(np.sort(x)), 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    return float(-n - np.mean((2 * i - 1) * (np.log(u) + np.log(1 - u[::-1]))))


def identificar_distribuicao(dados, coluna: str) -> ResultadoComposto:
    from app.reports.formatacao import fmt

    x = limpar_numerica(dados, coluna, n_minimo=20)
    linhas = []
    for nome, familia in DISTRIBUICOES_ID.items():
        if nome in ("Lognormal", "Exponencial", "Weibull", "Gama") and np.any(x <= 0):
            linhas.append([nome, "—", "exige dados positivos"])
            continue
        try:
            if nome == "Lognormal":
                parametros = familia.fit(x, floc=0)
            elif nome in ("Exponencial",):
                parametros = familia.fit(x, floc=0)
            elif nome in ("Weibull", "Gama"):
                parametros = familia.fit(x, floc=0)
            else:
                parametros = familia.fit(x)
            ad = _estatistica_ad(x, familia(*parametros))
            parametros_txt = ", ".join(fmt(float(v), 3) for v in parametros)
            linhas.append([nome, fmt(ad, 3), parametros_txt])
        except Exception:
            linhas.append([nome, "—", "falha no ajuste"])

    validas = [linha for linha in linhas if linha[1] != "—"]
    validas.sort(key=lambda linha: float(linha[1].replace(".", "").replace(",", ".")))
    melhor = validas[0][0] if validas else "—"
    return ResultadoComposto(
        titulo=f"Identificação de distribuição individual: {coluna}",
        itens=[
            ("tabela", ["distribuição", "AD (menor = melhor)",
                        "parâmetros ajustados"], validas +
             [linha for linha in linhas if linha[1] == "—"]),
            ("interpretacao",
             f"Melhor ajuste pela estatística de Anderson-Darling: {melhor}. "
             "Confirme visualmente com o gráfico de probabilidade antes de usar "
             "em capabilidade não normal."),
            ("nota", "Ajustes por máxima verossimilhança (localização fixada em 0 "
                     "para distribuições de dados positivos)."),
        ],
    )


def capabilidade_atributos(defeituosos: int, total: int,
                           nivel_confianca: float = 0.95) -> ResultadoComposto:
    from app.reports.formatacao import fmt, fmt_p

    if total <= 0 or defeituosos < 0 or defeituosos > total:
        raise ErroAnalise("Informe 0 ≤ defeituosos ≤ total, com total > 0.")
    p = defeituosos / total
    ic = stats.binomtest(defeituosos, total, 0.5).proportion_ci(
        confidence_level=nivel_confianca, method="exact")
    z_bench = float(stats.norm.ppf(1 - p)) if 0 < p < 1 else float("inf")
    return ResultadoComposto(
        titulo="Capabilidade de processo por atributos (binomial)",
        itens=[
            ("tabela", ["medida", "valor"],
             [["inspecionados", str(total)], ["defeituosos", str(defeituosos)],
              ["% defeituosa", fmt(100 * p, 3) + "%"],
              ["PPM", fmt(1e6 * p, 0)],
              [f"IC {fmt(100 * nivel_confianca, 0)}% da % defeituosa",
               f"({fmt(100 * ic.low, 3)}%; {fmt(100 * ic.high, 3)}%)"],
              ["Z de referência (bench)", fmt(z_bench, 2)]]),
            ("interpretacao",
             f"O processo produz {fmt(100 * p, 2)}% de defeituosos "
             f"({fmt(1e6 * p, 0)} PPM). O Z de referência corresponde ao quantil "
             "normal dessa proporção — quanto maior, melhor."),
        ],
    )
