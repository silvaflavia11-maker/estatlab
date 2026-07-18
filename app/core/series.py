"""Séries temporais: tendência, decomposição, suavizações, ACF/PACF/CCF e
ARIMA (statsmodels)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .resultados import ErroAnalise, ResultadoComposto
from .util import limpar_numerica


def _medidas_erro(y: np.ndarray, ajustado: np.ndarray) -> list[list[str]]:
    from app.reports.formatacao import fmt

    validos = ~np.isnan(ajustado)
    erro = y[validos] - ajustado[validos]
    mad = float(np.mean(np.abs(erro)))
    msd = float(np.mean(erro**2))
    com_y = y[validos] != 0
    mape = float(np.mean(np.abs(erro[com_y] / y[validos][com_y])) * 100)
    return [["MAPE (%)", fmt(mape, 2)], ["MAD", fmt(mad)], ["MSD", fmt(msd)]]


def tendencia(dados, coluna: str, modelo: str = "linear",
              horizonte: int = 6) -> ResultadoComposto:
    from app.reports.formatacao import fmt

    y = limpar_numerica(dados, coluna, n_minimo=8)
    t = np.arange(1, y.size + 1, dtype=float)
    t_futuro = np.arange(y.size + 1, y.size + horizonte + 1, dtype=float)

    if modelo == "linear":
        matriz = sm.add_constant(t)
        ajuste = sm.OLS(y, matriz).fit()
        ajustado = np.asarray(ajuste.fittedvalues)
        previsao = ajuste.predict(sm.add_constant(t_futuro))
        b = ajuste.params
        equacao = f"y = {fmt(float(b[0]))} + {fmt(float(b[1]))}·t"
    elif modelo == "quadratico":
        matriz = sm.add_constant(np.column_stack([t, t**2]))
        ajuste = sm.OLS(y, matriz).fit()
        ajustado = np.asarray(ajuste.fittedvalues)
        previsao = ajuste.predict(
            sm.add_constant(np.column_stack([t_futuro, t_futuro**2])))
        b = ajuste.params
        equacao = (f"y = {fmt(float(b[0]))} + {fmt(float(b[1]))}·t "
                   f"+ {fmt(float(b[2]))}·t²")
    elif modelo == "exponencial":
        if np.any(y <= 0):
            raise ErroAnalise("Tendência exponencial exige valores positivos.")
        ajuste = sm.OLS(np.log(y), sm.add_constant(t)).fit()
        ajustado = np.exp(np.asarray(ajuste.fittedvalues))
        previsao = np.exp(ajuste.predict(sm.add_constant(t_futuro)))
        b = ajuste.params
        equacao = f"y = {fmt(float(np.exp(b[0])))}·({fmt(float(np.exp(b[1])), 4)})^t"
    else:
        raise ErroAnalise("Modelo inválido: linear, quadratico ou exponencial.")

    linhas_prev = [[str(int(tf)), fmt(float(p))]
                   for tf, p in zip(t_futuro, previsao)]
    return ResultadoComposto(
        titulo=f"Análise de tendência ({modelo}): {coluna}",
        itens=[
            ("nota", "Equação ajustada: " + equacao),
            ("tabela", ["medida de acurácia", "valor"], _medidas_erro(y, ajustado)),
            ("subtitulo", f"Previsões ({horizonte} períodos)"),
            ("tabela", ["período", "previsão"], linhas_prev),
            ("interpretacao", "As medidas MAPE/MAD/MSD comparam modelos de "
                              "tendência: menores valores indicam melhor ajuste. "
                              "Previsões de tendência valem para horizontes "
                              "curtos."),
        ],
        dados={"y": y, "ajustado": ajustado, "previsao": np.asarray(previsao),
               "coluna": coluna},
    )


def decomposicao(dados, coluna: str, periodo: int,
                 modelo: str = "aditivo") -> ResultadoComposto:
    from statsmodels.tsa.seasonal import seasonal_decompose

    from app.reports.formatacao import fmt

    y = limpar_numerica(dados, coluna, n_minimo=2 * periodo + 2)
    if periodo < 2:
        raise ErroAnalise("O período sazonal deve ser pelo menos 2.")
    tipo = "additive" if modelo == "aditivo" else "multiplicative"
    if tipo == "multiplicative" and np.any(y <= 0):
        raise ErroAnalise("Decomposição multiplicativa exige valores positivos.")
    resultado = seasonal_decompose(y, model=tipo, period=periodo)

    indices = np.arange(periodo)
    sazonal_1ciclo = resultado.seasonal[:periodo]
    linhas = [[str(i + 1), fmt(float(s))]
              for i, s in zip(indices, sazonal_1ciclo)]
    return ResultadoComposto(
        titulo=f"Decomposição ({modelo}, período {periodo}): {coluna}",
        itens=[
            ("subtitulo", "Índices sazonais (um ciclo)"),
            ("tabela", ["posição no ciclo", "índice sazonal"], linhas),
            ("interpretacao",
             ("No modelo aditivo, o índice é somado à tendência; "
              if modelo == "aditivo" else
              "No modelo multiplicativo, o índice multiplica a tendência; ")
             + "posições com índices maiores são os picos sazonais."),
        ],
        dados={"y": y, "tendencia": np.asarray(resultado.trend),
               "sazonal": np.asarray(resultado.seasonal),
               "residuo": np.asarray(resultado.resid), "coluna": coluna},
    )


def suavizacao(dados, coluna: str, metodo: str = "media_movel",
               parametro: float = 3, periodo: int = 12,
               horizonte: int = 6) -> ResultadoComposto:
    """``metodo``: media_movel (parametro = janela), ses (parametro = α),
    holt (dupla) ou winters (tripla, exige ``periodo``)."""
    from app.reports.formatacao import fmt

    y = limpar_numerica(dados, coluna, n_minimo=8)
    serie = pd.Series(y)

    if metodo == "media_movel":
        janela = int(parametro)
        if janela < 2:
            raise ErroAnalise("A janela deve ser pelo menos 2.")
        ajustado = serie.rolling(janela).mean().to_numpy()
        previsao = np.full(horizonte, float(np.mean(y[-janela:])))
        descricao = f"média móvel de janela {janela}"
    else:
        from statsmodels.tsa.holtwinters import (
            ExponentialSmoothing,
            Holt,
            SimpleExpSmoothing,
        )

        if metodo == "ses":
            modelo = (SimpleExpSmoothing(serie, initialization_method="estimated")
                      .fit(smoothing_level=parametro if 0 < parametro < 1 else None))
            descricao = (f"suavização exponencial simples "
                         f"(α = {fmt(float(modelo.params['smoothing_level']), 3)})")
        elif metodo == "holt":
            modelo = Holt(serie, initialization_method="estimated").fit()
            descricao = "suavização exponencial dupla (Holt)"
        elif metodo == "winters":
            if y.size < 2 * periodo + 2:
                raise ErroAnalise(f"Winters com período {periodo} exige pelo "
                                  f"menos {2 * periodo + 2} observações.")
            modelo = ExponentialSmoothing(
                serie, trend="add", seasonal="add", seasonal_periods=periodo,
                initialization_method="estimated").fit()
            descricao = f"método de Winters (aditivo, período {periodo})"
        else:
            raise ErroAnalise("Método inválido.")
        ajustado = np.asarray(modelo.fittedvalues)
        previsao = np.asarray(modelo.forecast(horizonte))

    linhas_prev = [[str(y.size + i + 1), fmt(float(p))]
                   for i, p in enumerate(previsao)]
    return ResultadoComposto(
        titulo=f"Suavização — {descricao}: {coluna}",
        itens=[
            ("tabela", ["medida de acurácia", "valor"], _medidas_erro(y, ajustado)),
            ("subtitulo", f"Previsões ({horizonte} períodos)"),
            ("tabela", ["período", "previsão"], linhas_prev),
            ("interpretacao", "Compare métodos pelas medidas de acurácia. "
                              "Use Holt quando há tendência e Winters quando há "
                              "tendência e sazonalidade."),
        ],
        dados={"y": y, "ajustado": ajustado, "previsao": previsao,
               "coluna": coluna},
    )


def autocorrelacao(dados, coluna: str, dados2=None, coluna2: str | None = None,
                   defasagens: int = 20) -> ResultadoComposto:
    """ACF e PACF de uma série; CCF quando uma segunda coluna é dada."""
    from statsmodels.tsa.stattools import acf, ccf, pacf

    from app.reports.formatacao import fmt

    y = limpar_numerica(dados, coluna, n_minimo=10)
    defasagens = int(min(defasagens, y.size // 2 - 1))
    limite = 1.96 / np.sqrt(y.size)

    if dados2 is not None:
        y2 = limpar_numerica(dados2, coluna2, n_minimo=10)
        n = min(y.size, y2.size)
        valores_ccf = ccf(y[:n], y2[:n], adjusted=False)[:defasagens + 1]
        linhas = [[str(k), fmt(float(v), 3),
                   "significativa" if abs(v) > limite else ""]
                  for k, v in enumerate(valores_ccf)]
        return ResultadoComposto(
            titulo=f"Correlação cruzada: {coluna} × {coluna2}",
            itens=[("tabela", ["defasagem", "CCF", ""], linhas),
                   ("interpretacao",
                    f"Valores além de ±{fmt(limite, 3)} (≈ 2/√n) indicam "
                    "correlação cruzada significativa naquela defasagem.")],
            dados={"tipo": "ccf", "valores": valores_ccf, "limite": limite,
                   "coluna": f"{coluna} × {coluna2}"},
        )

    valores_acf = acf(y, nlags=defasagens)
    valores_pacf = pacf(y, nlags=defasagens)
    linhas = [[str(k), fmt(float(a), 3), fmt(float(p), 3)]
              for k, (a, p) in enumerate(zip(valores_acf, valores_pacf))]
    significativas = [k for k in range(1, defasagens + 1)
                      if abs(valores_acf[k]) > limite]
    return ResultadoComposto(
        titulo=f"Autocorrelação (ACF/PACF): {coluna}",
        itens=[
            ("tabela", ["defasagem", "ACF", "PACF"], linhas),
            ("interpretacao",
             (f"Defasagens com ACF significativa (±{fmt(limite, 3)}): "
              + ", ".join(map(str, significativas)) + "."
              if significativas else
              "Nenhuma autocorrelação significativa: a série se comporta como "
              "ruído aleatório.")
             + " ACF decai lentamente em séries não estacionárias; PACF ajuda a "
               "escolher a ordem AR do modelo ARIMA."),
        ],
        dados={"tipo": "acf", "acf": valores_acf, "pacf": valores_pacf,
               "limite": limite, "coluna": coluna},
    )


def arima(dados, coluna: str, p: int, d: int, q: int,
          horizonte: int = 6, alfa: float = 0.05) -> ResultadoComposto:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.arima.model import ARIMA

    from app.reports.formatacao import fmt, fmt_p

    y = limpar_numerica(dados, coluna, n_minimo=15)
    if max(p, q) > 5 or d > 2:
        raise ErroAnalise("Use p, q ≤ 5 e d ≤ 2.")
    try:
        modelo = ARIMA(y, order=(p, d, q)).fit()
    except Exception as erro:
        raise ErroAnalise(f"O ajuste ARIMA({p},{d},{q}) falhou: {erro}")

    linhas_coef = []
    for nome in modelo.param_names:
        if nome == "sigma2":
            continue
        idx = modelo.param_names.index(nome)
        linhas_coef.append([nome, fmt(float(modelo.params[idx])),
                            fmt(float(modelo.bse[idx])),
                            fmt_p(float(modelo.pvalues[idx]))])

    lb = acorr_ljungbox(modelo.resid, lags=[min(10, y.size // 3)])
    p_lb = float(lb["lb_pvalue"].iloc[0])
    previsao = modelo.get_forecast(horizonte)
    media_prev = np.asarray(previsao.predicted_mean)
    ic_prev = previsao.conf_int(alpha=alfa)
    ic_prev = np.asarray(ic_prev)
    linhas_prev = [[str(y.size + i + 1), fmt(float(media_prev[i])),
                    f"({fmt(float(ic_prev[i][0]))}; {fmt(float(ic_prev[i][1]))})"]
                   for i in range(horizonte)]

    itens: list[tuple] = [
        ("subtitulo", "Coeficientes"),
        ("tabela", ["termo", "coeficiente", "EP", "p-valor"], linhas_coef or
         [["(somente diferenciação)", "—", "—", "—"]]),
        ("nota", f"AIC = {fmt(float(modelo.aic), 2)}  •  "
                 f"BIC = {fmt(float(modelo.bic), 2)}  •  n = {y.size}"),
        ("subtitulo", f"Previsões ({horizonte} períodos)"),
        ("tabela", ["período", "previsão",
                    f"IC {fmt(100 * (1 - alfa), 0)}%"], linhas_prev),
        ("interpretacao",
         f"Diagnóstico de Ljung-Box dos resíduos: p = {fmt_p(p_lb)} — "
         + ("os resíduos se comportam como ruído branco (modelo adequado)."
            if p_lb >= 0.05 else
            "há autocorrelação remanescente nos resíduos; tente outras ordens "
            "(p, d, q).")),
    ]
    return ResultadoComposto(
        titulo=f"ARIMA({p},{d},{q}): {coluna}",
        itens=itens,
        dados={"y": y, "ajustado": np.asarray(modelo.fittedvalues),
               "previsao": media_prev, "ic_previsao": ic_prev,
               "residuos": np.asarray(modelo.resid), "coluna": coluna},
    )
