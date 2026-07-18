"""Estatística descritiva."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .resultados import ResultadoDescritiva
from .util import contar_ausentes, limpar_numerica


def descritiva(dados, coluna: str, nivel_confianca: float = 0.95) -> ResultadoDescritiva:
    ausentes = contar_ausentes(dados)
    x = limpar_numerica(dados, coluna, n_minimo=2)
    n = x.size
    serie = pd.Series(x)

    media = float(np.mean(x))
    dp = float(np.std(x, ddof=1))
    ep = dp / np.sqrt(n)
    t_crit = stats.t.ppf(0.5 + nivel_confianca / 2, df=n - 1)
    q1, mediana, q3 = np.percentile(x, [25, 50, 75])

    return ResultadoDescritiva(
        coluna=coluna,
        n=n,
        ausentes=ausentes,
        media=media,
        ep_media=float(ep),
        dp=dp,
        variancia=dp**2,
        minimo=float(np.min(x)),
        q1=float(q1),
        mediana=float(mediana),
        q3=float(q3),
        maximo=float(np.max(x)),
        amplitude=float(np.max(x) - np.min(x)),
        aiq=float(q3 - q1),
        # skew/kurt do pandas: estimadores ajustados (mesma convenção do Minitab)
        assimetria=float(serie.skew()),
        curtose=float(serie.kurt()),
        ic_media=(media - t_crit * ep, media + t_crit * ep),
        nivel_confianca=nivel_confianca,
    )
