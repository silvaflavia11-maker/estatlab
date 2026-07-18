"""Utilidades compartilhadas do motor estatístico."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .resultados import ErroAnalise

# Mapeamento das alternativas em PT-BR para as convenções das bibliotecas.
ALT_SCIPY = {"bilateral": "two-sided", "menor": "less", "maior": "greater"}
ALT_STATSMODELS = {"bilateral": "two-sided", "menor": "smaller", "maior": "larger"}

SIMBOLO_ALT = {"bilateral": "≠", "menor": "<", "maior": ">"}


def validar_alternativa(alternativa: str) -> str:
    if alternativa not in ALT_SCIPY:
        raise ErroAnalise(
            f"Alternativa inválida: {alternativa!r}. Use 'bilateral', 'menor' ou 'maior'."
        )
    return alternativa


def limpar_numerica(dados, nome: str, n_minimo: int = 2) -> np.ndarray:
    """Converte para array float sem NaN; valida tamanho mínimo."""
    serie = pd.to_numeric(pd.Series(dados), errors="coerce")
    limpa = serie.dropna().to_numpy(dtype=float)
    if limpa.size < n_minimo:
        raise ErroAnalise(
            f"A coluna '{nome}' tem apenas {limpa.size} valor(es) numérico(s) válido(s); "
            f"esta análise exige pelo menos {n_minimo}."
        )
    return limpa


def contar_ausentes(dados) -> int:
    serie = pd.to_numeric(pd.Series(dados), errors="coerce")
    return int(serie.isna().sum())


def resumo_amostra(nome: str, x: np.ndarray) -> dict:
    """Linha-resumo padrão usada nas tabelas de amostras dos testes."""
    return {
        "amostra": nome,
        "n": int(x.size),
        "média": float(np.mean(x)),
        "desvio-padrão": float(np.std(x, ddof=1)),
        "EP da média": float(np.std(x, ddof=1) / np.sqrt(x.size)),
    }
