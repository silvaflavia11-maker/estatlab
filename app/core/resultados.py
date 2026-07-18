"""Estruturas de resultado das análises.

As funções de ``app.core`` recebem dados brutos (arrays/Series) e devolvem
estas dataclasses. A renderização (HTML, interpretação didática formatada)
fica em ``app.reports`` — aqui só há dados e os textos semânticos
(hipóteses e conclusão) que dependem da análise.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class ErroAnalise(Exception):
    """Erro de validação com mensagem amigável, exibida ao usuário."""


@dataclass
class ResultadoTeste:
    """Resultado de um teste de hipóteses."""

    titulo: str
    h0: str
    h1: str
    nome_estatistica: str
    estatistica: float
    p_valor: float
    alfa: float
    # Complemento da frase "há evidência de que ..." usado na conclusão.
    conclusao_h1: str
    gl: float | None = None
    ic: tuple[float, float] | None = None
    nivel_confianca: float | None = None
    descricao_ic: str = ""
    # Uma linha por amostra: {"nome":…, "n":…, "média":…, "desvio-padrão":…}
    amostras: list[dict] = field(default_factory=list)
    # Pares rótulo → valor exibidos como informações adicionais.
    detalhes: dict = field(default_factory=dict)
    # Avisos sobre pressupostos (normalidade, tamanho de amostra etc.).
    avisos: list[str] = field(default_factory=list)


@dataclass
class ResultadoDescritiva:
    coluna: str
    n: int
    ausentes: int
    media: float
    ep_media: float
    dp: float
    variancia: float
    minimo: float
    q1: float
    mediana: float
    q3: float
    maximo: float
    amplitude: float
    aiq: float
    assimetria: float
    curtose: float
    ic_media: tuple[float, float]
    nivel_confianca: float


@dataclass
class ResultadoTabela:
    """Resultado tabular genérico (matriz de correlação, covariância...)."""

    titulo: str
    cabecalhos: list[str]
    linhas: list[list[str]]
    notas: list[str] = field(default_factory=list)


@dataclass
class ResultadoComposto:
    """Análise com várias seções (ANOVA, regressão…).

    ``itens`` é uma sequência ordenada de tuplas:
      ("subtitulo", texto)
      ("tabela", cabecalhos, linhas)
      ("interpretacao", texto)   → destaque de decisão/conclusão
      ("nota", texto)            → observação simples
      ("aviso", texto)           → alerta de pressuposto
    ``dados`` carrega arrays para gráficos (resíduos, ajustados…), não renderizados.
    """

    titulo: str
    itens: list[tuple]
    dados: dict = field(default_factory=dict)
