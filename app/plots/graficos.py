"""Geração de gráficos (matplotlib) da Fase 1.

Cada função recebe dados já extraídos da planilha e devolve uma ``Figure``.
Nenhuma dependência de Qt aqui — a integração fica em ``app.ui``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats

from app.core.resultados import ErroAnalise
from app.core.util import limpar_numerica

COR = "#2e6e80"
COR_DESTAQUE = "#c0563a"

FORMATOS_EXPORTACAO = "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tif);;BMP (*.bmp);;SVG (*.svg)"


def _figura(titulo: str) -> tuple[Figure, "object"]:
    fig = Figure(figsize=(7, 4.6), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title(titulo)
    ax.grid(True, alpha=0.3)
    fig.set_layout_engine("tight")
    return fig, ax


def histograma(dados, coluna: str, classes: int | None = None) -> Figure:
    x = limpar_numerica(dados, coluna)
    fig, ax = _figura(f"Histograma de {coluna}")
    ax.hist(x, bins=classes or "auto", color=COR, edgecolor="white")
    ax.set_xlabel(coluna)
    ax.set_ylabel("frequência")
    return fig


def boxplot(series: list, rotulos: list[str]) -> Figure:
    dados = [limpar_numerica(s, r) for s, r in zip(series, rotulos)]
    fig, ax = _figura("Boxplot de " + ", ".join(rotulos))
    bp = ax.boxplot(dados, tick_labels=rotulos, patch_artist=True)
    for caixa in bp["boxes"]:
        caixa.set_facecolor(COR)
        caixa.set_alpha(0.7)
    ax.set_ylabel("valor")
    return fig


def dotplot(dados, coluna: str) -> Figure:
    x = np.sort(limpar_numerica(dados, coluna))
    # Agrupa em até 60 classes e empilha um ponto por observação.
    n_classes = min(60, max(10, int(np.sqrt(x.size) * 3)))
    contagens, bordas = np.histogram(x, bins=n_classes)
    centros = (bordas[:-1] + bordas[1:]) / 2
    fig, ax = _figura(f"Dotplot de {coluna}")
    for centro, contagem in zip(centros, contagens):
        if contagem:
            ax.plot([centro] * contagem, np.arange(1, contagem + 1), "o",
                    color=COR, markersize=5)
    ax.set_xlabel(coluna)
    ax.set_yticks([])
    ax.set_ylim(bottom=0)
    return fig


def dispersao(dx, dy, col_x: str, col_y: str) -> Figure:
    par = pd.DataFrame({
        "x": pd.to_numeric(pd.Series(dx), errors="coerce"),
        "y": pd.to_numeric(pd.Series(dy), errors="coerce"),
    }).dropna()
    if len(par) < 2:
        raise ErroAnalise(f"O par '{col_x}' × '{col_y}' tem menos de 2 observações completas.")
    fig, ax = _figura(f"Dispersão: {col_y} × {col_x}")
    ax.plot(par["x"], par["y"], "o", color=COR, alpha=0.8)
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    return fig


def matriz_dispersao(df: pd.DataFrame, colunas: list[str]) -> Figure:
    if len(colunas) < 2:
        raise ErroAnalise("Selecione pelo menos 2 colunas numéricas.")
    dados = df[colunas].apply(pd.to_numeric, errors="coerce").dropna()
    if len(dados) < 2:
        raise ErroAnalise("Menos de 2 observações completas nas colunas selecionadas.")
    fig = Figure(figsize=(7.5, 7), dpi=100)
    eixos = fig.subplots(len(colunas), len(colunas))
    pd.plotting.scatter_matrix(dados, ax=eixos, color=COR, alpha=0.75,
                               hist_kwds={"color": COR, "edgecolor": "white"})
    fig.suptitle("Matriz de dispersão")
    return fig


def barras(dados, coluna: str) -> Figure:
    serie = pd.Series(dados).dropna()
    serie = serie[serie.astype(str).str.strip() != ""]
    if serie.empty:
        raise ErroAnalise(f"A coluna '{coluna}' está vazia.")
    contagens = serie.astype(str).value_counts()
    fig, ax = _figura(f"Gráfico de barras de {coluna}")
    ax.bar(contagens.index, contagens.values, color=COR, edgecolor="white")
    ax.set_xlabel(coluna)
    ax.set_ylabel("contagem")
    if max(len(str(v)) for v in contagens.index) > 8 or len(contagens) > 8:
        ax.tick_params(axis="x", rotation=45)
    return fig


def pizza(dados, coluna: str) -> Figure:
    serie = pd.Series(dados).dropna()
    serie = serie[serie.astype(str).str.strip() != ""]
    if serie.empty:
        raise ErroAnalise(f"A coluna '{coluna}' está vazia.")
    contagens = serie.astype(str).value_counts()
    if len(contagens) > 12:
        raise ErroAnalise(
            f"A coluna '{coluna}' tem {len(contagens)} categorias — demais para um "
            "gráfico de pizza. Use o gráfico de barras."
        )
    fig = Figure(figsize=(6.4, 5.2), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title(f"Gráfico de pizza de {coluna}")
    ax.pie(contagens.values, labels=list(contagens.index), autopct="%1.1f%%",
           startangle=90, counterclock=False)
    fig.set_layout_engine("tight")
    return fig


def serie_temporal(dados, coluna: str) -> Figure:
    x = limpar_numerica(dados, coluna)
    fig, ax = _figura(f"Série temporal de {coluna}")
    ax.plot(np.arange(1, x.size + 1), x, "-o", color=COR, markersize=4)
    ax.set_xlabel("índice (ordem dos dados)")
    ax.set_ylabel(coluna)
    return fig


def residuos_4paineis(residuos, ajustados, titulo: str) -> Figure:
    """Painel 2×2: resíduos × ajustados, probabilidade normal, histograma e
    resíduos × ordem — verificação visual dos pressupostos."""
    residuos = np.asarray(residuos, dtype=float)
    ajustados = np.asarray(ajustados, dtype=float)
    fig = Figure(figsize=(8.6, 6.4), dpi=100)
    eixos = fig.subplots(2, 2)
    fig.suptitle(f"Gráficos de resíduos — {titulo}")

    ax = eixos[0][0]
    ax.plot(ajustados, residuos, "o", color=COR, alpha=0.75)
    ax.axhline(0, color=COR_DESTAQUE, linewidth=1)
    ax.set_xlabel("valores ajustados")
    ax.set_ylabel("resíduos")
    ax.set_title("resíduos × ajustados", fontsize=10)

    ax = eixos[0][1]
    (quantis, ordenados), (a, b, r) = stats.probplot(residuos, dist="norm")
    ax.plot(quantis, ordenados, "o", color=COR, alpha=0.75)
    ax.plot(quantis, a * quantis + b, "-", color=COR_DESTAQUE)
    ax.set_xlabel("quantis teóricos")
    ax.set_ylabel("resíduos")
    ax.set_title("probabilidade normal dos resíduos", fontsize=10)

    ax = eixos[1][0]
    ax.hist(residuos, bins="auto", color=COR, edgecolor="white")
    ax.set_xlabel("resíduos")
    ax.set_ylabel("frequência")
    ax.set_title("histograma dos resíduos", fontsize=10)

    ax = eixos[1][1]
    ax.plot(np.arange(1, residuos.size + 1), residuos, "-o", color=COR,
            markersize=3.5)
    ax.axhline(0, color=COR_DESTAQUE, linewidth=1)
    ax.set_xlabel("ordem das observações")
    ax.set_ylabel("resíduos")
    ax.set_title("resíduos × ordem", fontsize=10)

    for linha in eixos:
        for ax in linha:
            ax.grid(True, alpha=0.3)
    fig.set_layout_engine("tight")
    return fig


def efeitos_principais(grupos: dict[str, np.ndarray], fator: str,
                       resposta: str) -> Figure:
    fig, ax = _figura(f"Efeitos principais: {resposta} × {fator}")
    niveis = list(grupos)
    medias = [float(np.mean(grupos[nivel])) for nivel in niveis]
    ax.plot(niveis, medias, "-o", color=COR)
    ax.axhline(float(np.mean(np.concatenate(list(grupos.values())))),
               color=COR_DESTAQUE, linestyle="--", linewidth=1,
               label="média geral")
    ax.set_xlabel(fator)
    ax.set_ylabel(f"média de {resposta}")
    ax.legend()
    return fig


def interacao(medias, fator_a: str, fator_b: str, resposta: str) -> Figure:
    """``medias``: Series com MultiIndex (nível A, nível B) → média."""
    fig, ax = _figura(f"Interação: {fator_a} × {fator_b}")
    tabela = medias.unstack()
    for coluna in tabela.columns:
        ax.plot(tabela.index.astype(str), tabela[coluna], "-o",
                label=f"{fator_b} = {coluna}")
    ax.set_xlabel(fator_a)
    ax.set_ylabel(f"média de {resposta}")
    ax.legend()
    return fig


def distribuicao(curva: dict, titulo: str) -> Figure:
    """Gráfico de pdf/pmf gerado por ``core.distribuicoes.dados_curva``."""
    fig, ax = _figura(titulo)
    x, y = curva["x"], curva["y"]
    if curva["discreta"]:
        ax.vlines(x, 0, y, color=COR, linewidth=3)
        ax.plot(x, y, "o", color=COR)
        if curva["sombra_ate"] is not None:
            marcados = x <= curva["sombra_ate"]
            ax.vlines(x[marcados], 0, y[marcados], color=COR_DESTAQUE, linewidth=3)
        ax.set_ylabel("P(X = x)")
    else:
        ax.plot(x, y, "-", color=COR)
        if curva["sombra_ate"] is not None:
            marcados = x <= curva["sombra_ate"]
            ax.fill_between(x[marcados], 0, y[marcados], color=COR_DESTAQUE,
                            alpha=0.5)
        ax.set_ylabel("densidade")
    if curva.get("prob_sombra") is not None:
        ax.text(0.02, 0.95,
                f"P(X ≤ {curva['sombra_ate']:g}) = {curva['prob_sombra']:.4f}"
                .replace(".", ","),
                transform=ax.transAxes, va="top")
    ax.set_xlabel("x")
    return fig


def curva_poder(funcao_poder, n_min: int, n_max: int, titulo: str,
                n_atual: float | None = None) -> Figure:
    fig, ax = _figura(titulo)
    ns = np.unique(np.linspace(max(2, n_min), n_max, 60).astype(int))
    poderes = [funcao_poder(float(m)) for m in ns]
    ax.plot(ns, poderes, "-", color=COR)
    ax.axhline(0.8, color=COR_DESTAQUE, linestyle="--", linewidth=1,
               label="poder = 0,80")
    if n_atual:
        ax.axvline(n_atual, color="#666666", linestyle=":", linewidth=1)
    ax.set_xlabel("tamanho da amostra (n)")
    ax.set_ylabel("poder")
    ax.set_ylim(0, 1.02)
    ax.legend()
    return fig


def probabilidade_normal(dados, coluna: str) -> Figure:
    x = limpar_numerica(dados, coluna, n_minimo=4)
    fig, ax = _figura(f"Gráfico de probabilidade normal de {coluna}")
    (quantis, ordenados), (inclinacao, intercepto, r) = stats.probplot(x, dist="norm")
    ax.plot(quantis, ordenados, "o", color=COR)
    ax.plot(quantis, inclinacao * quantis + intercepto, "-", color=COR_DESTAQUE)
    ax.set_xlabel("quantis teóricos da normal")
    ax.set_ylabel(coluna)
    ax.text(0.02, 0.95, f"R² da reta = {r**2:.4f}".replace(".", ","),
            transform=ax.transAxes, va="top")
    return fig
