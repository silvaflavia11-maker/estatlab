"""Gráficos da Fase 3: cartas de controle, Pareto, Ishikawa, Multi-Vari,
gage run chart e relatórios de capabilidade."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats

from app.core.cep import CartaControle
from app.core.resultados import ErroAnalise

from .graficos import COR, COR_DESTAQUE

COR_LIMITE = "#c0563a"
COR_CENTRO = "#2e8b57"


def _desenhar_carta(ax, carta: CartaControle) -> None:
    n = carta.pontos.size
    x = np.arange(1, n + 1)
    ax.plot(x, carta.pontos, "-o", color=COR, markersize=4, zorder=3)
    ax.step(x, carta.lc, where="mid", color=COR_CENTRO, linewidth=1.2)
    ax.step(x, carta.lsc, where="mid", color=COR_LIMITE, linewidth=1.2)
    ax.step(x, carta.lic, where="mid", color=COR_LIMITE, linewidth=1.2)
    fora = sorted(carta.indices_violacao())
    if fora:
        ax.plot(x[fora], carta.pontos[fora], "s", color="red", markersize=7,
                zorder=4, label="causa especial")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(carta.nome, fontsize=10)
    ax.set_ylabel(carta.rotulo_y, fontsize=9)
    ax.grid(True, alpha=0.3)
    # rótulos dos limites à direita
    ax.annotate(f"LSC={carta.lsc[-1]:.4g}".replace(".", ","),
                xy=(n, carta.lsc[-1]), fontsize=7, color=COR_LIMITE)
    ax.annotate(f"LC={carta.lc[-1]:.4g}".replace(".", ","),
                xy=(n, carta.lc[-1]), fontsize=7, color=COR_CENTRO)
    ax.annotate(f"LIC={carta.lic[-1]:.4g}".replace(".", ","),
                xy=(n, carta.lic[-1]), fontsize=7, color=COR_LIMITE)


def carta_controle(cartas: list[CartaControle], titulo: str) -> Figure:
    fig = Figure(figsize=(8.5, 3.4 * len(cartas)), dpi=100)
    eixos = fig.subplots(len(cartas), 1, sharex=True)
    if len(cartas) == 1:
        eixos = [eixos]
    for ax, carta in zip(eixos, cartas):
        _desenhar_carta(ax, carta)
    eixos[-1].set_xlabel("amostra")
    fig.suptitle(titulo)
    fig.set_layout_engine("tight")
    return fig


def pareto(contagens: pd.Series, coluna: str) -> Figure:
    fig = Figure(figsize=(7.5, 4.8), dpi=100)
    ax = fig.add_subplot(111)
    valores = contagens.values
    acumulado = 100 * np.cumsum(valores) / valores.sum()
    ax.bar(range(len(valores)), valores, color=COR, edgecolor="white")
    ax.set_xticks(range(len(valores)))
    ax.set_xticklabels([str(c) for c in contagens.index], rotation=45,
                       ha="right")
    ax.set_ylabel("contagem")
    ax.set_title(f"Diagrama de Pareto — {coluna}")
    ax2 = ax.twinx()
    ax2.plot(range(len(valores)), acumulado, "-o", color=COR_DESTAQUE,
             markersize=4)
    ax2.axhline(80, color="#888888", linestyle="--", linewidth=1)
    ax2.set_ylabel("% acumulado")
    ax2.set_ylim(0, 105)
    fig.set_layout_engine("tight")
    return fig


def ishikawa(problema: str, causas: dict[str, list[str]]) -> Figure:
    categorias = {c: lista for c, lista in causas.items() if lista}
    if not categorias:
        raise ErroAnalise("Informe pelo menos uma causa em alguma categoria.")
    fig = Figure(figsize=(10, 6), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(-5, 5)
    # espinha central
    ax.annotate("", xy=(8.6, 0), xytext=(0.4, 0),
                arrowprops=dict(arrowstyle="-|>", color=COR, lw=2))
    ax.text(8.7, 0, problema, va="center", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#eef6f2", ec=COR))
    nomes = list(categorias)
    metade = (len(nomes) + 1) // 2
    posicoes_x = np.linspace(1.2, 7.4, max(metade, 1))
    for k, nome in enumerate(nomes):
        acima = k < metade
        x0 = posicoes_x[k if acima else k - metade]
        y1 = 3.6 if acima else -3.6
        ax.plot([x0, x0 + 1.4], [y1, 0], color=COR, lw=1.5)
        ax.text(x0, y1 + (0.25 if acima else -0.45), nome, fontsize=10,
                fontweight="bold", color=COR)
        for j, causa in enumerate(categorias[nome][:5]):
            frac = (j + 1) / 6
            xc = x0 + 1.4 * frac
            yc = y1 * (1 - frac)
            ax.plot([xc - 0.8, xc], [yc, yc], color="#777777", lw=1)
            ax.text(xc - 0.85, yc, causa, fontsize=8, ha="right", va="center")
    ax.set_title("Diagrama de causa e efeito (Ishikawa)")
    return fig


def multivari(dados: pd.DataFrame, resposta: str,
              fatores: list[str]) -> Figure:
    """``dados``: colunas y, f0, f1[, f2] (de ``qualidade.multivari_dados``)."""
    paineis = sorted(dados["f2"].unique()) if "f2" in dados else [None]
    fig = Figure(figsize=(4.6 * len(paineis) + 2, 4.8), dpi=100)
    eixos = fig.subplots(1, len(paineis), sharey=True)
    if len(paineis) == 1:
        eixos = [eixos]
    for ax, painel in zip(eixos, paineis):
        sub = dados if painel is None else dados[dados["f2"] == painel]
        niveis_f0 = sorted(sub["f0"].unique())
        for nivel1 in sorted(sub["f1"].unique()):
            medias = [sub[(sub.f0 == n0) & (sub.f1 == nivel1)]["y"].mean()
                      for n0 in niveis_f0]
            ax.plot(niveis_f0, medias, "-o", label=f"{fatores[1]}={nivel1}")
        medias_f0 = [sub[sub.f0 == n0]["y"].mean() for n0 in niveis_f0]
        ax.plot(niveis_f0, medias_f0, "--s", color="black", alpha=0.6,
                label=f"média por {fatores[0]}")
        ax.set_xlabel(fatores[0])
        ax.grid(True, alpha=0.3)
        if painel is not None:
            ax.set_title(f"{fatores[2]} = {painel}", fontsize=10)
    eixos[0].set_ylabel(f"média de {resposta}")
    eixos[0].legend(fontsize=8)
    fig.suptitle("Carta Multi-Vari")
    fig.set_layout_engine("tight")
    return fig


def gage_run(dados: pd.DataFrame, medicao: str) -> Figure:
    """``dados``: colunas y, p (peça), o (operador) — de ``core.msa``."""
    fig = Figure(figsize=(8.5, 4.8), dpi=100)
    ax = fig.add_subplot(111)
    pecas = sorted(dados["p"].unique())
    operadores = sorted(dados["o"].unique())
    posicao = 0
    marcadores = ["o", "s", "^", "D", "v", "P", "*"]
    posicoes_peca = []
    for peca in pecas:
        centro_ini = posicao
        for k, oper in enumerate(operadores):
            valores = dados[(dados.p == peca) & (dados.o == oper)]["y"].to_numpy()
            xs = posicao + np.arange(valores.size)
            ax.plot(xs, valores, marcadores[k % len(marcadores)] + "-",
                    markersize=5, label=oper if peca == pecas[0] else None)
            posicao += valores.size
        posicoes_peca.append((centro_ini + posicao - 1) / 2)
        ax.axvline(posicao - 0.5, color="#cccccc", linewidth=0.8)
    ax.axhline(float(dados["y"].mean()), color=COR_DESTAQUE, linestyle="--",
               linewidth=1, label="média geral")
    ax.set_xticks(posicoes_peca)
    ax.set_xticklabels([str(p) for p in pecas])
    ax.set_xlabel("peça")
    ax.set_ylabel(medicao)
    ax.set_title("Gage run chart (medições por peça e operador)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.set_layout_engine("tight")
    return fig


def _histograma_capabilidade(ax, dados_cap: dict) -> None:
    x = dados_cap["x"]
    mu, sigma_d, sigma_g = (dados_cap["mu"], dados_cap["sigma_d"],
                            dados_cap["sigma_g"])
    ax.hist(x, bins="auto", density=True, color=COR, edgecolor="white",
            alpha=0.75)
    grade = np.linspace(min(x.min(), mu - 4 * sigma_g),
                        max(x.max(), mu + 4 * sigma_g), 300)
    ax.plot(grade, stats.norm.pdf(grade, mu, sigma_d), "-",
            color=COR_CENTRO, label="dentro")
    ax.plot(grade, stats.norm.pdf(grade, mu, sigma_g), "--",
            color="#555555", label="geral")
    for limite, nome in ((dados_cap["lie"], "LIE"), (dados_cap["lse"], "LSE")):
        if limite is not None:
            ax.axvline(limite, color=COR_LIMITE, linewidth=1.6)
            ax.text(limite, ax.get_ylim()[1] * 0.95, nome, color=COR_LIMITE,
                    ha="center", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def capabilidade_histograma(dados_cap: dict) -> Figure:
    fig = Figure(figsize=(7.5, 4.8), dpi=100)
    ax = fig.add_subplot(111)
    _histograma_capabilidade(ax, dados_cap)
    ax.set_title(f"Capabilidade — {dados_cap['coluna']}")
    ax.set_xlabel(dados_cap["coluna"])
    fig.set_layout_engine("tight")
    return fig


def relatorio_capabilidade(dados_cap: dict) -> Figure:
    """Relatório de Capabilidade Completo: cartas I e MR, histograma com
    especificações, probabilidade normal e painel de índices."""
    from app.core.cep import carta_i_mr

    x = dados_cap["x"]
    cartas, _ = carta_i_mr(x, dados_cap["coluna"])
    fig = Figure(figsize=(11, 7.5), dpi=100)
    eixos = fig.subplots(2, 3)

    _desenhar_carta(eixos[0][0], cartas[0])
    _desenhar_carta(eixos[1][0], cartas[1])
    eixos[1][0].set_xlabel("observação")

    _histograma_capabilidade(eixos[0][1], dados_cap)
    eixos[0][1].set_title("histograma com especificações", fontsize=10)

    ax = eixos[1][1]
    (quantis, ordenados), (a, b, r) = stats.probplot(x, dist="norm")
    ax.plot(quantis, ordenados, "o", color=COR, markersize=3.5)
    ax.plot(quantis, a * quantis + b, "-", color=COR_LIMITE)
    ax.set_title("probabilidade normal", fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = eixos[0][2]
    ax.plot(np.arange(1, min(26, x.size + 1)), x[-min(25, x.size):], "-o",
            color=COR, markersize=4)
    ax.set_title("últimas 25 observações", fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = eixos[1][2]
    ax.set_axis_off()
    mu, sd, sg = dados_cap["mu"], dados_cap["sigma_d"], dados_cap["sigma_g"]
    lie, lse = dados_cap["lie"], dados_cap["lse"]

    def indice(sigma):
        vals = []
        if lse is not None:
            vals.append((lse - mu) / (3 * sigma))
        if lie is not None:
            vals.append((mu - lie) / (3 * sigma))
        return min(vals)

    texto = (f"n = {x.size}\nmédia = {mu:.4g}\nσ dentro = {sd:.4g}\n"
             f"σ geral = {sg:.4g}\n\nCpk = {indice(sd):.2f}\n"
             f"Ppk = {indice(sg):.2f}")
    if lie is not None and lse is not None:
        texto += (f"\nCp = {(lse - lie) / (6 * sd):.2f}"
                  f"\nPp = {(lse - lie) / (6 * sg):.2f}")
    ax.text(0.05, 0.95, texto.replace(".", ","), va="top", fontsize=11,
            family="monospace")
    ax.set_title("índices", fontsize=10)

    fig.suptitle(f"Relatório de Capabilidade Completo — {dados_cap['coluna']}")
    fig.set_layout_engine("tight")
    return fig


def run_chart(dados, coluna: str) -> Figure:
    from app.core.util import limpar_numerica

    x = limpar_numerica(dados, coluna, n_minimo=2)
    fig = Figure(figsize=(7.5, 4.4), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(np.arange(1, x.size + 1), x, "-o", color=COR, markersize=4)
    ax.axhline(float(np.median(x)), color=COR_DESTAQUE, linestyle="--",
               linewidth=1.2, label="mediana")
    ax.set_xlabel("ordem da observação")
    ax.set_ylabel(coluna)
    ax.set_title(f"Run chart — {coluna}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.set_layout_engine("tight")
    return fig
