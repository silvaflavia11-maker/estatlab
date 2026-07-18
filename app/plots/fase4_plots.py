"""Gráficos da Fase 4: DOE, séries temporais, multivariada e confiabilidade."""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from scipy import stats

from app.core.resultados import ErroAnalise

from .graficos import COR, COR_DESTAQUE, _figura


# ------------------------------------------------------------------- DOE

def pareto_efeitos(efeitos: dict[str, float], margem: float | None,
                   titulo: str) -> Figure:
    nomes = sorted(efeitos, key=lambda t: abs(efeitos[t]))
    valores = [abs(efeitos[n]) for n in nomes]
    fig = Figure(figsize=(7.5, 0.45 * len(nomes) + 2.2), dpi=100)
    ax = fig.add_subplot(111)
    ax.barh(nomes, valores, color=COR, edgecolor="white")
    if margem is not None:
        ax.axvline(margem, color=COR_DESTAQUE, linestyle="--",
                   label="margem de Lenth")
        ax.legend(fontsize=8)
    ax.set_xlabel("|efeito|")
    ax.set_title(f"Pareto dos efeitos — {titulo}")
    ax.grid(True, axis="x", alpha=0.3)
    fig.set_layout_engine("tight")
    return fig


def normal_efeitos(efeitos: dict[str, float], margem: float | None,
                   titulo: str, meio_normal: bool = False) -> Figure:
    nomes = list(efeitos)
    valores = np.array([efeitos[n] for n in nomes])
    fig, ax = _figura(("Meio-normal" if meio_normal else "Normal")
                      + f" dos efeitos — {titulo}")
    n = len(valores)
    if meio_normal:
        ordem = np.argsort(np.abs(valores))
        pontos = np.abs(valores)[ordem]
        quantis = stats.halfnorm.ppf((np.arange(1, n + 1) - 0.5) / n)
        ax.set_xlabel("|efeito|")
    else:
        ordem = np.argsort(valores)
        pontos = valores[ordem]
        quantis = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
        ax.set_xlabel("efeito")
    ax.plot(pontos, quantis, "o", color=COR)
    destacar = ([i for i, idx in enumerate(ordem)
                 if margem is not None and abs(valores[idx]) > margem])
    for i in destacar:
        ax.annotate(nomes[ordem[i]], (pontos[i], quantis[i]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points",
                    color=COR_DESTAQUE)
        ax.plot(pontos[i], quantis[i], "o", color=COR_DESTAQUE)
    ax.set_ylabel("quantis teóricos")
    return fig


def cubo(dados_modelo: dict) -> Figure:
    """Cubo com médias ajustadas nos vértices (exatamente 3 fatores)."""
    from app.core.doe import prever_modelo

    fatores = dados_modelo["fatores"]
    if len(fatores) != 3:
        raise ErroAnalise("O gráfico de cubo exige exatamente 3 fatores.")
    fig = Figure(figsize=(7, 6), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    for a in (-1, 1):
        for b in (-1, 1):
            for c in (-1, 1):
                y = prever_modelo(dados_modelo,
                                  dict(zip(fatores, (a, b, c))))
                ax.scatter(a, b, c, color=COR, s=40)
                ax.text(a, b, c, f" {y:.3g}".replace(".", ","), fontsize=9)
    for inicio, fim in [((-1, -1, -1), (1, -1, -1)), ((-1, -1, -1), (-1, 1, -1)),
                        ((-1, -1, -1), (-1, -1, 1)), ((1, 1, -1), (-1, 1, -1)),
                        ((1, 1, -1), (1, -1, -1)), ((1, 1, -1), (1, 1, 1)),
                        ((1, -1, 1), (-1, -1, 1)), ((1, -1, 1), (1, 1, 1)),
                        ((1, -1, 1), (1, -1, -1)), ((-1, 1, 1), (1, 1, 1)),
                        ((-1, 1, 1), (-1, -1, 1)), ((-1, 1, 1), (-1, 1, -1))]:
        xs, ys, zs = zip(inicio, fim)
        ax.plot(xs, ys, zs, color="#999999", linewidth=1)
    ax.set_xlabel(fatores[0])
    ax.set_ylabel(fatores[1])
    ax.set_zlabel(fatores[2])
    ax.set_title("Gráfico de cubo (respostas previstas)")
    return fig


def contorno_superficie(dados_modelo: dict, fator_x: str, fator_y: str,
                        em_3d: bool = False) -> Figure:
    """Contorno ou superfície 3D (rotacionável) do modelo ajustado."""
    from app.core.doe import prever_modelo

    fatores = dados_modelo["fatores"]
    base = dados_modelo.get("x", dados_modelo.get("codificado"))
    grade_x = np.linspace(float(base[fator_x].min()),
                          float(base[fator_x].max()), 40)
    grade_y = np.linspace(float(base[fator_y].min()),
                          float(base[fator_y].max()), 40)
    mx, my = np.meshgrid(grade_x, grade_y)
    fixos = {f: float(base[f].mean()) for f in fatores
             if f not in (fator_x, fator_y)}
    mz = np.empty_like(mx)
    for i in range(mx.shape[0]):
        for j in range(mx.shape[1]):
            ponto = {fator_x: float(mx[i, j]), fator_y: float(my[i, j]), **fixos}
            mz[i, j] = prever_modelo(dados_modelo, ponto)

    resposta = dados_modelo["resposta"]
    if em_3d:
        fig = Figure(figsize=(7.6, 6.2), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        superficie = ax.plot_surface(mx, my, mz, cmap="viridis", alpha=0.9)
        fig.colorbar(superficie, shrink=0.6, label=resposta)
        ax.set_zlabel(resposta)
        ax.set_title(f"Superfície de resposta — {resposta} "
                     "(arraste para girar)")
    else:
        fig = Figure(figsize=(7.2, 5.6), dpi=100)
        ax = fig.add_subplot(111)
        contorno = ax.contourf(mx, my, mz, levels=12, cmap="viridis")
        linhas = ax.contour(mx, my, mz, levels=12, colors="white",
                            linewidths=0.6)
        ax.clabel(linhas, fontsize=7, fmt=lambda v: f"{v:.3g}")
        fig.colorbar(contorno, label=resposta)
        ax.set_title(f"Contorno — {resposta}")
    ax.set_xlabel(fator_x)
    ax.set_ylabel(fator_y)
    if fixos:
        fig.text(0.01, 0.01,
                 "fixados: " + ", ".join(f"{k} = {v:.3g}" for k, v in fixos.items()),
                 fontsize=7)
    fig.set_layout_engine("tight")
    return fig


# --------------------------------------------------------- séries temporais

def serie_previsao(dados_serie: dict, titulo: str) -> Figure:
    y = dados_serie["y"]
    ajustado = dados_serie.get("ajustado")
    previsao = dados_serie.get("previsao")
    fig, ax = _figura(titulo)
    t = np.arange(1, y.size + 1)
    ax.plot(t, y, "-o", color=COR, markersize=3.5, label="dados")
    if ajustado is not None:
        ax.plot(t, ajustado, "-", color=COR_DESTAQUE, label="ajustado")
    if previsao is not None and len(previsao):
        t_prev = np.arange(y.size + 1, y.size + len(previsao) + 1)
        ax.plot(t_prev, previsao, "--s", color="#2e8b57", markersize=4,
                label="previsão")
        ic = dados_serie.get("ic_previsao")
        if ic is not None:
            ax.fill_between(t_prev, ic[:, 0], ic[:, 1], color="#2e8b57",
                            alpha=0.18)
    ax.set_xlabel("período")
    ax.set_ylabel(dados_serie.get("coluna", "valor"))
    ax.legend(fontsize=8)
    return fig


def decomposicao_paineis(dados_dec: dict) -> Figure:
    fig = Figure(figsize=(8.5, 7), dpi=100)
    eixos = fig.subplots(4, 1, sharex=True)
    series = [("dados", dados_dec["y"]), ("tendência", dados_dec["tendencia"]),
              ("sazonal", dados_dec["sazonal"]), ("resíduo", dados_dec["residuo"])]
    for ax, (nome, valores) in zip(eixos, series):
        ax.plot(np.arange(1, len(valores) + 1), valores, "-", color=COR)
        ax.set_ylabel(nome, fontsize=9)
        ax.grid(True, alpha=0.3)
    eixos[-1].set_xlabel("período")
    fig.suptitle(f"Decomposição — {dados_dec['coluna']}")
    fig.set_layout_engine("tight")
    return fig


def acf_pacf(dados_acf: dict) -> Figure:
    limite = dados_acf["limite"]
    if dados_acf["tipo"] == "ccf":
        series = [("CCF", dados_acf["valores"])]
    else:
        series = [("ACF", dados_acf["acf"]), ("PACF", dados_acf["pacf"])]
    fig = Figure(figsize=(8, 3.2 * len(series)), dpi=100)
    eixos = fig.subplots(len(series), 1)
    if len(series) == 1:
        eixos = [eixos]
    for ax, (nome, valores) in zip(eixos, series):
        defasagens = np.arange(len(valores))
        ax.vlines(defasagens, 0, valores, color=COR, linewidth=3)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(limite, color=COR_DESTAQUE, linestyle="--", linewidth=1)
        ax.axhline(-limite, color=COR_DESTAQUE, linestyle="--", linewidth=1)
        ax.set_ylabel(nome)
        ax.grid(True, alpha=0.3)
    eixos[-1].set_xlabel("defasagem")
    fig.suptitle(f"{'/'.join(n for n, _ in series)} — {dados_acf['coluna']}")
    fig.set_layout_engine("tight")
    return fig


# ------------------------------------------------------------- multivariada

def scree_escores(dados_pca: dict) -> Figure:
    autovalores = dados_pca["autovalores"]
    escores = dados_pca["escores"]
    fig = Figure(figsize=(9.6, 4.4), dpi=100)
    ax1, ax2 = fig.subplots(1, 2)
    ax1.plot(np.arange(1, len(autovalores) + 1), autovalores, "-o", color=COR)
    ax1.axhline(1, color=COR_DESTAQUE, linestyle="--", linewidth=1,
                label="autovalor = 1")
    ax1.set_xlabel("componente")
    ax1.set_ylabel("autovalor")
    ax1.set_title("Scree plot", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax2.plot(escores[:, 0], escores[:, 1], "o", color=COR, alpha=0.7)
    ax2.axhline(0, color="#999999", linewidth=0.8)
    ax2.axvline(0, color="#999999", linewidth=0.8)
    ax2.set_xlabel("CP1")
    ax2.set_ylabel("CP2")
    ax2.set_title("Escores (CP1 × CP2)", fontsize=10)
    ax2.grid(True, alpha=0.3)
    fig.suptitle("Componentes principais")
    fig.set_layout_engine("tight")
    return fig


def dendrograma(dados_cluster: dict) -> Figure:
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrograma

    if dados_cluster.get("ligacao") is None:
        raise ErroAnalise("Dendrograma disponível apenas para o método "
                          "hierárquico.")
    fig = Figure(figsize=(8.5, 5), dpi=100)
    ax = fig.add_subplot(111)
    scipy_dendrograma(dados_cluster["ligacao"], ax=ax, no_labels=True,
                      color_threshold=0)
    ax.set_title("Dendrograma (ligação de Ward)")
    ax.set_ylabel("distância")
    fig.set_layout_engine("tight")
    return fig


def mapa_correspondencia(dados_ca: dict) -> Figure:
    fig, ax = _figura("Mapa de correspondência")
    cl = dados_ca["coord_linhas"]
    cc = dados_ca["coord_colunas"]
    ax.plot(cl[:, 0], cl[:, 1] if cl.shape[1] > 1 else np.zeros(len(cl)),
            "o", color=COR)
    for i, rotulo in enumerate(dados_ca["rotulos_linhas"]):
        ax.annotate(rotulo, (cl[i, 0], cl[i, 1] if cl.shape[1] > 1 else 0),
                    fontsize=9, color=COR, xytext=(4, 4),
                    textcoords="offset points")
    ax.plot(cc[:, 0], cc[:, 1] if cc.shape[1] > 1 else np.zeros(len(cc)),
            "s", color=COR_DESTAQUE)
    for i, rotulo in enumerate(dados_ca["rotulos_colunas"]):
        ax.annotate(rotulo, (cc[i, 0], cc[i, 1] if cc.shape[1] > 1 else 0),
                    fontsize=9, color=COR_DESTAQUE, xytext=(4, -10),
                    textcoords="offset points")
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.axvline(0, color="#999999", linewidth=0.8)
    ax.set_xlabel("dimensão 1")
    ax.set_ylabel("dimensão 2")
    return fig


# ------------------------------------------------------------ confiabilidade

def sobrevivencia(dados_conf: dict) -> Figure:
    fig = Figure(figsize=(9.6, 4.4), dpi=100)
    ax1, ax2 = fig.subplots(1, 2)
    if "km" in dados_conf:
        km = dados_conf["km"]
        km.plot_survival_function(ax=ax1, color=COR)
        ax1.set_title("Sobrevivência (Kaplan-Meier)", fontsize=10)
        tempos = dados_conf["tempos"]
        eventos = dados_conf["eventos"]
        ax2.hist(tempos[eventos == 1], bins="auto", color=COR,
                 edgecolor="white")
        ax2.set_title("Tempos de falha observados", fontsize=10)
        ax2.set_xlabel("tempo")
    else:
        ajuste = dados_conf["ajuste"]
        ajuste.plot_survival_function(ax=ax1, color=COR)
        ax1.set_title(f"Sobrevivência ({dados_conf['familia']})", fontsize=10)
        ajuste.plot_hazard(ax=ax2, color=COR_DESTAQUE)
        ax2.set_title("Taxa de falha h(t)", fontsize=10)
        ax2.set_xlabel("tempo")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("tempo")
    fig.suptitle(f"Confiabilidade — {dados_conf['coluna']}")
    fig.set_layout_engine("tight")
    return fig


def probabilidade_weibull(dados_conf: dict) -> Figure:
    """Gráfico de probabilidade Weibull (postos medianos, falhas observadas)."""
    tempos = np.sort(np.asarray(dados_conf["tempos"], dtype=float))
    if "eventos" in dados_conf:
        tempos = np.sort(np.asarray(dados_conf["tempos"])[
            np.asarray(dados_conf["eventos"]) == 1])
    if tempos.size < 3:
        raise ErroAnalise("Menos de 3 falhas observadas para o gráfico.")
    n = tempos.size
    postos = (np.arange(1, n + 1) - 0.3) / (n + 0.4)  # postos medianos
    fig, ax = _figura(f"Probabilidade Weibull — {dados_conf['coluna']}")
    x = np.log(tempos)
    y = np.log(-np.log(1 - postos))
    ax.plot(x, y, "o", color=COR)
    coef = np.polyfit(x, y, 1)
    ax.plot(x, np.polyval(coef, x), "-", color=COR_DESTAQUE,
            label=f"β ≈ {coef[0]:.2f}".replace(".", ","))
    ax.set_xlabel("ln(tempo)")
    ax.set_ylabel("ln(−ln(1−F))")
    ax.legend(fontsize=8)
    return fig


# ------------------------------------------------------------------ outros

def arvore_plot(dados_arvore: dict) -> Figure:
    from sklearn.tree import plot_tree

    fig = Figure(figsize=(11, 7), dpi=100)
    ax = fig.add_subplot(111)
    plot_tree(dados_arvore["modelo"], feature_names=dados_arvore["preditores"],
              filled=True, rounded=True, fontsize=8, ax=ax,
              class_names=True if dados_arvore["classificacao"] else None)
    ax.set_title(f"Árvore — {dados_arvore['resposta']}")
    return fig


def ajuste_nao_linear(dados_nl: dict) -> Figure:
    x, y = dados_nl["x"], dados_nl["y"]
    fig, ax = _figura(f"Ajuste não linear — {dados_nl['resposta']} × "
                      f"{dados_nl['preditor']}")
    ax.plot(x, y, "o", color=COR, alpha=0.75, label="dados")
    grade = np.linspace(x.min(), x.max(), 300)
    ax.plot(grade, dados_nl["funcao"](grade, *dados_nl["parametros"]), "-",
            color=COR_DESTAQUE, label="modelo")
    ax.set_xlabel(dados_nl["preditor"])
    ax.set_ylabel(dados_nl["resposta"])
    ax.legend(fontsize=8)
    return fig


def bootstrap_hist(dados_boot: dict) -> Figure:
    fig, ax = _figura(f"Distribuição bootstrap — {dados_boot['estatistica']} "
                      f"de {dados_boot['coluna']}")
    ax.hist(dados_boot["distribuicao"], bins=40, color=COR, edgecolor="white")
    ax.set_xlabel(dados_boot["estatistica"])
    ax.set_ylabel("frequência")
    return fig
