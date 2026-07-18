"""Análise multivariada: PCA, análise fatorial, discriminante, cluster,
correspondência e alfa de Cronbach."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage

from .resultados import ErroAnalise, ResultadoComposto


def _matriz_numerica(df: pd.DataFrame, colunas: list[str],
                     minimo_colunas: int = 2) -> pd.DataFrame:
    if len(colunas) < minimo_colunas:
        raise ErroAnalise(f"Selecione pelo menos {minimo_colunas} colunas.")
    matriz = df[colunas].apply(pd.to_numeric, errors="coerce").dropna()
    if len(matriz) < len(colunas) + 2:
        raise ErroAnalise("Observações completas insuficientes.")
    return matriz


def pca(df: pd.DataFrame, colunas: list[str],
        n_componentes: int | None = None) -> ResultadoComposto:
    """Componentes principais da matriz de correlação."""
    from app.reports.formatacao import fmt

    matriz = _matriz_numerica(df, colunas)
    padronizada = (matriz - matriz.mean()) / matriz.std(ddof=1)
    correlacao = np.corrcoef(padronizada, rowvar=False)
    autovalores, autovetores = np.linalg.eigh(correlacao)
    ordem = np.argsort(autovalores)[::-1]
    autovalores = autovalores[ordem]
    autovetores = autovetores[:, ordem]
    # convenção de sinal: maior carga absoluta positiva
    for j in range(autovetores.shape[1]):
        pico = np.argmax(np.abs(autovetores[:, j]))
        if autovetores[pico, j] < 0:
            autovetores[:, j] *= -1

    k = n_componentes or len(colunas)
    proporcao = autovalores / autovalores.sum()
    linhas_var = [[f"CP{j + 1}", fmt(float(autovalores[j]), 3),
                   fmt(100 * float(proporcao[j]), 1) + "%",
                   fmt(100 * float(proporcao[: j + 1].sum()), 1) + "%"]
                  for j in range(len(colunas))]
    linhas_cargas = [[coluna] + [fmt(float(autovetores[i, j]), 3)
                                 for j in range(min(k, 4))]
                     for i, coluna in enumerate(colunas)]
    escores = padronizada.to_numpy() @ autovetores

    n_kaiser = int(np.sum(autovalores > 1))
    return ResultadoComposto(
        titulo="Análise de componentes principais",
        itens=[
            ("subtitulo", "Autovalores (matriz de correlação)"),
            ("tabela", ["componente", "autovalor", "% variância",
                        "% acumulada"], linhas_var),
            ("subtitulo", "Cargas (autovetores)"),
            ("tabela", ["variável"] + [f"CP{j + 1}" for j in range(min(k, 4))],
             linhas_cargas),
            ("interpretacao",
             f"Critério de Kaiser (autovalor > 1) sugere reter {n_kaiser} "
             "componente(s). As cargas mostram o peso de cada variável em cada "
             "componente; use o gráfico de escores para visualizar os dados no "
             "novo espaço."),
        ],
        dados={"autovalores": autovalores, "cargas": autovetores,
               "escores": escores, "colunas": colunas},
    )


def analise_fatorial(df: pd.DataFrame, colunas: list[str],
                     n_fatores: int = 2) -> ResultadoComposto:
    from sklearn.decomposition import FactorAnalysis

    from app.reports.formatacao import fmt

    matriz = _matriz_numerica(df, colunas)
    if not 1 <= n_fatores < len(colunas):
        raise ErroAnalise("Número de fatores deve ser menor que o de variáveis.")
    padronizada = ((matriz - matriz.mean()) / matriz.std(ddof=1)).to_numpy()
    ajuste = FactorAnalysis(n_components=n_fatores, random_state=0)
    ajuste.fit(padronizada)
    cargas = ajuste.components_.T  # variáveis × fatores
    comunalidades = (cargas**2).sum(axis=1)

    linhas = [[coluna]
              + [fmt(float(cargas[i, j]), 3) for j in range(n_fatores)]
              + [fmt(float(comunalidades[i]), 3)]
              for i, coluna in enumerate(colunas)]
    return ResultadoComposto(
        titulo=f"Análise fatorial ({n_fatores} fatores, máxima verossimilhança)",
        itens=[
            ("tabela", ["variável"] + [f"Fator {j + 1}" for j in range(n_fatores)]
             + ["comunalidade"], linhas),
            ("interpretacao",
             "Cargas altas (|carga| > 0,5) indicam que a variável é bem "
             "representada pelo fator. A comunalidade é a fração da variância da "
             "variável explicada pelos fatores retidos."),
        ],
        dados={"cargas": cargas, "colunas": colunas},
    )


def discriminante(df: pd.DataFrame, grupo: str,
                  preditores: list[str]) -> ResultadoComposto:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    from app.reports.formatacao import fmt

    x = df[preditores].apply(pd.to_numeric, errors="coerce")
    g = df[grupo].astype(object)
    vazio = g.isna() | (g.astype(str).str.strip() == "")
    g = g.astype(str).where(~vazio, np.nan)
    dados = pd.concat([g.rename("__g"), x], axis=1).dropna()
    classes = sorted(dados["__g"].unique())
    if len(classes) < 2:
        raise ErroAnalise(f"A coluna '{grupo}' precisa de pelo menos 2 grupos.")

    modelo = LinearDiscriminantAnalysis()
    modelo.fit(dados[preditores], dados["__g"])
    previsto = modelo.predict(dados[preditores])
    acuracia = float((previsto == dados["__g"]).mean())

    matriz_conf = pd.crosstab(dados["__g"], pd.Series(previsto, index=dados.index))
    matriz_conf = matriz_conf.reindex(index=classes, columns=classes,
                                      fill_value=0)
    linhas_conf = [[classe] + [int(v) for v in matriz_conf.loc[classe]]
                   for classe in classes]
    return ResultadoComposto(
        titulo=f"Análise discriminante linear: {grupo}",
        itens=[
            ("nota", f"n = {len(dados)}; grupos: " + ", ".join(classes)),
            ("subtitulo", "Matriz de confusão (reclassificação dos próprios dados)"),
            ("tabela", ["real \\ previsto"] + classes, linhas_conf),
            ("tabela", ["medida", "valor"],
             [["taxa de acerto (resubstituição)", fmt(100 * acuracia, 1) + "%"]]),
            ("interpretacao",
             f"A função discriminante classifica corretamente "
             f"{fmt(100 * acuracia, 1)}% das observações usadas no ajuste. "
             "Essa taxa é otimista; para avaliação honesta, use validação com "
             "dados novos."),
            ("aviso", "A LDA pressupõe normalidade multivariada e matrizes de "
                      "covariância iguais entre os grupos."),
        ],
        dados={"modelo": modelo, "previsto": previsto, "dados": dados},
    )


def cluster(df: pd.DataFrame, colunas: list[str], k: int,
            metodo: str = "kmeans") -> ResultadoComposto:
    from app.reports.formatacao import fmt

    matriz = _matriz_numerica(df, colunas)
    if not 2 <= k <= min(20, len(matriz) - 1):
        raise ErroAnalise("Número de grupos inválido para o tamanho dos dados.")
    padronizada = ((matriz - matriz.mean()) / matriz.std(ddof=1)).to_numpy()

    if metodo == "kmeans":
        from sklearn.cluster import KMeans

        ajuste = KMeans(n_clusters=k, n_init=10, random_state=0)
        rotulos = ajuste.fit_predict(padronizada) + 1
        descricao = "k-médias (dados padronizados)"
        ligacao = None
    elif metodo == "hierarquico":
        ligacao = linkage(padronizada, method="ward")
        rotulos = fcluster(ligacao, t=k, criterion="maxclust")
        descricao = "hierárquico (ligação de Ward, dados padronizados)"
    else:
        raise ErroAnalise("Método inválido: 'kmeans' ou 'hierarquico'.")

    linhas = []
    for grupo in range(1, k + 1):
        membros = matriz[rotulos == grupo]
        linhas.append([f"grupo {grupo}", len(membros)]
                      + [fmt(float(membros[c].mean())) for c in colunas])
    serie_rotulos = pd.Series(np.nan, index=df.index, dtype=object)
    serie_rotulos.loc[matriz.index] = rotulos
    return ResultadoComposto(
        titulo=f"Análise de agrupamento — {descricao}",
        itens=[
            ("tabela", ["grupo", "n"] + [f"média de {c}" for c in colunas],
             linhas),
            ("interpretacao",
             f"As observações foram divididas em {k} grupos pelos perfis das "
             "variáveis selecionadas. Os rótulos foram gravados em uma nova "
             "coluna da planilha; use boxplots por grupo para caracterizá-los."),
        ],
        dados={"rotulos": serie_rotulos, "ligacao": ligacao,
               "colunas": colunas, "matriz": matriz},
    )


def correspondencia(df: pd.DataFrame, col_linhas: str,
                    col_colunas: str) -> ResultadoComposto:
    """Análise de correspondência simples via SVD dos resíduos padronizados."""
    from app.reports.formatacao import fmt

    from .tabelas import _categorias

    pares = pd.DataFrame({"l": _categorias(df, col_linhas),
                          "c": _categorias(df, col_colunas)}).dropna()
    tabela = pd.crosstab(pares["l"], pares["c"])
    if tabela.shape[0] < 2 or tabela.shape[1] < 2:
        raise ErroAnalise("As duas colunas precisam de pelo menos 2 categorias.")
    matriz = tabela.to_numpy(dtype=float)
    total = matriz.sum()
    p = matriz / total
    massa_linha = p.sum(axis=1)
    massa_coluna = p.sum(axis=0)
    esperado = np.outer(massa_linha, massa_coluna)
    residuos = (p - esperado) / np.sqrt(esperado)
    u, s, vt = np.linalg.svd(residuos, full_matrices=False)
    inercia = s**2
    inercia_total = float(inercia.sum())
    n_dim = min(2, len(s))

    coord_linhas = (u[:, :n_dim] * s[:n_dim]) / np.sqrt(massa_linha)[:, None]
    coord_colunas = (vt.T[:, :n_dim] * s[:n_dim]) / np.sqrt(massa_coluna)[:, None]

    linhas_inercia = [[f"dimensão {j + 1}", fmt(float(inercia[j]), 4),
                       fmt(100 * float(inercia[j] / inercia_total), 1) + "%"]
                      for j in range(len(s)) if inercia[j] > 1e-12]
    return ResultadoComposto(
        titulo=f"Análise de correspondência: {col_linhas} × {col_colunas}",
        itens=[
            ("tabela", ["dimensão", "inércia", "% da inércia"], linhas_inercia),
            ("nota", f"Inércia total = {fmt(inercia_total, 4)} "
                     f"(= χ²/n da tabela de contingência)."),
            ("interpretacao",
             "No mapa de correspondência, categorias de linha e coluna próximas "
             "tendem a ocorrer juntas; as duas primeiras dimensões explicam "
             f"{fmt(100 * float(inercia[:n_dim].sum() / inercia_total), 1)}% da "
             "associação."),
        ],
        dados={"coord_linhas": coord_linhas, "coord_colunas": coord_colunas,
               "rotulos_linhas": list(tabela.index),
               "rotulos_colunas": list(tabela.columns)},
    )


def cronbach(df: pd.DataFrame, colunas: list[str]) -> ResultadoComposto:
    from app.reports.formatacao import fmt

    matriz = _matriz_numerica(df, colunas, minimo_colunas=3)
    k = len(colunas)
    variancias = matriz.var(ddof=1)
    var_total = float(matriz.sum(axis=1).var(ddof=1))
    alfa = k / (k - 1) * (1 - float(variancias.sum()) / var_total)

    linhas = []
    total = matriz.sum(axis=1)
    for coluna in colunas:
        resto = total - matriz[coluna]
        r_item = float(np.corrcoef(matriz[coluna], resto)[0, 1])
        sem_item = [c for c in colunas if c != coluna]
        var_sem = float(matriz[sem_item].sum(axis=1).var(ddof=1))
        alfa_sem = ((k - 1) / (k - 2) *
                    (1 - float(matriz[sem_item].var(ddof=1).sum()) / var_sem))
        linhas.append([coluna, fmt(r_item, 3), fmt(alfa_sem, 3)])

    if alfa >= 0.9:
        nivel = "excelente"
    elif alfa >= 0.8:
        nivel = "boa"
    elif alfa >= 0.7:
        nivel = "aceitável"
    else:
        nivel = "insuficiente (< 0,7)"
    return ResultadoComposto(
        titulo="Análise de itens — alfa de Cronbach",
        itens=[
            ("tabela", ["medida", "valor"],
             [["alfa de Cronbach", fmt(float(alfa), 3)],
              ["itens", str(k)], ["n", str(len(matriz))]]),
            ("subtitulo", "Estatísticas por item"),
            ("tabela", ["item", "correlação item-total (corrigida)",
                        "alfa se o item for removido"], linhas),
            ("interpretacao",
             f"Alfa = {fmt(float(alfa), 2)}: consistência interna {nivel}. "
             "Itens com correlação item-total baixa ou cujo alfa aumenta quando "
             "removidos são candidatos à revisão."),
        ],
    )
