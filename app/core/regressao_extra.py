"""Fase 4 — regressões adicionais e reamostragem: não linear, logística
ordinal/nominal, PLS, ortogonal (Deming), testes de equivalência (TOST),
bootstrap/aleatorização e árvores de classificação/regressão."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .resultados import ErroAnalise, ResultadoComposto, ResultadoTeste
from .util import limpar_numerica

# modelos não lineares disponíveis: nome → (f(x, *p), rótulos, chutes)
MODELOS_NAO_LINEARES = {
    "Exponencial: a·exp(b·x)": (
        lambda x, a, b: a * np.exp(b * x), ["a", "b"],
        lambda x, y: (max(abs(y).min(), 1e-3), 0.1)),
    "Potência: a·x^b": (
        lambda x, a, b: a * np.power(x, b), ["a", "b"],
        lambda x, y: (1.0, 1.0)),
    "Logística: c/(1+exp(−b(x−a)))": (
        lambda x, a, b, c: c / (1 + np.exp(-b * (x - a))), ["a", "b", "c"],
        lambda x, y: (float(np.median(x)), 1.0, float(y.max()))),
    "Michaelis-Menten: a·x/(b+x)": (
        lambda x, a, b: a * x / (b + x), ["a", "b"],
        lambda x, y: (float(y.max()), float(np.median(x)))),
}


def regressao_nao_linear(df: pd.DataFrame, resposta: str, preditor: str,
                         modelo: str, alfa: float = 0.05) -> ResultadoComposto:
    from app.reports.formatacao import fmt

    if modelo not in MODELOS_NAO_LINEARES:
        raise ErroAnalise(f"Modelo inválido: {modelo}.")
    funcao, rotulos, chute = MODELOS_NAO_LINEARES[modelo]
    dados = pd.DataFrame({
        "x": pd.to_numeric(df[preditor], errors="coerce"),
        "y": pd.to_numeric(df[resposta], errors="coerce"),
    }).dropna()
    if len(dados) < len(rotulos) + 3:
        raise ErroAnalise("Observações insuficientes para o modelo escolhido.")
    x, y = dados["x"].to_numpy(), dados["y"].to_numpy()
    if "Potência" in modelo and np.any(x <= 0):
        raise ErroAnalise("O modelo de potência exige x > 0.")
    try:
        parametros, cov = optimize.curve_fit(funcao, x, y, p0=chute(x, y),
                                             maxfev=20000)
    except Exception as erro:
        raise ErroAnalise(f"O ajuste não convergiu: {erro}")

    ajustado = funcao(x, *parametros)
    residuos = y - ajustado
    gl = len(x) - len(parametros)
    ep = np.sqrt(np.diag(cov))
    t_crit = stats.t.ppf(1 - alfa / 2, gl)
    ss_res = float(np.sum(residuos**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    linhas = [[rotulo, fmt(float(p), 5), fmt(float(e), 5),
               f"({fmt(float(p - t_crit * e))}; {fmt(float(p + t_crit * e))})"]
              for rotulo, p, e in zip(rotulos, parametros, ep)]
    return ResultadoComposto(
        titulo=f"Regressão não linear — {modelo}",
        itens=[
            ("subtitulo", "Parâmetros estimados"),
            ("tabela", ["parâmetro", "estimativa", "EP",
                        f"IC {fmt(100 * (1 - alfa), 0)}%"], linhas),
            ("nota", f"R² = {fmt(100 * r2, 2)}%  •  s = "
                     f"{fmt(float(np.sqrt(ss_res / gl)))}  •  n = {len(x)}"),
            ("interpretacao", "Os ICs vêm da aproximação linear local "
                              "(matriz de covariância do ajuste); com poucos "
                              "dados eles são apenas indicativos."),
            ("aviso", "Verifique o gráfico dos resíduos: um padrão sistemático "
                      "indica que a forma funcional escolhida não é adequada."),
        ],
        dados={"x": x, "y": y, "ajustado": ajustado, "residuos": residuos,
               "funcao": funcao, "parametros": parametros,
               "resposta": resposta, "preditor": preditor},
    )


def logistica_ordinal(df: pd.DataFrame, resposta: str, preditores: list[str],
                      ordem: list[str] | None = None,
                      alfa: float = 0.05) -> ResultadoComposto:
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    from app.reports.formatacao import fmt, fmt_p

    x = df[preditores].apply(pd.to_numeric, errors="coerce")
    y_bruto = df[resposta].astype(object)
    vazio = y_bruto.isna() | (y_bruto.astype(str).str.strip() == "")
    y_bruto = y_bruto.astype(str).where(~vazio, np.nan)
    dados = pd.concat([y_bruto.rename("__y"), x], axis=1).dropna()
    categorias = ordem or sorted(dados["__y"].unique())
    if len(categorias) < 3:
        raise ErroAnalise("A logística ordinal exige 3+ categorias ordenadas; "
                          "com 2, use a logística binária.")
    if set(dados["__y"]) - set(categorias):
        raise ErroAnalise("A ordem informada não cobre todas as categorias.")
    y = pd.Series(pd.Categorical(dados["__y"], categories=categorias,
                                 ordered=True), index=dados.index)

    try:
        modelo = OrderedModel(y, dados[preditores].astype(float),
                              distr="logit").fit(method="bfgs", disp=False)
    except Exception as erro:
        raise ErroAnalise(f"O ajuste não convergiu: {erro}")

    linhas = []
    for nome in preditores:
        coef = float(modelo.params[nome])
        linhas.append([nome, fmt(coef, 4), fmt(float(modelo.bse[nome]), 4),
                       fmt_p(float(modelo.pvalues[nome])),
                       fmt(float(np.exp(coef)), 3)])
    return ResultadoComposto(
        titulo=f"Regressão logística ordinal: {resposta}",
        itens=[
            ("nota", "Ordem das categorias: " + " < ".join(categorias)),
            ("tabela", ["preditor", "coeficiente", "EP", "p-valor",
                        "razão de chances"], linhas),
            ("nota", f"n = {len(dados)}  •  pseudo-R² (McFadden) = "
                     f"{fmt(float(modelo.prsquared), 3)}"),
            ("interpretacao", "Razão de chances > 1: aumentos no preditor elevam "
                              "a chance de categorias mais altas na ordem. O "
                              "modelo pressupõe chances proporcionais entre os "
                              "cortes."),
        ],
    )


def logistica_nominal(df: pd.DataFrame, resposta: str, preditores: list[str],
                      alfa: float = 0.05) -> ResultadoComposto:
    import statsmodels.api as sm

    from app.reports.formatacao import fmt, fmt_p

    x = df[preditores].apply(pd.to_numeric, errors="coerce")
    y_bruto = df[resposta].astype(object)
    vazio = y_bruto.isna() | (y_bruto.astype(str).str.strip() == "")
    y_bruto = y_bruto.astype(str).where(~vazio, np.nan)
    dados = pd.concat([y_bruto.rename("__y"), x], axis=1).dropna()
    categorias = sorted(dados["__y"].unique())
    if len(categorias) < 3:
        raise ErroAnalise("A logística nominal exige 3+ categorias; com 2, use "
                          "a logística binária.")
    codigos = pd.Categorical(dados["__y"], categories=categorias).codes
    try:
        modelo = sm.MNLogit(codigos, sm.add_constant(dados[preditores])).fit(
            disp=False)
    except Exception as erro:
        raise ErroAnalise(f"O ajuste não convergiu: {erro}")

    itens: list[tuple] = [
        ("nota", f"Categoria de referência: \"{categorias[0]}\". "
                 f"n = {len(dados)}."),
    ]
    nomes_linhas = ["constante"] + preditores
    for j, categoria in enumerate(categorias[1:]):
        linhas = []
        for i, nome in enumerate(nomes_linhas):
            coef = float(modelo.params.iloc[i, j])
            p = float(modelo.pvalues.iloc[i, j])
            rc = "—" if nome == "constante" else fmt(float(np.exp(coef)), 3)
            linhas.append([nome, fmt(coef, 4), fmt_p(p), rc])
        itens.append(("subtitulo", f"logit(\"{categoria}\" vs "
                                   f"\"{categorias[0]}\")"))
        itens.append(("tabela", ["termo", "coeficiente", "p-valor",
                                 "razão de chances"], linhas))
    itens.append(("interpretacao",
                  "Cada bloco compara uma categoria com a referência; a razão "
                  "de chances indica como o preditor desloca a escolha entre "
                  "as duas."))
    return ResultadoComposto(titulo=f"Regressão logística nominal: {resposta}",
                             itens=itens)


def pls(df: pd.DataFrame, resposta: str, preditores: list[str],
        n_componentes: int = 2) -> ResultadoComposto:
    from sklearn.cross_decomposition import PLSRegression

    from app.reports.formatacao import fmt

    x = df[preditores].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[resposta], errors="coerce")
    dados = pd.concat([y.rename("__y"), x], axis=1).dropna()
    if not 1 <= n_componentes <= min(len(preditores), len(dados) - 1):
        raise ErroAnalise("Número de componentes inválido.")
    modelo = PLSRegression(n_components=n_componentes)
    modelo.fit(dados[preditores], dados["__y"])
    r2 = float(modelo.score(dados[preditores], dados["__y"]))
    ajustado = modelo.predict(dados[preditores]).ravel()

    x_pad = (dados[preditores] - dados[preditores].mean()).to_numpy()
    var_x_total = float((x_pad**2).sum())
    var_por_comp = [float((modelo.x_scores_[:, [j]]
                           @ modelo.x_loadings_[:, [j]].T).var(axis=0).sum())
                    for j in range(n_componentes)]

    linhas_coef = [[nome, fmt(float(c), 5)]
                   for nome, c in zip(preditores, modelo.coef_.ravel())]
    return ResultadoComposto(
        titulo=f"Mínimos quadrados parciais (PLS): {resposta}",
        itens=[
            ("nota", f"{n_componentes} componente(s)  •  n = {len(dados)}  •  "
                     f"R² = {fmt(100 * r2, 2)}%"),
            ("subtitulo", "Coeficientes (variáveis padronizadas internamente)"),
            ("tabela", ["preditor", "coeficiente"], linhas_coef),
            ("interpretacao", "O PLS constrói componentes que maximizam a "
                              "covariância com a resposta — útil com preditores "
                              "muito correlacionados ou mais preditores que "
                              "observações."),
        ],
        dados={"ajustado": ajustado, "y": dados["__y"].to_numpy(),
               "residuos": dados["__y"].to_numpy() - ajustado},
    )


def regressao_ortogonal(df: pd.DataFrame, resposta: str, preditor: str,
                        razao_variancias: float = 1.0,
                        alfa: float = 0.05) -> ResultadoComposto:
    """Regressão de Deming: erro nas duas variáveis; δ = var(erro y)/var(erro x)."""
    from app.reports.formatacao import fmt

    dados = pd.DataFrame({
        "x": pd.to_numeric(df[preditor], errors="coerce"),
        "y": pd.to_numeric(df[resposta], errors="coerce"),
    }).dropna()
    if len(dados) < 3:
        raise ErroAnalise("São necessárias pelo menos 3 observações completas.")
    if razao_variancias <= 0:
        raise ErroAnalise("A razão de variâncias deve ser positiva.")
    x, y = dados["x"].to_numpy(), dados["y"].to_numpy()
    sxx = float(np.var(x, ddof=1))
    syy = float(np.var(y, ddof=1))
    sxy = float(np.cov(x, y, ddof=1)[0, 1])
    if sxy == 0:
        raise ErroAnalise("Covariância nula: não há relação linear a estimar.")
    delta = razao_variancias
    inclinacao = ((syy - delta * sxx
                   + np.sqrt((syy - delta * sxx) ** 2 + 4 * delta * sxy**2))
                  / (2 * sxy))
    intercepto = float(y.mean() - inclinacao * x.mean())

    # IC da inclinação por bootstrap (percentil)
    rng = np.random.default_rng(0)
    inclinacoes = []
    for _ in range(1000):
        indices = rng.integers(0, len(x), len(x))
        xb, yb = x[indices], y[indices]
        sxxb, syyb = np.var(xb, ddof=1), np.var(yb, ddof=1)
        sxyb = np.cov(xb, yb, ddof=1)[0, 1]
        if sxyb != 0:
            inclinacoes.append((syyb - delta * sxxb + np.sqrt(
                (syyb - delta * sxxb) ** 2 + 4 * delta * sxyb**2)) / (2 * sxyb))
    lo, hi = np.percentile(inclinacoes, [100 * alfa / 2, 100 * (1 - alfa / 2)])

    return ResultadoComposto(
        titulo=f"Regressão ortogonal (Deming): {resposta} × {preditor}",
        itens=[
            ("tabela", ["item", "valor"],
             [["inclinação", fmt(float(inclinacao), 5)],
              [f"IC {fmt(100 * (1 - alfa), 0)}% da inclinação (bootstrap)",
               f"({fmt(float(lo), 4)}; {fmt(float(hi), 4)})"],
              ["intercepto", fmt(intercepto, 5)],
              ["razão de variâncias dos erros (δ)", fmt(delta, 3)],
              ["n", str(len(x))]]),
            ("interpretacao",
             "Use quando as duas variáveis têm erro de medição (ex.: comparação "
             "de métodos analíticos). Se o IC da inclinação contém 1 e o do "
             "intercepto contém 0, os métodos são estatisticamente equivalentes."),
        ],
        dados={"x": x, "y": y, "inclinacao": inclinacao,
               "intercepto": intercepto, "resposta": resposta,
               "preditor": preditor},
    )


def equivalencia(dados1, col1: str, margem_inferior: float,
                 margem_superior: float, dados2=None, col2: str | None = None,
                 pareado: bool = False, alfa: float = 0.05) -> ResultadoTeste:
    """TOST: 1 amostra (média dentro das margens), 2 amostras ou pareado."""
    from statsmodels.stats.weightstats import DescrStatsW, ttost_ind, ttost_paired

    if margem_inferior >= margem_superior:
        raise ErroAnalise("A margem inferior deve ser menor que a superior.")
    x1 = limpar_numerica(dados1, col1, n_minimo=3)

    if dados2 is None:
        p, t1, t2 = DescrStatsW(x1).ttost_mean(margem_inferior, margem_superior)
        titulo = f"Teste de equivalência (1 amostra): {col1}"
        h0 = (f"H₀: a média de '{col1}' está fora de "
              f"({margem_inferior}; {margem_superior})")
        conclusao = (f"a média de '{col1}' é equivalente à referência (dentro "
                     f"das margens)")
        amostras = [{"amostra": col1, "n": int(x1.size),
                     "média": float(np.mean(x1)),
                     "desvio-padrão": float(np.std(x1, ddof=1))}]
    else:
        x2 = limpar_numerica(dados2, col2, n_minimo=3)
        if pareado:
            n = min(x1.size, x2.size)
            p, t1, t2 = ttost_paired(x1[:n], x2[:n], margem_inferior,
                                     margem_superior)
            titulo = f"Teste de equivalência (pareado): {col1} − {col2}"
        else:
            p, t1, t2 = ttost_ind(x1, x2, margem_inferior, margem_superior)
            titulo = f"Teste de equivalência (2 amostras): {col1} × {col2}"
        h0 = (f"H₀: a diferença das médias está fora de "
              f"({margem_inferior}; {margem_superior})")
        conclusao = f"as médias de '{col1}' e '{col2}' são equivalentes"
        amostras = [{"amostra": c, "n": int(v.size), "média": float(np.mean(v)),
                     "desvio-padrão": float(np.std(v, ddof=1))}
                    for c, v in ((col1, x1), (col2, x2))]

    return ResultadoTeste(
        titulo=titulo,
        h0=h0,
        h1=f"H₁: equivalência — a diferença está dentro de "
           f"({margem_inferior}; {margem_superior})",
        nome_estatistica="TOST (maior p dos dois testes unilaterais)",
        estatistica=float(max(t1[0], t2[0])),
        p_valor=float(p),
        alfa=alfa,
        conclusao_h1=conclusao,
        amostras=amostras,
        detalhes={"margens": f"({margem_inferior}; {margem_superior})"},
        avisos=["No TOST, rejeitar H₀ demonstra equivalência — a lógica é "
                "invertida em relação ao teste t usual."],
    )


def bootstrap_ic(dados, coluna: str, estatistica: str = "média",
                 n_reamostras: int = 2000, alfa: float = 0.05,
                 semente: int | None = None) -> ResultadoComposto:
    from app.reports.formatacao import fmt

    x = limpar_numerica(dados, coluna, n_minimo=5)
    funcoes = {"média": np.mean, "mediana": np.median,
               "desvio-padrão": lambda v: np.std(v, ddof=1)}
    if estatistica not in funcoes:
        raise ErroAnalise("Estatística inválida.")
    resultado = stats.bootstrap(
        (x,), funcoes[estatistica], n_resamples=n_reamostras,
        confidence_level=1 - alfa, method="BCa",
        random_state=np.random.default_rng(semente))
    ic = resultado.confidence_interval
    return ResultadoComposto(
        titulo=f"Bootstrap ({estatistica}): {coluna}",
        itens=[
            ("tabela", ["item", "valor"],
             [["estatística observada", fmt(float(funcoes[estatistica](x)))],
              [f"IC {fmt(100 * (1 - alfa), 0)}% (BCa)",
               f"({fmt(float(ic.low))}; {fmt(float(ic.high))})"],
              ["reamostras", str(n_reamostras)],
              ["erro-padrão bootstrap", fmt(float(resultado.standard_error))]]),
            ("interpretacao",
             "O bootstrap reamostra os próprios dados para estimar a incerteza "
             "sem supor normalidade. O método BCa corrige viés e assimetria."),
        ],
        dados={"distribuicao": np.asarray(resultado.bootstrap_distribution),
               "coluna": coluna, "estatistica": estatistica},
    )


def teste_aleatorizacao(dados1, dados2, col1: str, col2: str,
                        n_permutacoes: int = 5000, alfa: float = 0.05,
                        semente: int | None = None) -> ResultadoTeste:
    x1 = limpar_numerica(dados1, col1)
    x2 = limpar_numerica(dados2, col2)
    resultado = stats.permutation_test(
        (x1, x2), lambda a, b, axis=-1: np.mean(a, axis=axis) - np.mean(b, axis=axis),
        n_resamples=n_permutacoes, alternative="two-sided",
        random_state=np.random.default_rng(semente))
    return ResultadoTeste(
        titulo=f"Teste de aleatorização: {col1} × {col2}",
        h0="H₀: as duas amostras vêm da mesma distribuição "
           "(diferença de médias nula)",
        h1="H₁: as médias diferem",
        nome_estatistica="diferença de médias",
        estatistica=float(resultado.statistic),
        p_valor=float(resultado.pvalue),
        alfa=alfa,
        conclusao_h1=f"a média de '{col1}' difere da média de '{col2}'",
        amostras=[{"amostra": c, "n": int(v.size), "média": float(np.mean(v))}
                  for c, v in ((col1, x1), (col2, x2))],
        detalhes={"permutações": n_permutacoes},
        avisos=["O p-valor vem da distribuição de permutação — não exige "
                "normalidade nem variâncias iguais."],
    )


def arvore(df: pd.DataFrame, resposta: str, preditores: list[str],
           profundidade: int = 3) -> ResultadoComposto:
    """Árvore de classificação (resposta categórica) ou regressão (numérica)."""
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    from app.reports.formatacao import fmt

    if not 1 <= profundidade <= 8:
        raise ErroAnalise("Use profundidade entre 1 e 8.")
    x = df[preditores].apply(pd.to_numeric, errors="coerce")
    y_numerica = pd.to_numeric(df[resposta], errors="coerce")
    classificacao = y_numerica.isna().any() or y_numerica.nunique() <= 5
    if classificacao:
        y = df[resposta].astype(object)
        vazio = y.isna() | (y.astype(str).str.strip() == "")
        y = y.astype(str).where(~vazio, np.nan)
    else:
        y = y_numerica
    dados = pd.concat([y.rename("__y"), x], axis=1).dropna()
    if len(dados) < 10:
        raise ErroAnalise("São necessárias pelo menos 10 observações completas.")

    if classificacao:
        modelo = DecisionTreeClassifier(max_depth=profundidade, random_state=0)
        modelo.fit(dados[preditores], dados["__y"])
        acerto = float(modelo.score(dados[preditores], dados["__y"]))
        medida = ["taxa de acerto (resubstituição)", fmt(100 * acerto, 1) + "%"]
        tipo = "classificação"
    else:
        modelo = DecisionTreeRegressor(max_depth=profundidade, random_state=0)
        modelo.fit(dados[preditores], dados["__y"])
        r2 = float(modelo.score(dados[preditores], dados["__y"]))
        medida = ["R² (resubstituição)", fmt(100 * r2, 1) + "%"]
        tipo = "regressão"

    importancias = sorted(zip(preditores, modelo.feature_importances_),
                          key=lambda par: -par[1])
    linhas_imp = [[nome, fmt(100 * float(v), 1) + "%"]
                  for nome, v in importancias if v > 0]
    return ResultadoComposto(
        titulo=f"Árvore de {tipo}: {resposta}",
        itens=[
            ("tabela", ["item", "valor"],
             [["tipo", tipo], ["profundidade máxima", str(profundidade)],
              ["n", str(len(dados))], ["folhas", str(modelo.get_n_leaves())],
              medida]),
            ("subtitulo", "Importância dos preditores"),
            ("tabela", ["preditor", "importância"], linhas_imp),
            ("interpretacao",
             "A árvore divide os dados por regras simples nos preditores mais "
             "informativos. A medida de resubstituição é otimista — avalie com "
             "dados novos antes de usar para decisão."),
        ],
        dados={"modelo": modelo, "preditores": preditores,
               "classificacao": classificacao, "resposta": resposta},
    )
