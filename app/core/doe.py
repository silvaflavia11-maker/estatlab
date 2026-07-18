"""Planejamento de experimentos (DOE).

Geração de planos: fatoriais 2^k (completos e frações via pyDOE3), Plackett-
Burman, fatorial geral, superfície de resposta (CCD e Box-Behnken) e arranjos
de Taguchi (L4, L8, L9, L12, L16 — construídos de frações regulares/PB; L9 é
o arranjo ortogonal clássico de 3 níveis).

Análise: modelo linear com efeitos = 2·coeficiente (fatores codificados
−1/+1). Sem graus de liberdade para o erro (plano saturado/sem réplicas), a
significância usa o método de Lenth (1989). Corridas perdidas são tratadas
por regressão (o plano fica desbalanceado, mas o ajuste permanece válido).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats

from .resultados import ErroAnalise, ResultadoComposto


# ------------------------------------------------------------ geração

def _aleatorizar(matriz: np.ndarray, aleatorizar: bool,
                 semente: int | None) -> tuple[np.ndarray, np.ndarray]:
    ordem_padrao = np.arange(1, len(matriz) + 1)
    if not aleatorizar:
        return matriz, ordem_padrao
    rng = np.random.default_rng(semente)
    ordem = rng.permutation(len(matriz))
    return matriz[ordem], ordem_padrao[ordem]


def _montar_plano(matriz: np.ndarray, nomes: list[str], replicas: int,
                  aleatorizar: bool, semente: int | None) -> pd.DataFrame:
    matriz = np.tile(matriz, (replicas, 1))
    matriz, ordem_padrao = _aleatorizar(matriz, aleatorizar, semente)
    plano = pd.DataFrame(matriz, columns=nomes)
    plano.insert(0, "ordem_padrao", ordem_padrao)
    plano.insert(0, "ordem_execucao", np.arange(1, len(plano) + 1))
    plano["resposta"] = np.nan
    return plano


def gerar_fatorial_2k(n_fatores: int, replicas: int = 1,
                      fracao: int = 0, aleatorizar: bool = True,
                      semente: int | None = None) -> pd.DataFrame:
    """Fatorial 2^(k−fracao); fração regular de resolução máxima (pyDOE3)."""
    if not 2 <= n_fatores <= 9:
        raise ErroAnalise("Use de 2 a 9 fatores.")
    if fracao < 0 or n_fatores - fracao < 2:
        raise ErroAnalise("Fração inválida: k − fração deve ser ≥ 2.")
    nomes = [f"F{i + 1}" for i in range(n_fatores)]
    if fracao == 0:
        from pyDOE3 import ff2n

        matriz = ff2n(n_fatores)
    else:
        from pyDOE3 import fracfact

        base = n_fatores - fracao
        letras = [chr(97 + i) for i in range(base)]
        geradores = list(letras)
        # geradores das colunas extras: interações de maior ordem primeiro
        candidatos = ["".join(c) for tamanho in range(base, 1, -1)
                      for c in combinations(letras, tamanho)]
        geradores += candidatos[:fracao]
        matriz = fracfact(" ".join(geradores))
    return _montar_plano(matriz, nomes, replicas, aleatorizar, semente)


def gerar_plackett_burman(n_fatores: int, aleatorizar: bool = True,
                          semente: int | None = None) -> pd.DataFrame:
    if not 2 <= n_fatores <= 23:
        raise ErroAnalise("Plackett-Burman: use de 2 a 23 fatores.")
    from pyDOE3 import pbdesign

    matriz = pbdesign(n_fatores)
    nomes = [f"F{i + 1}" for i in range(n_fatores)]
    return _montar_plano(matriz[:, :n_fatores], nomes, 1, aleatorizar, semente)


def gerar_fatorial_geral(niveis: list[int], replicas: int = 1,
                         aleatorizar: bool = True,
                         semente: int | None = None) -> pd.DataFrame:
    if not 2 <= len(niveis) <= 6 or any(nv < 2 for nv in niveis):
        raise ErroAnalise("Use 2 a 6 fatores, cada um com pelo menos 2 níveis.")
    from pyDOE3 import fullfact

    matriz = fullfact(niveis) + 1  # níveis 1..n
    nomes = [f"F{i + 1}" for i in range(len(niveis))]
    return _montar_plano(matriz, nomes, replicas, aleatorizar, semente)


def gerar_superficie(n_fatores: int, tipo: str = "ccd",
                     aleatorizar: bool = True,
                     semente: int | None = None) -> pd.DataFrame:
    if tipo == "ccd":
        if not 2 <= n_fatores <= 6:
            raise ErroAnalise("CCD: use de 2 a 6 fatores.")
        from pyDOE3 import ccdesign

        matriz = ccdesign(n_fatores, center=(4, 4), alpha="rotatable",
                          face="ccc")
    elif tipo == "box-behnken":
        if not 3 <= n_fatores <= 6:
            raise ErroAnalise("Box-Behnken: use de 3 a 6 fatores.")
        from pyDOE3 import bbdesign

        matriz = bbdesign(n_fatores, center=3)
    else:
        raise ErroAnalise("Tipo inválido: use 'ccd' ou 'box-behnken'.")
    nomes = [f"F{i + 1}" for i in range(n_fatores)]
    return _montar_plano(np.round(matriz, 4), nomes, 1, aleatorizar, semente)


_L9 = np.array([[1, 1, 1, 1], [1, 2, 2, 2], [1, 3, 3, 3],
                [2, 1, 2, 3], [2, 2, 3, 1], [2, 3, 1, 2],
                [3, 1, 3, 2], [3, 2, 1, 3], [3, 3, 2, 1]])


def gerar_taguchi(arranjo: str) -> pd.DataFrame:
    """Arranjos ortogonais L4, L8, L9, L12 e L16 (colunas = fatores)."""
    from pyDOE3 import fracfact, pbdesign

    arranjo = arranjo.upper()
    if arranjo == "L4":
        matriz = fracfact("a b ab")
    elif arranjo == "L8":
        matriz = fracfact("a b ab c ac bc abc")
    elif arranjo == "L16":
        letras = ["a", "b", "c", "d"]
        termos = [
            "".join(c) for tamanho in range(1, 5)
            for c in combinations(letras, tamanho)
        ]
        matriz = fracfact(" ".join(termos))
    elif arranjo == "L12":
        matriz = pbdesign(11)
    elif arranjo == "L9":
        matriz = _L9.astype(float)
    else:
        raise ErroAnalise("Arranjo inválido: use L4, L8, L9, L12 ou L16.")
    if arranjo != "L9":
        matriz = np.where(matriz < 0, 1, 2)  # níveis 1/2
    nomes = [f"F{i + 1}" for i in range(matriz.shape[1])]
    plano = pd.DataFrame(matriz.astype(int), columns=nomes)
    plano.insert(0, "corrida", np.arange(1, len(plano) + 1))
    plano["resposta_1"] = np.nan
    plano["resposta_2"] = np.nan
    return plano


# ------------------------------------------------------------- análise

def _codificar(df: pd.DataFrame, fatores: list[str]) -> tuple[pd.DataFrame, dict]:
    """Codifica cada fator para −1/+1 (2 níveis distintos exigidos)."""
    codificado = pd.DataFrame(index=df.index)
    niveis_reais: dict[str, tuple] = {}
    for fator in fatores:
        valores = pd.to_numeric(df[fator], errors="coerce")
        if valores.isna().any() and df[fator].notna().any():
            valores = df[fator].astype(str)
            valores = valores.where(valores.str.strip() != "", np.nan)
        unicos = sorted(pd.Series(valores).dropna().unique())
        if len(unicos) != 2:
            raise ErroAnalise(
                f"O fator '{fator}' precisa ter exatamente 2 níveis "
                f"(encontrados: {len(unicos)}). Para mais níveis, use ANOVA.")
        mapa = {unicos[0]: -1.0, unicos[1]: 1.0}
        codificado[fator] = pd.Series(valores).map(mapa)
        niveis_reais[fator] = (unicos[0], unicos[1])
    return codificado, niveis_reais


def _matriz_modelo(codificado: pd.DataFrame, fatores: list[str],
                   ordem_interacao: int) -> pd.DataFrame:
    matriz = codificado[fatores].copy()
    for tamanho in range(2, ordem_interacao + 1):
        for combo in combinations(fatores, tamanho):
            matriz["*".join(combo)] = codificado[list(combo)].prod(axis=1)
    return matriz


def _lenth(efeitos: np.ndarray) -> tuple[float, float]:
    """PSE e margem de erro (α=0,05) de Lenth para planos sem réplicas."""
    abs_efeitos = np.abs(efeitos)
    s0 = 1.5 * np.median(abs_efeitos)
    pse = 1.5 * np.median(abs_efeitos[abs_efeitos <= 2.5 * s0])
    gl = len(efeitos) / 3
    return float(pse), float(stats.t.ppf(0.975, gl) * pse)


def analise_fatorial(df: pd.DataFrame, resposta: str, fatores: list[str],
                     ordem_interacao: int = 2,
                     alfa: float = 0.05) -> ResultadoComposto:
    from app.reports.formatacao import fmt, fmt_p

    if len(fatores) < 2:
        raise ErroAnalise("Selecione pelo menos 2 fatores.")
    y = pd.to_numeric(df[resposta], errors="coerce")
    codificado, niveis = _codificar(df, fatores)
    dados = pd.concat([y.rename("__y"), codificado], axis=1).dropna()
    perdidas = len(df.dropna(subset=fatores, how="any")) - len(dados)
    if len(dados) < len(fatores) + 2:
        raise ErroAnalise("Corridas com resposta insuficientes para o ajuste.")

    matriz = _matriz_modelo(dados[fatores], fatores,
                            min(ordem_interacao, len(fatores)))
    if len(dados) <= matriz.shape[1]:
        raise ErroAnalise("Mais termos que corridas: reduza a ordem das "
                          "interações ou acrescente réplicas.")
    modelo = sm.OLS(dados["__y"], sm.add_constant(matriz)).fit()
    termos = list(matriz.columns)
    efeitos = 2 * modelo.params[termos]

    itens: list[tuple] = [
        ("nota", "Níveis codificados: "
                 + "; ".join(f"{fator}: {niveis[fator][0]} → −1, "
                             f"{niveis[fator][1]} → +1" for fator in fatores)),
    ]
    if perdidas > 0:
        itens.append(("aviso", f"{perdidas} corrida(s) sem resposta foram "
                               "excluídas (corridas perdidas); o plano ficou "
                               "desbalanceado e os efeitos são estimados por "
                               "regressão."))

    sem_erro = modelo.df_resid == 0
    linhas = []
    if sem_erro:
        pse, margem = _lenth(efeitos.to_numpy())
        for termo in termos:
            significativo = abs(efeitos[termo]) > margem
            linhas.append([termo, fmt(float(efeitos[termo])),
                           fmt(float(modelo.params[termo])),
                           "significativo" if significativo else "—"])
        itens += [
            ("subtitulo", "Efeitos estimados (método de Lenth — sem réplicas)"),
            ("tabela", ["termo", "efeito", "coeficiente", "Lenth (α = 0,05)"],
             linhas),
            ("nota", f"PSE de Lenth = {fmt(pse)}; margem de erro = {fmt(margem)}. "
                     "Efeitos além da margem são considerados ativos."),
        ]
        significativos = [t for t in termos if abs(efeitos[t]) > margem]
    else:
        for termo in termos:
            p = float(modelo.pvalues[termo])
            linhas.append([termo, fmt(float(efeitos[termo])),
                           fmt(float(modelo.params[termo])),
                           fmt(float(modelo.bse[termo])), fmt_p(p),
                           "significativo" if p < alfa else "—"])
        itens += [
            ("subtitulo", "Efeitos estimados"),
            ("tabela", ["termo", "efeito", "coeficiente", "EP", "p-valor", ""],
             linhas),
            ("nota", f"R² = {fmt(100 * modelo.rsquared, 2)}%  •  "
                     f"R² ajustado = {fmt(100 * modelo.rsquared_adj, 2)}%  •  "
                     f"s = {fmt(float(np.sqrt(modelo.mse_resid)))}"),
        ]
        significativos = [t for t in termos if modelo.pvalues[t] < alfa]

    itens.append(("interpretacao",
                  ("Termos ativos: " + ", ".join(significativos) + ". "
                   if significativos else "Nenhum termo ativo detectado. ")
                  + "O efeito é a variação média na resposta ao passar o fator "
                    "do nível −1 para +1."))
    return ResultadoComposto(
        titulo=f"Análise fatorial (2 níveis): {resposta}",
        itens=itens,
        dados={"efeitos": {t: float(efeitos[t]) for t in termos},
               "modelo": modelo, "fatores": fatores, "niveis": niveis,
               "matriz_colunas": termos, "resposta": resposta,
               "residuos": np.asarray(modelo.resid),
               "ajustados": np.asarray(modelo.fittedvalues),
               "codificado": dados, "sem_erro": sem_erro,
               "margem_lenth": margem if sem_erro else None},
    )


def analise_superficie(df: pd.DataFrame, resposta: str,
                       fatores: list[str], alfa: float = 0.05) -> ResultadoComposto:
    """Modelo quadrático completo (superfície de resposta)."""
    from app.reports.formatacao import fmt, fmt_p

    if not 2 <= len(fatores) <= 6:
        raise ErroAnalise("Use de 2 a 6 fatores.")
    y = pd.to_numeric(df[resposta], errors="coerce")
    x = df[fatores].apply(pd.to_numeric, errors="coerce")
    dados = pd.concat([y.rename("__y"), x], axis=1).dropna()

    matriz = dados[fatores].copy()
    for f1, f2 in combinations(fatores, 2):
        matriz[f"{f1}*{f2}"] = dados[f1] * dados[f2]
    for fator in fatores:
        matriz[f"{fator}²"] = dados[fator] ** 2
    if len(dados) <= matriz.shape[1] + 1:
        raise ErroAnalise("Corridas insuficientes para o modelo quadrático "
                          f"({matriz.shape[1] + 1} termos).")
    modelo = sm.OLS(dados["__y"], sm.add_constant(matriz)).fit()

    linhas = [[("constante" if t == "const" else t),
               fmt(float(modelo.params[t])), fmt(float(modelo.bse[t])),
               fmt_p(float(modelo.pvalues[t]))]
              for t in modelo.params.index]
    return ResultadoComposto(
        titulo=f"Superfície de resposta: {resposta}",
        itens=[
            ("subtitulo", "Coeficientes do modelo quadrático"),
            ("tabela", ["termo", "coeficiente", "EP", "p-valor"], linhas),
            ("nota", f"R² = {fmt(100 * modelo.rsquared, 2)}%  •  "
                     f"R² ajustado = {fmt(100 * modelo.rsquared_adj, 2)}%  •  "
                     f"n = {int(modelo.nobs)}"),
            ("interpretacao", "Termos quadráticos significativos indicam "
                              "curvatura — a resposta tem máximo, mínimo ou sela "
                              "na região estudada. Use os gráficos de contorno/"
                              "superfície e a otimização para localizá-los."),
        ],
        dados={"modelo": modelo, "fatores": fatores, "resposta": resposta,
               "x": dados[fatores], "residuos": np.asarray(modelo.resid),
               "ajustados": np.asarray(modelo.fittedvalues)},
    )


def prever_modelo(dados_modelo: dict, valores: dict[str, float]) -> float:
    """Prevê a resposta de um modelo fatorial/superfície em um ponto."""
    fatores = dados_modelo["fatores"]
    modelo = dados_modelo["modelo"]
    linha = {"const": 1.0}
    for nome in modelo.params.index:
        if nome == "const":
            continue
        if nome.endswith("²"):
            linha[nome] = valores[nome[:-1]] ** 2
        elif "*" in nome:
            produto = 1.0
            for parte in nome.split("*"):
                produto *= valores[parte]
            linha[nome] = produto
        else:
            linha[nome] = valores[nome]
    return float(sum(modelo.params[nome] * valor
                     for nome, valor in linha.items()))


def otimizar_resposta(dados_modelo: dict, objetivo: str = "maximizar",
                      alvo: float | None = None) -> ResultadoComposto:
    """Otimização por desirability em uma grade + refinamento local.

    Região de busca: intervalo observado de cada fator no plano.
    """
    from app.reports.formatacao import fmt

    fatores = dados_modelo["fatores"]
    if objetivo not in ("maximizar", "minimizar", "alvo"):
        raise ErroAnalise("Objetivo inválido.")
    if objetivo == "alvo" and alvo is None:
        raise ErroAnalise("Informe o valor-alvo.")
    base = dados_modelo.get("x", dados_modelo.get("codificado"))
    limites = [(float(base[f].min()), float(base[f].max())) for f in fatores]

    def prever_ponto(ponto: np.ndarray) -> float:
        return prever_modelo(dados_modelo, dict(zip(fatores, ponto)))

    sinal = -1.0 if objetivo == "maximizar" else 1.0

    def funcao_objetivo(ponto: np.ndarray) -> float:
        y = prever_ponto(ponto)
        if objetivo == "alvo":
            return (y - alvo) ** 2
        return sinal * y

    # multi-início: grade grossa + refinamento local
    grade = [np.linspace(lo, hi, 5) for lo, hi in limites]
    melhor = None
    for ponto in np.stack(np.meshgrid(*grade), axis=-1).reshape(-1, len(fatores)):
        resultado = optimize.minimize(funcao_objetivo, ponto, method="L-BFGS-B",
                                      bounds=limites)
        if melhor is None or resultado.fun < melhor.fun:
            melhor = resultado
    otimo = melhor.x
    y_otimo = prever_ponto(otimo)

    descricao = {"maximizar": "maximizar a resposta",
                 "minimizar": "minimizar a resposta",
                 "alvo": f"atingir o alvo {fmt(alvo) if alvo is not None else ''}"}
    return ResultadoComposto(
        titulo=f"Otimização de resposta: {dados_modelo['resposta']}",
        itens=[
            ("tabela", ["fator", "ajuste ótimo"],
             [[f, fmt(float(v))] for f, v in zip(fatores, otimo)]),
            ("tabela", ["item", "valor"],
             [["objetivo", descricao[objetivo]],
              ["resposta prevista no ótimo", fmt(y_otimo)]]),
            ("interpretacao",
             "Ajustes ótimos dentro da região experimental estudada. Confirme "
             "com corridas de verificação no ponto recomendado antes de adotar."),
        ],
    )


def analise_taguchi(df: pd.DataFrame, fatores: list[str],
                    respostas: list[str],
                    criterio: str = "maior-melhor") -> ResultadoComposto:
    """Razões sinal-ruído por corrida e efeitos dos fatores sobre S/N."""
    from app.reports.formatacao import fmt

    if not respostas:
        raise ErroAnalise("Selecione pelo menos 1 coluna de resposta.")
    matriz_y = df[respostas].apply(pd.to_numeric, errors="coerce")
    fatores_df = df[fatores].astype(object)
    completos = pd.concat([fatores_df, matriz_y], axis=1).dropna()
    if len(completos) < 4:
        raise ErroAnalise("Corridas completas insuficientes.")
    y = completos[respostas].to_numpy(dtype=float)

    if criterio == "maior-melhor":
        if np.any(y <= 0):
            raise ErroAnalise("'Maior é melhor' exige respostas positivas.")
        sn = -10 * np.log10(np.mean(1 / y**2, axis=1))
    elif criterio == "menor-melhor":
        sn = -10 * np.log10(np.mean(y**2, axis=1))
    elif criterio == "nominal-melhor":
        medias = y.mean(axis=1)
        desvios = y.std(axis=1, ddof=1)
        if np.any(desvios == 0):
            raise ErroAnalise("'Nominal é melhor' exige variação entre réplicas.")
        sn = 10 * np.log10(medias**2 / desvios**2)
    else:
        raise ErroAnalise("Critério inválido.")

    linhas_efeito = []
    ranques = []
    for fator in fatores:
        medias_nivel = pd.Series(sn).groupby(
            completos[fator].astype(str).to_numpy()).mean()
        delta = float(medias_nivel.max() - medias_nivel.min())
        ranques.append((delta, fator, medias_nivel))
    ranques.sort(reverse=True)
    for posicao, (delta, fator, medias_nivel) in enumerate(ranques, start=1):
        linhas_efeito.append(
            [fator,
             "; ".join(f"{nivel}: {fmt(float(v), 2)}"
                       for nivel, v in medias_nivel.items()),
             fmt(delta, 2), str(posicao)])
    melhor_niveis = {fator: str(medias.idxmax())
                     for _, fator, medias in ranques}
    return ResultadoComposto(
        titulo="Análise de Taguchi (razão sinal-ruído)",
        itens=[
            ("nota", f"Critério: {criterio} — S/N maior é sempre melhor."),
            ("subtitulo", "Efeitos dos fatores sobre a razão S/N"),
            ("tabela", ["fator", "S/N médio por nível", "delta", "ranque"],
             linhas_efeito),
            ("interpretacao",
             "Ajuste recomendado (maior S/N): "
             + "; ".join(f"{f} = {n}" for f, n in melhor_niveis.items())
             + ". O ranque indica a ordem de influência na robustez."),
        ],
        dados={"sn": sn, "fatores": fatores, "dados": completos},
    )
