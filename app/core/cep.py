"""Cartas de controle (CEP).

Constantes: c4 é calculada exatamente (função gama); d2 e d3 vêm da tabela
clássica (Montgomery, Introduction to Statistical Quality Control) para
subgrupos de 2 a 10 — acima disso, use a carta Xbarra-S, cujas constantes
são todas calculadas.

Testes de causas especiais: numeração de Nelson (1–8).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import gammaln

from .resultados import ErroAnalise, ResultadoComposto
from .util import limpar_numerica

# d2 e d3 para n = 2..10 (tabela clássica)
D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847,
      9: 2.970, 10: 3.078}
D3_TAB = {2: 0.853, 3: 0.888, 4: 0.880, 5: 0.864, 6: 0.848, 7: 0.833,
          8: 0.820, 9: 0.808, 10: 0.797}

TESTES_DESCRICAO = {
    1: "1 ponto além de 3σ do centro",
    2: "9 pontos seguidos do mesmo lado do centro",
    3: "6 pontos seguidos crescentes ou decrescentes",
    4: "14 pontos seguidos alternando para cima e para baixo",
    5: "2 de 3 pontos além de 2σ (mesmo lado)",
    6: "4 de 5 pontos além de 1σ (mesmo lado)",
    7: "15 pontos seguidos dentro de 1σ",
    8: "8 pontos seguidos além de 1σ (qualquer lado)",
}
TESTES_PADRAO = {1, 2, 3, 4}


def c4(n: int) -> float:
    return float(np.sqrt(2 / (n - 1))
                 * np.exp(gammaln(n / 2) - gammaln((n - 1) / 2)))


@dataclass
class CartaControle:
    nome: str
    rotulo_y: str
    pontos: np.ndarray
    lc: np.ndarray
    lsc: np.ndarray
    lic: np.ndarray
    violacoes: dict[int, list[int]] = field(default_factory=dict)
    extras: dict = field(default_factory=dict)

    def indices_violacao(self) -> set[int]:
        return {i for indices in self.violacoes.values() for i in indices}


# ------------------------------------------------------- causas especiais

def aplicar_testes(pontos: np.ndarray, lc: np.ndarray, lsc: np.ndarray,
                   lic: np.ndarray, testes: set[int]) -> dict[int, list[int]]:
    """Aplica os testes de Nelson selecionados; devolve teste → índices."""
    n = pontos.size
    sigma = (lsc - lc) / 3.0
    com_zonas = np.all(np.isfinite(sigma)) and np.all(sigma > 0)
    z = (pontos - lc) / sigma if com_zonas else np.zeros(n)
    lado = np.sign(pontos - lc)
    resultado: dict[int, list[int]] = {}

    def marca(teste: int, indice: int):
        resultado.setdefault(teste, []).append(indice)

    if 1 in testes:
        for i in range(n):
            if pontos[i] > lsc[i] or pontos[i] < lic[i]:
                marca(1, i)
    if 2 in testes:
        for i in range(8, n):
            janela = lado[i - 8:i + 1]
            if np.all(janela == janela[0]) and janela[0] != 0:
                marca(2, i)
    if 3 in testes:
        difs = np.diff(pontos)
        for i in range(5, n - 1 + 1):
            janela = difs[i - 5:i]
            if np.all(janela > 0) or np.all(janela < 0):
                marca(3, i)
    if 4 in testes:
        difs = np.sign(np.diff(pontos))
        for i in range(13, n):
            janela = difs[i - 13:i]
            if np.all(janela != 0) and np.all(janela[1:] * janela[:-1] < 0):
                marca(4, i)
    if com_zonas:
        if 5 in testes:
            for i in range(2, n):
                for sinal in (1, -1):
                    if np.sum(sinal * z[i - 2:i + 1] > 2) >= 2:
                        marca(5, i)
                        break
        if 6 in testes:
            for i in range(4, n):
                for sinal in (1, -1):
                    if np.sum(sinal * z[i - 4:i + 1] > 1) >= 4:
                        marca(6, i)
                        break
        if 7 in testes:
            for i in range(14, n):
                if np.all(np.abs(z[i - 14:i + 1]) < 1):
                    marca(7, i)
        if 8 in testes:
            for i in range(7, n):
                if np.all(np.abs(z[i - 7:i + 1]) > 1):
                    marca(8, i)
    return resultado


# ------------------------------------------------------------- utilitários

def _estagios_segmentos(estagios, n: int) -> list[np.ndarray]:
    """Índices de cada estágio (segmentos consecutivos), na ordem dos dados."""
    if estagios is None:
        return [np.arange(n)]
    rotulos = pd.Series(estagios).astype(str).to_numpy()[:n]
    segmentos, inicio = [], 0
    for i in range(1, n + 1):
        if i == n or rotulos[i] != rotulos[inicio]:
            segmentos.append(np.arange(inicio, i))
            inicio = i
    return segmentos


def _montar(nome: str, rotulo: str, pontos, lc, lsc, lic,
            testes: set[int]) -> CartaControle:
    pontos = np.asarray(pontos, dtype=float)
    lc = np.asarray(lc, dtype=float)
    lsc = np.asarray(lsc, dtype=float)
    lic = np.asarray(lic, dtype=float)
    carta = CartaControle(nome, rotulo, pontos, lc, lsc, lic)
    carta.violacoes = aplicar_testes(pontos, lc, lsc, lic, testes)
    return carta


def _resumo(titulo: str, cartas: list[CartaControle],
            notas: list[str]) -> ResultadoComposto:
    from app.reports.formatacao import fmt

    itens: list[tuple] = []
    for carta in cartas:
        itens.append(("subtitulo", carta.nome))
        itens.append(("tabela", ["item", "valor"],
                      [["linha central (último estágio)", fmt(float(carta.lc[-1]))],
                       ["LSC", fmt(float(carta.lsc[-1]))],
                       ["LIC", fmt(float(carta.lic[-1]))],
                       ["pontos", str(carta.pontos.size)],
                       ["pontos fora de controle",
                        str(len(carta.indices_violacao()))]]))
        if carta.violacoes:
            linhas = [[f"teste {t}", TESTES_DESCRICAO[t],
                       ", ".join(str(i + 1) for i in indices)]
                      for t, indices in sorted(carta.violacoes.items())]
            itens.append(("tabela", ["teste", "descrição", "pontos (posição)"],
                          linhas))
    total = sum(len(c.indices_violacao()) for c in cartas)
    if total:
        itens.append(("interpretacao",
                      f"Foram sinalizados {total} ponto(s) com padrão de causa "
                      "especial. Investigue o que ocorreu nesses momentos do "
                      "processo antes de recalcular os limites."))
    else:
        itens.append(("interpretacao",
                      "Nenhum teste de causa especial foi sinalizado: o processo "
                      "aparenta estar sob controle estatístico (apenas variação "
                      "de causas comuns)."))
    for nota in notas:
        itens.append(("nota", nota))
    return ResultadoComposto(titulo=titulo, itens=itens)


def _subagrupar(x: np.ndarray, tamanho: int) -> np.ndarray:
    if tamanho < 2:
        raise ErroAnalise("O tamanho do subgrupo deve ser pelo menos 2.")
    completos = (x.size // tamanho) * tamanho
    if completos < 2 * tamanho:
        raise ErroAnalise("Dados insuficientes: são necessários pelo menos 2 "
                          "subgrupos completos.")
    return x[:completos].reshape(-1, tamanho)


# ------------------------------------------------------------ cartas I-MR

def carta_i_mr(dados, coluna: str, estagios=None,
               testes: set[int] = TESTES_PADRAO):
    x = limpar_numerica(dados, coluna, n_minimo=5)
    segmentos = _estagios_segmentos(estagios, x.size)
    lc_i = np.empty(x.size)
    dist_i = np.empty(x.size)
    mr = np.full(x.size, np.nan)
    lc_mr = np.empty(x.size)
    lsc_mr = np.empty(x.size)
    for seg in segmentos:
        xs = x[seg]
        if xs.size < 2:
            raise ErroAnalise("Cada estágio precisa de pelo menos 2 observações.")
        mrs = np.abs(np.diff(xs))
        mr_barra = float(np.mean(mrs))
        mr[seg[1:]] = mrs
        lc_i[seg] = float(np.mean(xs))
        dist_i[seg] = 3 * mr_barra / D2[2]
        lc_mr[seg] = mr_barra
        lsc_mr[seg] = 3.267 * mr_barra  # D4 para n=2
    carta_i = _montar("Carta I (valores individuais)", coluna, x,
                      lc_i, lc_i + dist_i, lc_i - dist_i, testes)
    validos = ~np.isnan(mr)
    carta_mr = _montar("Carta MR (amplitude móvel)", "amplitude móvel",
                       np.where(validos, mr, np.nan), lc_mr, lsc_mr,
                       np.zeros(x.size), testes & {1})
    notas = ["σ estimado por MR-barra/d₂ (d₂ = 1,128)."]
    if estagios is not None:
        notas.append(f"Limites calculados por estágio ({len(segmentos)} estágios).")
    return [carta_i, carta_mr], _resumo(f"Carta I-MR: {coluna}",
                                        [carta_i, carta_mr], notas)


# --------------------------------------------------------- Xbarra-R / S

def _carta_xbar(dados, coluna: str, tamanho: int, usar_s: bool,
                estagios=None, testes: set[int] = TESTES_PADRAO):
    x = limpar_numerica(dados, coluna, n_minimo=4)
    if not usar_s and tamanho > 10:
        raise ErroAnalise("Para subgrupos maiores que 10, use a carta Xbarra-S.")
    matriz = _subagrupar(x, tamanho)
    medias = matriz.mean(axis=1)
    disp = matriz.std(axis=1, ddof=1) if usar_s else np.ptp(matriz, axis=1)
    k = medias.size

    estagios_sub = None
    if estagios is not None:
        rotulos = pd.Series(estagios).astype(str).to_numpy()
        estagios_sub = [rotulos[i * tamanho] for i in range(k)]
    segmentos = _estagios_segmentos(estagios_sub, k)

    lc_m = np.empty(k)
    dist_m = np.empty(k)
    lc_d = np.empty(k)
    lsc_d = np.empty(k)
    lic_d = np.empty(k)
    for seg in segmentos:
        if seg.size < 2:
            raise ErroAnalise("Cada estágio precisa de pelo menos 2 subgrupos.")
        media_geral = float(np.mean(medias[seg]))
        d_barra = float(np.mean(disp[seg]))
        lc_m[seg] = media_geral
        lc_d[seg] = d_barra
        if usar_s:
            c = c4(tamanho)
            sigma = d_barra / c
            fator = 3 * np.sqrt(1 - c**2) / c
            lsc_d[seg] = d_barra * (1 + fator)
            lic_d[seg] = max(0.0, d_barra * (1 - fator))
        else:
            sigma = d_barra / D2[tamanho]
            d3_rel = D3_TAB[tamanho] / D2[tamanho]
            lsc_d[seg] = d_barra * (1 + 3 * d3_rel)
            lic_d[seg] = max(0.0, d_barra * (1 - 3 * d3_rel))
        dist_m[seg] = 3 * sigma / np.sqrt(tamanho)

    nome_disp = "Carta S (desvio-padrão)" if usar_s else "Carta R (amplitude)"
    carta_m = _montar("Carta Xbarra (médias dos subgrupos)",
                      f"média de {coluna}", medias, lc_m, lc_m + dist_m,
                      lc_m - dist_m, testes)
    carta_d = _montar(nome_disp, "dispersão", disp, lc_d, lsc_d, lic_d,
                      testes & {1})
    sufixo = "S" if usar_s else "R"
    notas = [f"Subgrupos de tamanho {tamanho} formados por linhas consecutivas; "
             f"{k} subgrupos completos."]
    descartadas = x.size - k * tamanho
    if descartadas:
        notas.append(f"{descartadas} observação(ões) finais descartadas por não "
                     "completarem um subgrupo.")
    if estagios is not None:
        notas.append(f"Limites calculados por estágio ({len(segmentos)} estágios).")
    return [carta_m, carta_d], _resumo(f"Carta Xbarra-{sufixo}: {coluna}",
                                       [carta_m, carta_d], notas)


def carta_xbar_r(dados, coluna, tamanho, estagios=None, testes=TESTES_PADRAO):
    return _carta_xbar(dados, coluna, tamanho, usar_s=False, estagios=estagios,
                       testes=testes)


def carta_xbar_s(dados, coluna, tamanho, estagios=None, testes=TESTES_PADRAO):
    return _carta_xbar(dados, coluna, tamanho, usar_s=True, estagios=estagios,
                       testes=testes)


# ---------------------------------------------------------- por atributos

def _tamanhos(tamanhos, k: int, rotulo: str) -> np.ndarray:
    if np.isscalar(tamanhos):
        arr = np.full(k, float(tamanhos))
    else:
        arr = pd.to_numeric(pd.Series(tamanhos), errors="coerce").dropna().to_numpy()[:k]
        if arr.size != k:
            raise ErroAnalise(f"A coluna de tamanhos tem {arr.size} valores "
                              f"válidos, mas há {k} amostras de {rotulo}.")
    if np.any(arr <= 0):
        raise ErroAnalise("Todos os tamanhos de amostra devem ser positivos.")
    return arr


def carta_p(defeituosos, coluna: str, tamanhos, testes=TESTES_PADRAO,
            np_chart: bool = False):
    d = limpar_numerica(defeituosos, coluna, n_minimo=2)
    n = _tamanhos(tamanhos, d.size, coluna)
    if np.any(d > n):
        raise ErroAnalise("Há amostras com mais defeituosos que o tamanho da amostra.")
    p_barra = float(d.sum() / n.sum())
    if np_chart:
        if not np.allclose(n, n[0]):
            raise ErroAnalise("A carta NP exige tamanho de amostra constante; "
                              "use a carta P.")
        centro = p_barra * n[0]
        dist = 3 * np.sqrt(n[0] * p_barra * (1 - p_barra))
        carta = _montar("Carta NP (nº de defeituosos)", "defeituosos", d,
                        np.full(d.size, centro),
                        np.full(d.size, centro + dist),
                        np.full(d.size, max(0.0, centro - dist)), testes)
        titulo = f"Carta NP: {coluna}"
    else:
        p = d / n
        dist = 3 * np.sqrt(p_barra * (1 - p_barra) / n)
        carta = _montar("Carta P (proporção defeituosa)", "proporção", p,
                        np.full(d.size, p_barra),
                        np.minimum(1.0, p_barra + dist),
                        np.maximum(0.0, p_barra - dist), testes)
        titulo = f"Carta P: {coluna}"
    notas = [f"p-barra = {p_barra:.4f}".replace(".", ",")]
    if not np_chart and not np.allclose(n, n[0]):
        notas.append("Tamanhos de amostra variáveis: os limites variam por ponto.")
    return [carta], _resumo(titulo, [carta], notas)


def carta_c(contagens, coluna: str, tamanhos=None, testes=TESTES_PADRAO):
    """C (tamanho constante) ou U (por unidade) quando ``tamanhos`` é dado."""
    c = limpar_numerica(contagens, coluna, n_minimo=2)
    if np.any(c < 0):
        raise ErroAnalise("Contagens de defeitos não podem ser negativas.")
    if tamanhos is None:
        c_barra = float(np.mean(c))
        dist = 3 * np.sqrt(c_barra)
        carta = _montar("Carta C (nº de defeitos)", "defeitos", c,
                        np.full(c.size, c_barra),
                        np.full(c.size, c_barra + dist),
                        np.full(c.size, max(0.0, c_barra - dist)), testes)
        titulo = f"Carta C: {coluna}"
        notas = [f"c-barra = {c_barra:.4f}".replace(".", ",")]
    else:
        n = _tamanhos(tamanhos, c.size, coluna)
        u = c / n
        u_barra = float(c.sum() / n.sum())
        dist = 3 * np.sqrt(u_barra / n)
        carta = _montar("Carta U (defeitos por unidade)", "defeitos/unidade", u,
                        np.full(c.size, u_barra), u_barra + dist,
                        np.maximum(0.0, u_barra - dist), testes)
        titulo = f"Carta U: {coluna}"
        notas = [f"u-barra = {u_barra:.4f}".replace(".", ",")]
    return [carta], _resumo(titulo, [carta], notas)


# ------------------------------------------------- ponderadas no tempo

def _sigma_mr(x: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(x))) / D2[2])


def carta_ma(dados, coluna: str, janela: int = 3, testes={1}):
    x = limpar_numerica(dados, coluna, n_minimo=max(5, janela + 1))
    if janela < 2:
        raise ErroAnalise("A janela da média móvel deve ser pelo menos 2.")
    sigma = _sigma_mr(x)
    media = float(np.mean(x))
    ma = np.array([np.mean(x[max(0, i - janela + 1):i + 1])
                   for i in range(x.size)])
    n_i = np.minimum(np.arange(1, x.size + 1), janela)
    dist = 3 * sigma / np.sqrt(n_i)
    carta = _montar(f"Carta MA (média móvel, janela {janela})",
                    f"média móvel de {coluna}", ma,
                    np.full(x.size, media), media + dist, media - dist, testes)
    return [carta], _resumo(f"Carta MA: {coluna}", [carta],
                            ["σ estimado por MR-barra/d₂; limites mais largos "
                             "nos primeiros pontos (janela incompleta)."])


def carta_ewma(dados, coluna: str, lam: float = 0.2, testes={1}):
    x = limpar_numerica(dados, coluna, n_minimo=5)
    if not 0 < lam <= 1:
        raise ErroAnalise("λ deve estar entre 0 e 1 (usual: 0,1 a 0,3).")
    sigma = _sigma_mr(x)
    media = float(np.mean(x))
    z = np.empty(x.size)
    anterior = media
    for i in range(x.size):
        anterior = lam * x[i] + (1 - lam) * anterior
        z[i] = anterior
    i = np.arange(1, x.size + 1)
    dist = 3 * sigma * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * i)))
    carta = _montar(f"Carta EWMA (λ = {lam:g})".replace(".", ","),
                    f"EWMA de {coluna}", z, np.full(x.size, media),
                    media + dist, media - dist, testes)
    return [carta], _resumo(f"Carta EWMA: {coluna}", [carta],
                            ["EWMA detecta pequenos deslocamentos da média mais "
                             "rápido que a carta I."])


def carta_cusum(dados, coluna: str, alvo: float | None = None, k: float = 0.5,
                h: float = 4.0):
    x = limpar_numerica(dados, coluna, n_minimo=5)
    sigma = _sigma_mr(x)
    if alvo is None:
        alvo = float(np.mean(x))
    folga = k * sigma
    limite = h * sigma
    c_mais = np.empty(x.size)
    c_menos = np.empty(x.size)
    cm = cn = 0.0
    for i in range(x.size):
        cm = max(0.0, x[i] - (alvo + folga) + cm)
        cn = max(0.0, (alvo - folga) - x[i] + cn)
        c_mais[i] = cm
        c_menos[i] = -cn
    pontos = np.where(np.abs(c_mais) >= np.abs(c_menos), c_mais, c_menos)
    carta = CartaControle(
        f"Carta CUSUM (k = {k:g}σ, h = {h:g}σ)".replace(".", ","),
        f"soma acumulada de {coluna}", pontos,
        np.zeros(x.size), np.full(x.size, limite), np.full(x.size, -limite))
    fora = [i for i in range(x.size)
            if c_mais[i] > limite or c_menos[i] < -limite]
    if fora:
        carta.violacoes = {1: fora}
    carta.extras = {"c_mais": c_mais, "c_menos": c_menos}
    from app.reports.formatacao import fmt

    return [carta], _resumo(
        f"Carta CUSUM: {coluna}", [carta],
        [f"Alvo = {fmt(float(alvo))}; σ estimado por MR-barra/d₂. A CUSUM "
         "acumula desvios do alvo e detecta pequenos deslocamentos persistentes."])
