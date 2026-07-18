"""Formatação PT-BR e renderização HTML dos resultados para a sessão."""
from __future__ import annotations

import html
import math

from app.core.resultados import (
    ResultadoComposto,
    ResultadoDescritiva,
    ResultadoTabela,
    ResultadoTeste,
)

CSS_SESSAO = """
h3 { color: #1f4e5f; margin-bottom: 2px; }
table.resultado { border-collapse: collapse; margin: 6px 0; }
table.resultado th { background-color: #1f4e5f; color: white; padding: 3px 10px;
                     border: 1px solid #9bb7c0; }
table.resultado td { padding: 3px 10px; border: 1px solid #b8cdd4; }
p.hipoteses { margin: 4px 0; }
p.decisao { background-color: #eef6f2; padding: 6px 8px; border-left: 4px solid #2e8b57; }
p.decisao-rejeita { background-color: #fdf1ec; border-left: 4px solid #c0563a;
                    padding: 6px 8px; }
p.aviso { color: #8a5a00; background-color: #fff8e6; padding: 4px 8px;
          border-left: 4px solid #d9a300; }
hr { border: 0; border-top: 1px solid #cccccc; margin: 14px 0; }
"""


def fmt(x, dec: int = 4) -> str:
    """Número em formato PT-BR: vírgula decimal e ponto de milhar."""
    if x is None:
        return "—"
    if isinstance(x, float):
        if math.isnan(x):
            return "—"
        if math.isinf(x):
            return "+∞" if x > 0 else "−∞"
    if isinstance(x, int):
        return f"{x:,}".replace(",", ".")
    texto = f"{x:,.{dec}f}"
    return texto.replace(",", "\0").replace(".", ",").replace("\0", ".")


def fmt_p(p: float) -> str:
    if p is None or math.isnan(p):
        return "—"
    if p < 0.001:
        return "< 0,001"
    return fmt(p, 3)


def _esc(texto) -> str:
    return html.escape(str(texto))


def _tabela(cabecalhos: list[str], linhas: list[list[str]]) -> str:
    ths = "".join(f"<th>{_esc(c)}</th>" for c in cabecalhos)
    trs = "".join(
        "<tr>" + "".join(f"<td>{_esc(v)}</td>" for v in linha) + "</tr>"
        for linha in linhas
    )
    return f'<table class="resultado"><tr>{ths}</tr>{trs}</table>'


def _fmt_celula(v) -> str:
    if isinstance(v, float):
        return fmt(v)
    if isinstance(v, int):
        return fmt(v)
    return str(v)


def render_teste(r: ResultadoTeste) -> str:
    """HTML de um teste de hipóteses com interpretação didática."""
    partes = [f"<h3>{_esc(r.titulo)}</h3>"]

    partes.append(
        f'<p class="hipoteses"><b>Hipóteses:</b><br>{_esc(r.h0)}<br>{_esc(r.h1)}</p>'
    )

    if r.amostras:
        cabecalhos = list(r.amostras[0].keys())
        linhas = [[_fmt_celula(a[c]) for c in cabecalhos] for a in r.amostras]
        partes.append(_tabela(cabecalhos, linhas))

    resumo = [["estatística do teste", f"{r.nome_estatistica} = {fmt(r.estatistica)}"]]
    if r.gl is not None:
        gl = int(r.gl) if float(r.gl).is_integer() else r.gl
        resumo.append(["graus de liberdade", fmt(gl, 2) if isinstance(gl, float) else fmt(gl)])
    resumo.append(["p-valor", fmt_p(r.p_valor)])
    if r.ic is not None and r.nivel_confianca:
        resumo.append([
            f"IC {fmt(100 * r.nivel_confianca, 0)}%",
            f"({fmt(r.ic[0])}; {fmt(r.ic[1])})" + (f" — {r.descricao_ic}" if r.descricao_ic else ""),
        ])
    for chave, valor in r.detalhes.items():
        resumo.append([chave, _fmt_celula(valor)])
    partes.append(_tabela(["resultado", "valor"], resumo))

    # Interpretação didática
    p_txt, alfa_txt = fmt_p(r.p_valor), fmt(r.alfa, 2)
    if r.p_valor < r.alfa:
        partes.append(
            f'<p class="decisao-rejeita"><b>Decisão:</b> como p = {p_txt} &lt; '
            f"α = {alfa_txt}, <b>rejeita-se H₀</b>: há evidência de que "
            f"{_esc(r.conclusao_h1)}.</p>"
        )
    else:
        partes.append(
            f'<p class="decisao"><b>Decisão:</b> como p = {p_txt} ≥ α = {alfa_txt}, '
            f"<b>não se rejeita H₀</b>: não há evidência suficiente de que "
            f"{_esc(r.conclusao_h1)}. Atenção: não rejeitar H₀ não prova que H₀ "
            f"seja verdadeira.</p>"
        )
    for aviso in r.avisos:
        partes.append(f'<p class="aviso">⚠ {_esc(aviso)}</p>')
    partes.append("<hr>")
    return "".join(partes)


def render_descritiva(r: ResultadoDescritiva) -> str:
    linhas = [
        ["n (válidos)", fmt(r.n)],
        ["valores ausentes", fmt(r.ausentes)],
        ["média", fmt(r.media)],
        ["EP da média", fmt(r.ep_media)],
        ["desvio-padrão", fmt(r.dp)],
        ["variância", fmt(r.variancia)],
        ["mínimo", fmt(r.minimo)],
        ["1º quartil (Q1)", fmt(r.q1)],
        ["mediana", fmt(r.mediana)],
        ["3º quartil (Q3)", fmt(r.q3)],
        ["máximo", fmt(r.maximo)],
        ["amplitude", fmt(r.amplitude)],
        ["amplitude interquartil", fmt(r.aiq)],
        ["assimetria", fmt(r.assimetria)],
        ["curtose", fmt(r.curtose)],
        [f"IC {fmt(100 * r.nivel_confianca, 0)}% para a média",
         f"({fmt(r.ic_media[0])}; {fmt(r.ic_media[1])})"],
    ]
    interpretacao = (
        f"Com {fmt(100 * r.nivel_confianca, 0)}% de confiança, a média populacional de "
        f"'{_esc(r.coluna)}' está entre {fmt(r.ic_media[0])} e {fmt(r.ic_media[1])}. "
    )
    if abs(r.assimetria) < 0.5:
        interpretacao += "A distribuição é aproximadamente simétrica."
    elif r.assimetria > 0:
        interpretacao += "A distribuição é assimétrica à direita (cauda para valores altos)."
    else:
        interpretacao += "A distribuição é assimétrica à esquerda (cauda para valores baixos)."
    return (
        f"<h3>Estatística descritiva: {_esc(r.coluna)}</h3>"
        + _tabela(["medida", "valor"], linhas)
        + f'<p class="decisao"><b>Interpretação:</b> {interpretacao}</p><hr>'
    )


def render_composto(r: ResultadoComposto) -> str:
    partes = [f"<h3>{_esc(r.titulo)}</h3>"]
    for item in r.itens:
        tipo = item[0]
        if tipo == "subtitulo":
            partes.append(f"<p><b>{_esc(item[1])}</b></p>")
        elif tipo == "tabela":
            partes.append(_tabela(item[1], [[_fmt_celula(v) for v in linha]
                                            for linha in item[2]]))
        elif tipo == "interpretacao":
            partes.append(f'<p class="decisao"><b>Interpretação:</b> {_esc(item[1])}</p>')
        elif tipo == "nota":
            partes.append(f'<p class="hipoteses">• {_esc(item[1])}</p>')
        elif tipo == "aviso":
            partes.append(f'<p class="aviso">⚠ {_esc(item[1])}</p>')
    partes.append("<hr>")
    return "".join(partes)


def render_tabela(r: ResultadoTabela) -> str:
    partes = [f"<h3>{_esc(r.titulo)}</h3>", _tabela(r.cabecalhos, r.linhas)]
    for nota in r.notas:
        partes.append(f'<p class="hipoteses">• {_esc(nota)}</p>')
    partes.append("<hr>")
    return "".join(partes)
