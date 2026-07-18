"""Fumaça da interface para a Fase 2: menus presentes e fluxo core→sessão."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def aplicacao():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def janela(aplicacao):
    from app.ui.janela_principal import JanelaPrincipal

    j = JanelaPrincipal()
    j._dados_exemplo()
    yield j
    j.close()


def test_menus_fase2_presentes(janela):
    titulos = [acao.menu().title().replace("&", "")
               for acao in janela.menuBar().actions() if acao.menu()]
    for esperado in ("Regressão", "ANOVA", "Não Paramétricos", "Tabelas",
                     "Distribuições", "Poder e Amostra"):
        assert esperado in titulos


def test_anova_com_graficos_na_sessao(janela):
    from app.core.anova import anova_1via
    from app.reports.formatacao import render_composto

    resultado = anova_1via(janela._planilha_atual().df, "nota_prova2", "turma",
                           comparacao="tukey")
    janela.sessao.acrescentar(render_composto(resultado))
    assert "ANOVA de 1 fator" in janela.sessao.html_bruto()
    janela._graficos_residuos(resultado, "teste")
    assert len(janela._janelas_graficos) == 1


def test_regressao_e_logistica_com_dados_exemplo(janela):
    from app.core.regressao import regressao_linear, regressao_logistica

    df = janela._planilha_atual().df
    r1 = regressao_linear(df, "peso_kg", ["altura_cm"])
    assert any("R²" in str(item) for item in r1.itens)
    r2 = regressao_logistica(df, "aprovado", ["nota_prova1"])
    assert "aprovado" in r2.titulo


def test_escrever_coluna_gerada(janela):
    from app.core.distribuicoes import gerar_aleatorios

    valores = gerar_aleatorios("Normal", [10.0, 2.0], 50, semente=3)
    janela._escrever_coluna("simulado", valores)
    planilha = janela._planilha_atual()
    assert "simulado" in planilha.colunas_numericas()
    assert planilha.valores("simulado").dropna().size == 50
    # nome duplicado ganha sufixo
    janela._escrever_coluna("simulado", valores)
    assert "simulado_2" in planilha.colunas()
