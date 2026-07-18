"""Fumaça da interface para a Fase 3."""
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
    j._dados_exemplo_qualidade()
    yield j
    j.close()


def test_menus_fase3_presentes(janela):
    titulos = [acao.menu().title().replace("&", "")
               for acao in janela.menuBar().actions() if acao.menu()]
    for esperado in ("Cartas de Controle", "Qualidade", "MSA"):
        assert esperado in titulos


def test_dados_exemplo_qualidade_planilhas(janela):
    nomes = [janela.abas.tabText(i) for i in range(janela.abas.count())]
    assert "Processo (exemplo)" in nomes
    assert "MSA (exemplo)" in nomes


def test_carta_imr_estagios_na_sessao(janela):
    from app.core.cep import carta_i_mr
    from app.reports.formatacao import render_composto

    nomes = [janela.abas.tabText(i) for i in range(janela.abas.count())]
    janela.abas.setCurrentIndex(nomes.index("Processo (exemplo)"))
    planilha = janela._planilha_atual()
    cartas, resumo = carta_i_mr(planilha.valores("medida"), "medida",
                                estagios=planilha.valores("estagio"))
    janela.sessao.acrescentar(render_composto(resumo))
    assert "Carta I-MR" in janela.sessao.html_bruto()
    assert cartas[0].lc[0] != pytest.approx(cartas[0].lc[-1], abs=0.3)


def test_gage_rr_dados_exemplo(janela):
    from app.core.msa import gage_rr_cruzado

    nomes = [janela.abas.tabText(i) for i in range(janela.abas.count())]
    janela.abas.setCurrentIndex(nomes.index("MSA (exemplo)"))
    resultado = gage_rr_cruzado(janela._planilha_atual().df, "medição",
                                "peça", "operador")
    assert resultado.dados["ndc"] >= 4  # sistema de medição capaz no exemplo
    assert resultado.dados["pct_grr"] < 30


def test_planilha_coleta_cria_aba(janela):
    from app.core.msa import planilha_coleta_grr
    from app.worksheet.model import Planilha

    tabela = planilha_coleta_grr(4, 2, 2, semente=3)
    janela._nova_planilha(Planilha("Coleta Gage R&R", tabela))
    nomes = [janela.abas.tabText(i) for i in range(janela.abas.count())]
    assert "Coleta Gage R&R" in nomes
    assert janela._planilha_atual().coluna_e_numerica("ordem")
