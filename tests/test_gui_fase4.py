"""Fumaça da interface para a Fase 4."""
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
    j._dados_exemplo_fase4()
    yield j
    j.close()


def test_menus_fase4_presentes(janela):
    titulos = [acao.menu().title().replace("&", "")
               for acao in janela.menuBar().actions() if acao.menu()]
    for esperado in ("DOE", "Séries Temporais", "Multivariada",
                     "Confiabilidade"):
        assert esperado in titulos


def test_dados_exemplo_fase4_planilhas(janela):
    nomes = [janela.abas.tabText(i) for i in range(janela.abas.count())]
    for nome in ("DOE (exemplo)", "Série temporal (exemplo)",
                 "Confiabilidade (exemplo)"):
        assert nome in nomes


def test_fluxo_doe_completo(janela):
    from app.core.doe import analise_fatorial, otimizar_resposta
    from app.reports.formatacao import render_composto

    nomes = [janela.abas.tabText(i) for i in range(janela.abas.count())]
    janela.abas.setCurrentIndex(nomes.index("DOE (exemplo)"))
    resultado = analise_fatorial(janela._planilha_atual().df, "resposta",
                                 ["F1", "F2", "F3"])
    janela._modelo_doe = resultado.dados
    janela.sessao.acrescentar(render_composto(resultado))
    assert "Análise fatorial" in janela.sessao.html_bruto()
    # otimização usa o último modelo
    otimo = otimizar_resposta(janela._exigir_modelo_doe(), "maximizar")
    janela.sessao.acrescentar(render_composto(otimo))
    assert "Otimização" in janela.sessao.html_bruto()


def test_exigir_modelo_doe_sem_analise(janela):
    from app.core.resultados import ErroAnalise

    janela._modelo_doe = None
    with pytest.raises(ErroAnalise, match="análise fatorial"):
        janela._exigir_modelo_doe()


def test_serie_e_confiabilidade_dados_exemplo(janela):
    from app.core.confiabilidade import analise_parametrica
    from app.core.series import suavizacao

    nomes = [janela.abas.tabText(i) for i in range(janela.abas.count())]
    janela.abas.setCurrentIndex(nomes.index("Série temporal (exemplo)"))
    r_serie = suavizacao(janela._planilha_atual().valores("vendas"), "vendas",
                         "winters", periodo=12)
    assert r_serie.dados["previsao"].size == 6

    janela.abas.setCurrentIndex(nomes.index("Confiabilidade (exemplo)"))
    r_conf = analise_parametrica(janela._planilha_atual().df, "tempo_h",
                                 "falha", familia="Weibull")
    forma = float(r_conf.dados["ajuste"].rho_)
    assert 1.3 < forma < 2.4  # verdadeiro: 1,8
