"""Teste de fumaça da interface: instancia a janela e exercita fluxos básicos
sem display (plataforma offscreen do Qt)."""
import os

import numpy as np
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
    yield j
    j.close()


def test_janela_abre_com_planilha_vazia(janela):
    assert janela.abas.count() == 1
    modelo = janela._modelo_atual()
    assert modelo.rowCount() >= 100
    assert modelo.columnCount() >= 10


def test_edicao_de_celula_ptbr(janela):
    modelo = janela._modelo_atual()
    indice = modelo.index(0, 0)
    assert modelo.setData(indice, "1.234,5")
    assert janela._planilha_atual().df.iat[0, 0] == 1234.5
    assert modelo.data(indice) == "1234,5"


def test_dados_exemplo_e_analise_na_sessao(janela):
    janela._dados_exemplo()
    planilha = janela._planilha_atual()
    assert "altura_cm" in planilha.colunas_numericas()
    assert planilha.tipo_coluna("turma") == "texto"

    from app.core.descritiva import descritiva
    from app.reports.formatacao import render_descritiva

    resultado = descritiva(planilha.valores("altura_cm"), "altura_cm")
    janela.sessao.acrescentar(render_descritiva(resultado))
    assert "Estatística descritiva: altura_cm" in janela.sessao.html_bruto()


def test_grafico_gera_figura(janela):
    janela._dados_exemplo()
    planilha = janela._planilha_atual()
    from app.plots import graficos

    figura = graficos.histograma(planilha.valores("altura_cm"), "altura_cm")
    assert figura.axes  # eixos criados


def test_projeto_salvar_abrir_roundtrip(janela, tmp_path, monkeypatch):
    janela._dados_exemplo()
    caminho = str(tmp_path / "teste.estat")

    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (caminho, "")))
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (caminho, "")))
    janela._salvar_projeto()
    assert os.path.exists(caminho)

    janela._abrir_projeto()
    nomes = [janela.abas.tabText(i) for i in range(janela.abas.count())]
    assert "Dados de exemplo" in nomes
    janela.abas.setCurrentIndex(nomes.index("Dados de exemplo"))
    planilha = janela._planilha_atual()
    assert "altura_cm" in planilha.colunas()
    valores = planilha.valores("altura_cm").dropna()
    assert len(valores) == 60
    assert np.isfinite(valores.astype(float)).all()
