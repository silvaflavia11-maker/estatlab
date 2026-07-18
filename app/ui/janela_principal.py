"""Janela principal do EstatLab."""
from __future__ import annotations

import json
import zipfile
from io import StringIO

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QTabWidget,
)

from app import __version__
from app.core import correlacao as mod_corr
from app.core import descritiva as mod_desc
from app.core import normalidade as mod_norm
from app.core import outliers as mod_out
from app.core import testes as mod_testes
from app.core.resultados import ErroAnalise
from app.plots import graficos
from app.reports.formatacao import render_descritiva, render_tabela, render_teste
from app.ui.acoes_fase2 import AcoesFase2
from app.ui.acoes_fase3 import AcoesFase3
from app.ui.acoes_fase4 import AcoesFase4
from app.ui.dialogos import DialogoAnalise
from app.ui.janela_grafico import JanelaGrafico
from app.ui.sessao import SessaoWidget
from app.ui.tabela import TabelaView
from app.worksheet.model import Planilha, QtPlanilhaModel

EXTENSAO_PROJETO = "Projeto EstatLab (*.estat)"


class JanelaPrincipal(AcoesFase2, AcoesFase3, AcoesFase4, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"EstatLab {__version__} — análise estatística educacional")
        self.resize(1250, 760)

        self.abas = QTabWidget()
        self.abas.setTabsClosable(False)
        self.setCentralWidget(self.abas)
        self._janelas_graficos: list[JanelaGrafico] = []

        self.sessao = SessaoWidget()
        dock = QDockWidget("Sessão (resultados e interpretação)", self)
        dock.setWidget(self.sessao)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.resizeDocks([dock], [500], Qt.Horizontal)

        self._nova_planilha()
        self._montar_menus()

    # ---------------------------------------------------------------- infra
    def _nova_planilha(self, planilha: Planilha | None = None) -> None:
        planilha = planilha or Planilha(f"Planilha {self.abas.count() + 1}")
        modelo = QtPlanilhaModel(planilha)
        visao = TabelaView()
        visao.setModel(modelo)
        self.abas.addTab(visao, planilha.nome)
        self.abas.setCurrentWidget(visao)

    def _modelo_atual(self) -> QtPlanilhaModel:
        return self.abas.currentWidget().model()

    def _planilha_atual(self) -> Planilha:
        return self._modelo_atual().planilha

    def _executar(self, acao) -> None:
        """Roda um handler traduzindo erros em mensagens amigáveis."""
        try:
            acao()
        except ErroAnalise as erro:
            QMessageBox.warning(self, "Não foi possível executar a análise", str(erro))
        except Exception as erro:  # nunca mostrar traceback ao usuário
            QMessageBox.critical(
                self, "Erro inesperado",
                f"Ocorreu um erro inesperado: {erro}\n\n"
                "Verifique os dados selecionados e tente novamente.",
            )

    def _colunas_numericas(self, minimo: int = 1) -> list[str]:
        colunas = self._planilha_atual().colunas_numericas()
        if len(colunas) < minimo:
            raise ErroAnalise(
                "A planilha atual não tem colunas numéricas suficientes para esta "
                f"análise (necessário: {minimo}). Digite ou importe dados primeiro."
            )
        return colunas

    def _todas_colunas_com_dados(self) -> list[str]:
        planilha = self._planilha_atual()
        colunas = [c for c in planilha.colunas() if planilha.tipo_coluna(c) != "vazia"]
        if not colunas:
            raise ErroAnalise("A planilha atual está vazia. Digite ou importe dados primeiro.")
        return colunas

    def _abrir_grafico(self, titulo: str, gerador) -> None:
        janela = JanelaGrafico(titulo, gerador, modelo=self._modelo_atual(), parent=self)
        self._janelas_graficos.append(janela)
        janela.destroyed.connect(
            lambda *_: self._janelas_graficos.remove(janela)
            if janela in self._janelas_graficos else None
        )
        janela.show()

    # ---------------------------------------------------------------- menus
    def _montar_menus(self) -> None:
        barra = self.menuBar()

        # -------- Arquivo
        arquivo = barra.addMenu("&Arquivo")
        arquivo.addAction("Novo projeto", self._novo_projeto)
        arquivo.addAction("Abrir projeto…", lambda: self._executar(self._abrir_projeto))
        arquivo.addAction("Salvar projeto…", lambda: self._executar(self._salvar_projeto))
        arquivo.addSeparator()
        arquivo.addAction("Importar dados (CSV/Excel)…",
                          lambda: self._executar(self._importar))
        arquivo.addAction("Exportar planilha atual…",
                          lambda: self._executar(self._exportar_planilha))
        arquivo.addSeparator()
        arquivo.addAction("Sair", self.close)

        # -------- Planilha
        planilha = barra.addMenu("&Planilha")
        planilha.addAction("Nova planilha", lambda: self._nova_planilha())
        planilha.addAction("Renomear planilha…", self._renomear_planilha)
        planilha.addSeparator()
        planilha.addAction("Adicionar 50 linhas",
                           lambda: self._modelo_atual().adicionar_linhas(50))
        planilha.addAction("Adicionar coluna…", lambda: self._executar(self._adicionar_coluna))
        planilha.addAction("Renomear coluna…", lambda: self._executar(self._renomear_coluna))
        planilha.addAction("Remover coluna…", lambda: self._executar(self._remover_coluna))

        # -------- Estatística Básica
        basica = barra.addMenu("&Estatística Básica")
        self._menu_basica = basica
        basica.addAction("Estatística descritiva…", lambda: self._executar(self._descritiva))
        basica.addSeparator()
        basica.addAction("Teste Z de 1 amostra…", lambda: self._executar(self._z1))
        basica.addAction("Teste t de 1 amostra…", lambda: self._executar(self._t1))
        basica.addAction("Teste t de 2 amostras…", lambda: self._executar(self._t2))
        basica.addAction("Teste t pareado…", lambda: self._executar(self._tpar))
        basica.addSeparator()
        basica.addAction("Teste para 1 proporção…", lambda: self._executar(self._prop1))
        basica.addAction("Teste para 2 proporções…", lambda: self._executar(self._prop2))
        basica.addSeparator()
        basica.addAction("Teste para 1 variância…", lambda: self._executar(self._var1))
        basica.addAction("Teste para 2 variâncias…", lambda: self._executar(self._var2))
        basica.addSeparator()
        basica.addAction("Taxa de Poisson: 1 amostra…", lambda: self._executar(self._pois1))
        basica.addAction("Taxa de Poisson: 2 amostras…", lambda: self._executar(self._pois2))
        basica.addSeparator()
        basica.addAction("Correlação…", lambda: self._executar(self._correlacao))
        basica.addAction("Covariância…", lambda: self._executar(self._covariancia))
        basica.addSeparator()
        basica.addAction("Teste de normalidade…", lambda: self._executar(self._normalidade))
        basica.addAction("Teste de outliers…", lambda: self._executar(self._outliers))

        # -------- Fase 2: Regressão, ANOVA, Não Paramétricos, Tabelas,
        # Distribuições e Poder
        self._montar_menus_fase2(barra)

        # -------- Fase 3: Cartas de Controle, Qualidade e MSA
        self._montar_menus_fase3(barra)

        # -------- Fase 4: DOE, Séries Temporais, Multivariada, Confiabilidade
        self._montar_menus_fase4(barra)

        # -------- Gráficos
        menu_graficos = barra.addMenu("&Gráficos")
        menu_graficos.addAction("Histograma…", lambda: self._executar(self._g_histograma))
        menu_graficos.addAction("Boxplot…", lambda: self._executar(self._g_boxplot))
        menu_graficos.addAction("Dotplot…", lambda: self._executar(self._g_dotplot))
        menu_graficos.addSeparator()
        menu_graficos.addAction("Dispersão…", lambda: self._executar(self._g_dispersao))
        menu_graficos.addAction("Matriz de dispersão…", lambda: self._executar(self._g_matriz))
        menu_graficos.addSeparator()
        menu_graficos.addAction("Barras…", lambda: self._executar(self._g_barras))
        menu_graficos.addAction("Pizza…", lambda: self._executar(self._g_pizza))
        menu_graficos.addSeparator()
        menu_graficos.addAction("Série temporal…", lambda: self._executar(self._g_serie))
        menu_graficos.addAction("Probabilidade normal…", lambda: self._executar(self._g_prob))

        # -------- Ajuda
        ajuda = barra.addMenu("A&juda")
        ajuda.addAction("Carregar dados de exemplo", self._dados_exemplo)
        ajuda.addAction("Carregar dados de exemplo (qualidade/MSA)",
                        self._dados_exemplo_qualidade)
        ajuda.addAction("Carregar dados de exemplo (DOE/séries/confiabilidade)",
                        self._dados_exemplo_fase4)
        ajuda.addAction("Sobre o EstatLab", self._sobre)

    # ---------------------------------------------------------------- arquivo
    def _novo_projeto(self) -> None:
        if QMessageBox.question(
            self, "Novo projeto",
            "Começar um projeto novo? Planilhas e sessão atuais serão descartadas "
            "(salve o projeto antes, se necessário).",
        ) != QMessageBox.Yes:
            return
        self.abas.clear()
        self._nova_planilha()
        self.sessao.carregar_html("")

    def _importar(self) -> None:
        caminho, _ = QFileDialog.getOpenFileName(
            self, "Importar dados", "",
            "Dados (*.csv *.xlsx *.xlsm);;CSV (*.csv);;Excel (*.xlsx *.xlsm)",
        )
        if caminho:
            self._nova_planilha(Planilha.de_arquivo(caminho))

    def _exportar_planilha(self) -> None:
        caminho, _ = QFileDialog.getSaveFileName(
            self, "Exportar planilha", self._planilha_atual().nome + ".csv",
            "CSV (*.csv);;Excel (*.xlsx)",
        )
        if caminho:
            self._planilha_atual().salvar(caminho)

    def _salvar_projeto(self) -> None:
        caminho, _ = QFileDialog.getSaveFileName(self, "Salvar projeto",
                                                 "projeto.estat", EXTENSAO_PROJETO)
        if not caminho:
            return
        nomes = []
        with zipfile.ZipFile(caminho, "w", zipfile.ZIP_DEFLATED) as zf:
            for i in range(self.abas.count()):
                planilha = self.abas.widget(i).model().planilha
                nomes.append(planilha.nome)
                buffer = StringIO()
                planilha._df_compacto().to_csv(buffer, index=False, sep=";", decimal=",")
                zf.writestr(f"planilhas/{i}.csv", buffer.getvalue())
            zf.writestr("sessao.html", self.sessao.html_bruto())
            zf.writestr("manifesto.json", json.dumps(
                {"versao": __version__, "planilhas": nomes}, ensure_ascii=False))
        self.statusBar().showMessage(f"Projeto salvo em {caminho}", 5000)

    def _abrir_projeto(self) -> None:
        caminho, _ = QFileDialog.getOpenFileName(self, "Abrir projeto", "",
                                                 EXTENSAO_PROJETO)
        if not caminho:
            return
        with zipfile.ZipFile(caminho) as zf:
            manifesto = json.loads(zf.read("manifesto.json").decode("utf-8"))
            self.abas.clear()
            for i, nome in enumerate(manifesto["planilhas"]):
                conteudo = zf.read(f"planilhas/{i}.csv").decode("utf-8")
                try:
                    df = pd.read_csv(StringIO(conteudo), sep=";", decimal=",")
                    self._nova_planilha(Planilha(nome, df))
                except pd.errors.EmptyDataError:
                    self._nova_planilha(Planilha(nome))  # planilha salva vazia
            try:
                self.sessao.carregar_html(zf.read("sessao.html").decode("utf-8"))
            except KeyError:
                self.sessao.carregar_html("")
        self.abas.setCurrentIndex(0)

    # ---------------------------------------------------------------- planilha
    def _renomear_planilha(self) -> None:
        atual = self.abas.currentIndex()
        nome, ok = QInputDialog.getText(self, "Renomear planilha", "Novo nome:",
                                        text=self.abas.tabText(atual))
        if ok and nome.strip():
            self._planilha_atual().nome = nome.strip()
            self.abas.setTabText(atual, nome.strip())

    def _adicionar_coluna(self) -> None:
        nome, ok = QInputDialog.getText(self, "Adicionar coluna", "Nome da nova coluna:")
        if ok and nome.strip():
            try:
                self._modelo_atual().adicionar_coluna(nome.strip())
            except ValueError as erro:
                raise ErroAnalise(str(erro))

    def _renomear_coluna(self) -> None:
        colunas = self._planilha_atual().colunas()
        atual, ok = QInputDialog.getItem(self, "Renomear coluna", "Coluna:", colunas,
                                         editable=False)
        if not ok:
            return
        novo, ok = QInputDialog.getText(self, "Renomear coluna", "Novo nome:", text=atual)
        if ok and novo.strip():
            try:
                self._modelo_atual().renomear_coluna(atual, novo.strip())
            except ValueError as erro:
                raise ErroAnalise(str(erro))

    def _remover_coluna(self) -> None:
        colunas = self._planilha_atual().colunas()
        nome, ok = QInputDialog.getItem(self, "Remover coluna", "Coluna:", colunas,
                                        editable=False)
        if ok and QMessageBox.question(
            self, "Remover coluna", f"Remover a coluna '{nome}' e todos os seus dados?"
        ) == QMessageBox.Yes:
            self._modelo_atual().remover_coluna(nome)

    # ---------------------------------------------------------------- análises
    def _descritiva(self) -> None:
        opcoes = (DialogoAnalise("Estatística descritiva", self)
                  .lista_colunas("colunas", "Colunas (selecione uma ou mais):",
                                 self._colunas_numericas())
                  .numero("confianca", "Nível de confiança:", 0.95, 0.5, 0.999, 3)
                  .executar())
        if not opcoes:
            return
        if not opcoes["colunas"]:
            raise ErroAnalise("Selecione pelo menos uma coluna.")
        planilha = self._planilha_atual()
        for coluna in opcoes["colunas"]:
            resultado = mod_desc.descritiva(planilha.valores(coluna), coluna,
                                            opcoes["confianca"])
            self.sessao.acrescentar(render_descritiva(resultado))

    def _z1(self) -> None:
        opcoes = (DialogoAnalise("Teste Z de 1 amostra", self)
                  .combo_coluna("coluna", "Coluna:", self._colunas_numericas())
                  .numero("mu0", "Média hipotética (μ₀):", 0.0)
                  .numero("sigma", "Desvio-padrão populacional (σ):", 1.0, 1e-12)
                  .alternativa_e_alfa().executar())
        if opcoes:
            planilha = self._planilha_atual()
            resultado = mod_testes.teste_z_1amostra(
                planilha.valores(opcoes["coluna"]), opcoes["coluna"],
                opcoes["mu0"], opcoes["sigma"], opcoes["alternativa"], opcoes["alfa"])
            self.sessao.acrescentar(render_teste(resultado))

    def _t1(self) -> None:
        opcoes = (DialogoAnalise("Teste t de 1 amostra", self)
                  .combo_coluna("coluna", "Coluna:", self._colunas_numericas())
                  .numero("mu0", "Média hipotética (μ₀):", 0.0)
                  .alternativa_e_alfa().executar())
        if opcoes:
            planilha = self._planilha_atual()
            resultado = mod_testes.teste_t_1amostra(
                planilha.valores(opcoes["coluna"]), opcoes["coluna"],
                opcoes["mu0"], opcoes["alternativa"], opcoes["alfa"])
            self.sessao.acrescentar(render_teste(resultado))

    def _t2(self) -> None:
        colunas = self._colunas_numericas(2)
        opcoes = (DialogoAnalise("Teste t de 2 amostras", self)
                  .combo_coluna("col1", "Amostra 1:", colunas)
                  .combo_coluna("col2", "Amostra 2:", colunas)
                  .caixa("iguais", "Assumir variâncias iguais (t combinado)")
                  .alternativa_e_alfa().executar())
        if opcoes:
            planilha = self._planilha_atual()
            resultado = mod_testes.teste_t_2amostras(
                planilha.valores(opcoes["col1"]), planilha.valores(opcoes["col2"]),
                opcoes["col1"], opcoes["col2"], opcoes["iguais"],
                opcoes["alternativa"], opcoes["alfa"])
            self.sessao.acrescentar(render_teste(resultado))

    def _tpar(self) -> None:
        colunas = self._colunas_numericas(2)
        opcoes = (DialogoAnalise("Teste t pareado", self)
                  .combo_coluna("col1", "Primeira medição:", colunas)
                  .combo_coluna("col2", "Segunda medição:", colunas)
                  .alternativa_e_alfa().executar())
        if opcoes:
            planilha = self._planilha_atual()
            resultado = mod_testes.teste_t_pareado(
                planilha.valores(opcoes["col1"]), planilha.valores(opcoes["col2"]),
                opcoes["col1"], opcoes["col2"], opcoes["alternativa"], opcoes["alfa"])
            self.sessao.acrescentar(render_teste(resultado))

    def _prop1(self) -> None:
        opcoes = (DialogoAnalise("Teste para 1 proporção", self)
                  .inteiro("sucessos", "Número de sucessos:", 10)
                  .inteiro("n", "Tamanho da amostra (n):", 50, minimo=1)
                  .numero("p0", "Proporção hipotética (p₀):", 0.5, 0.001, 0.999, 3)
                  .alternativa_e_alfa().executar())
        if opcoes:
            resultado = mod_testes.teste_1proporcao(
                opcoes["sucessos"], opcoes["n"], opcoes["p0"],
                opcoes["alternativa"], opcoes["alfa"])
            self.sessao.acrescentar(render_teste(resultado))

    def _prop2(self) -> None:
        opcoes = (DialogoAnalise("Teste para 2 proporções", self)
                  .inteiro("s1", "Amostra 1 — sucessos:", 10)
                  .inteiro("n1", "Amostra 1 — n:", 50, minimo=1)
                  .inteiro("s2", "Amostra 2 — sucessos:", 10)
                  .inteiro("n2", "Amostra 2 — n:", 50, minimo=1)
                  .alternativa_e_alfa().executar())
        if opcoes:
            resultado = mod_testes.teste_2proporcoes(
                opcoes["s1"], opcoes["n1"], opcoes["s2"], opcoes["n2"],
                opcoes["alternativa"], opcoes["alfa"])
            self.sessao.acrescentar(render_teste(resultado))

    def _var1(self) -> None:
        opcoes = (DialogoAnalise("Teste para 1 variância", self)
                  .combo_coluna("coluna", "Coluna:", self._colunas_numericas())
                  .numero("sigma0", "Desvio-padrão hipotético (σ₀):", 1.0, 1e-12)
                  .alternativa_e_alfa().executar())
        if opcoes:
            planilha = self._planilha_atual()
            resultado = mod_testes.teste_1variancia(
                planilha.valores(opcoes["coluna"]), opcoes["coluna"],
                opcoes["sigma0"], opcoes["alternativa"], opcoes["alfa"])
            self.sessao.acrescentar(render_teste(resultado))

    def _var2(self) -> None:
        colunas = self._colunas_numericas(2)
        opcoes = (DialogoAnalise("Teste para 2 variâncias", self)
                  .combo_coluna("col1", "Amostra 1:", colunas)
                  .combo_coluna("col2", "Amostra 2:", colunas)
                  .alternativa_e_alfa().executar())
        if opcoes:
            planilha = self._planilha_atual()
            resultado = mod_testes.teste_2variancias(
                planilha.valores(opcoes["col1"]), planilha.valores(opcoes["col2"]),
                opcoes["col1"], opcoes["col2"], opcoes["alternativa"], opcoes["alfa"])
            self.sessao.acrescentar(render_teste(resultado))

    def _pois1(self) -> None:
        opcoes = (DialogoAnalise("Teste para taxa de Poisson (1 amostra)", self)
                  .inteiro("eventos", "Número de eventos observados:", 10)
                  .numero("exposicao", "Exposição (tempo, área, unidades…):", 1.0, 1e-12)
                  .numero("taxa0", "Taxa hipotética (λ₀):", 1.0, 1e-12)
                  .alternativa_e_alfa().executar())
        if opcoes:
            resultado = mod_testes.teste_taxa_poisson_1amostra(
                opcoes["eventos"], opcoes["exposicao"], opcoes["taxa0"],
                opcoes["alternativa"], opcoes["alfa"])
            self.sessao.acrescentar(render_teste(resultado))

    def _pois2(self) -> None:
        opcoes = (DialogoAnalise("Teste para 2 taxas de Poisson", self)
                  .inteiro("e1", "Amostra 1 — eventos:", 10)
                  .numero("x1", "Amostra 1 — exposição:", 1.0, 1e-12)
                  .inteiro("e2", "Amostra 2 — eventos:", 10)
                  .numero("x2", "Amostra 2 — exposição:", 1.0, 1e-12)
                  .alternativa_e_alfa().executar())
        if opcoes:
            resultado = mod_testes.teste_taxa_poisson_2amostras(
                opcoes["e1"], opcoes["x1"], opcoes["e2"], opcoes["x2"],
                opcoes["alternativa"], opcoes["alfa"])
            self.sessao.acrescentar(render_teste(resultado))

    def _correlacao(self) -> None:
        opcoes = (DialogoAnalise("Correlação", self)
                  .lista_colunas("colunas", "Colunas (selecione 2 ou mais):",
                                 self._colunas_numericas(2))
                  .escolha("metodo", "Método:",
                           [("Pearson (linear)", "pearson"),
                            ("Spearman (postos)", "spearman")])
                  .numero("alfa", "Nível de significância (α):", 0.05, 0.001, 0.5, 3)
                  .executar())
        if opcoes:
            resultado = mod_corr.correlacao(self._planilha_atual().df,
                                            opcoes["colunas"], opcoes["metodo"],
                                            opcoes["alfa"])
            self.sessao.acrescentar(render_tabela(resultado))

    def _covariancia(self) -> None:
        opcoes = (DialogoAnalise("Covariância", self)
                  .lista_colunas("colunas", "Colunas (selecione 2 ou mais):",
                                 self._colunas_numericas(2))
                  .executar())
        if opcoes:
            resultado = mod_corr.covariancia(self._planilha_atual().df, opcoes["colunas"])
            self.sessao.acrescentar(render_tabela(resultado))

    def _normalidade(self) -> None:
        opcoes = (DialogoAnalise("Teste de normalidade", self)
                  .combo_coluna("coluna", "Coluna:", self._colunas_numericas())
                  .escolha("metodo", "Método:",
                           [("Anderson-Darling", mod_norm.MET_AD),
                            ("Shapiro-Wilk", mod_norm.MET_SW),
                            ("Kolmogorov-Smirnov (Lilliefors)", mod_norm.MET_KS)])
                  .numero("alfa", "Nível de significância (α):", 0.05, 0.001, 0.5, 3)
                  .caixa("grafico", "Mostrar gráfico de probabilidade normal", True)
                  .executar())
        if opcoes:
            planilha = self._planilha_atual()
            coluna = opcoes["coluna"]
            resultado = mod_norm.teste_normalidade(planilha.valores(coluna), coluna,
                                                   opcoes["metodo"], opcoes["alfa"])
            self.sessao.acrescentar(render_teste(resultado))
            if opcoes["grafico"]:
                self._abrir_grafico(
                    f"Probabilidade normal — {coluna}",
                    lambda: graficos.probabilidade_normal(planilha.valores(coluna), coluna))

    def _outliers(self) -> None:
        opcoes = (DialogoAnalise("Teste de outliers", self)
                  .combo_coluna("coluna", "Coluna:", self._colunas_numericas())
                  .escolha("metodo", "Método:",
                           [("Grubbs (n ≥ 3, um outlier)", "grubbs"),
                            ("Q de Dixon (3 ≤ n ≤ 10)", "dixon")])
                  .escolha("alfa", "Nível de significância (α):",
                           [("0,05", 0.05), ("0,01", 0.01)])
                  .executar())
        if opcoes:
            planilha = self._planilha_atual()
            funcao = mod_out.teste_grubbs if opcoes["metodo"] == "grubbs" else mod_out.teste_dixon
            resultado = funcao(planilha.valores(opcoes["coluna"]), opcoes["coluna"],
                               opcoes["alfa"])
            self.sessao.acrescentar(render_teste(resultado))

    # ---------------------------------------------------------------- gráficos
    def _g_histograma(self) -> None:
        opcoes = (DialogoAnalise("Histograma", self)
                  .combo_coluna("coluna", "Coluna:", self._colunas_numericas())
                  .inteiro("classes", "Número de classes (0 = automático):", 0, 0, 200)
                  .executar())
        if opcoes:
            planilha, coluna = self._planilha_atual(), opcoes["coluna"]
            classes = opcoes["classes"] or None
            self._abrir_grafico(f"Histograma — {coluna}",
                                lambda: graficos.histograma(planilha.valores(coluna),
                                                            coluna, classes))

    def _g_boxplot(self) -> None:
        opcoes = (DialogoAnalise("Boxplot", self)
                  .lista_colunas("colunas", "Colunas (uma ou mais):",
                                 self._colunas_numericas())
                  .executar())
        if opcoes:
            if not opcoes["colunas"]:
                raise ErroAnalise("Selecione pelo menos uma coluna.")
            planilha, colunas = self._planilha_atual(), opcoes["colunas"]
            self._abrir_grafico(
                "Boxplot — " + ", ".join(colunas),
                lambda: graficos.boxplot([planilha.valores(c) for c in colunas], colunas))

    def _g_dotplot(self) -> None:
        opcoes = (DialogoAnalise("Dotplot", self)
                  .combo_coluna("coluna", "Coluna:", self._colunas_numericas())
                  .executar())
        if opcoes:
            planilha, coluna = self._planilha_atual(), opcoes["coluna"]
            self._abrir_grafico(f"Dotplot — {coluna}",
                                lambda: graficos.dotplot(planilha.valores(coluna), coluna))

    def _g_dispersao(self) -> None:
        colunas = self._colunas_numericas(2)
        opcoes = (DialogoAnalise("Gráfico de dispersão", self)
                  .combo_coluna("x", "Eixo X:", colunas)
                  .combo_coluna("y", "Eixo Y:", colunas)
                  .executar())
        if opcoes:
            planilha = self._planilha_atual()
            cx, cy = opcoes["x"], opcoes["y"]
            self._abrir_grafico(f"Dispersão — {cy} × {cx}",
                                lambda: graficos.dispersao(planilha.valores(cx),
                                                           planilha.valores(cy), cx, cy))

    def _g_matriz(self) -> None:
        opcoes = (DialogoAnalise("Matriz de dispersão", self)
                  .lista_colunas("colunas", "Colunas (2 ou mais):",
                                 self._colunas_numericas(2))
                  .executar())
        if opcoes:
            planilha, colunas = self._planilha_atual(), opcoes["colunas"]
            self._abrir_grafico("Matriz de dispersão",
                                lambda: graficos.matriz_dispersao(planilha.df, colunas))

    def _g_barras(self) -> None:
        opcoes = (DialogoAnalise("Gráfico de barras", self)
                  .combo_coluna("coluna", "Coluna (categorias):",
                                self._todas_colunas_com_dados())
                  .executar())
        if opcoes:
            planilha, coluna = self._planilha_atual(), opcoes["coluna"]
            self._abrir_grafico(f"Barras — {coluna}",
                                lambda: graficos.barras(planilha.valores(coluna), coluna))

    def _g_pizza(self) -> None:
        opcoes = (DialogoAnalise("Gráfico de pizza", self)
                  .combo_coluna("coluna", "Coluna (categorias):",
                                self._todas_colunas_com_dados())
                  .executar())
        if opcoes:
            planilha, coluna = self._planilha_atual(), opcoes["coluna"]
            self._abrir_grafico(f"Pizza — {coluna}",
                                lambda: graficos.pizza(planilha.valores(coluna), coluna))

    def _g_serie(self) -> None:
        opcoes = (DialogoAnalise("Série temporal", self)
                  .combo_coluna("coluna", "Coluna (na ordem das linhas):",
                                self._colunas_numericas())
                  .executar())
        if opcoes:
            planilha, coluna = self._planilha_atual(), opcoes["coluna"]
            self._abrir_grafico(f"Série temporal — {coluna}",
                                lambda: graficos.serie_temporal(planilha.valores(coluna),
                                                                coluna))

    def _g_prob(self) -> None:
        opcoes = (DialogoAnalise("Gráfico de probabilidade normal", self)
                  .combo_coluna("coluna", "Coluna:", self._colunas_numericas())
                  .executar())
        if opcoes:
            planilha, coluna = self._planilha_atual(), opcoes["coluna"]
            self._abrir_grafico(
                f"Probabilidade normal — {coluna}",
                lambda: graficos.probabilidade_normal(planilha.valores(coluna), coluna))

    # ---------------------------------------------------------------- ajuda
    def _dados_exemplo(self) -> None:
        rng = np.random.default_rng(7)
        n = 60
        altura = np.round(rng.normal(170, 8, n), 1)
        peso = np.round(22 * (altura / 100) ** 2 + rng.normal(0, 6, n), 1)
        nota_a = np.round(np.clip(rng.normal(6.5, 1.4, n), 0, 10), 1)
        nota_b = np.round(np.clip(nota_a + rng.normal(0.4, 0.8, n), 0, 10), 1)
        turma = rng.choice(["Manhã", "Tarde", "Noite"], n, p=[0.45, 0.35, 0.2])
        faltas = rng.poisson(np.where(turma == "Noite", 4.0, 2.0))
        aprovado = np.where(nota_b >= 6, "sim", "não")
        df = pd.DataFrame({
            "altura_cm": altura, "peso_kg": peso,
            "nota_prova1": nota_a, "nota_prova2": nota_b, "turma": turma,
            "faltas": faltas, "aprovado": aprovado,
        })
        self._nova_planilha(Planilha("Dados de exemplo", df))
        self.statusBar().showMessage(
            "Dados de exemplo carregados: 60 alunos (altura, peso, notas, turma, "
            "faltas e aprovação).", 8000)

    def _sobre(self) -> None:
        QMessageBox.about(
            self, "Sobre o EstatLab",
            f"<b>EstatLab {__version__}</b> (nome provisório)<br>"
            "Aplicativo estatístico educacional em português.<br><br>"
            "Fase 1: estatística básica, testes de hipóteses e gráficos.<br>"
            "Cálculos: scipy, statsmodels, numpy e pandas.<br>"
            "Uso interno — projeto educacional.",
        )
