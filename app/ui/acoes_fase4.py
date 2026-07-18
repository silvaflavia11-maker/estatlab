"""Menus e handlers da Fase 4 (mixin da JanelaPrincipal): DOE, séries
temporais, multivariada, confiabilidade e regressões adicionais."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import confiabilidade as mod_conf
from app.core import doe as mod_doe
from app.core import multivariada as mod_mv
from app.core import regressao_extra as mod_rx
from app.core import series as mod_se
from app.core.resultados import ErroAnalise
from app.plots import fase4_plots as f4
from app.reports.formatacao import render_composto, render_teste
from app.ui.dialogos import DialogoAnalise
from app.worksheet.model import Planilha

SEM_COLUNA = "— nenhuma —"


class AcoesFase4:
    # ---------------------------------------------------------------- menus
    def _montar_menus_fase4(self, barra) -> None:
        self._modelo_doe: dict | None = None

        doe = barra.addMenu("D&OE")
        criar = doe.addMenu("Criar plano")
        criar.addAction("Fatorial de 2 níveis…",
                        lambda: self._executar(self._f4_criar_2k))
        criar.addAction("Plackett-Burman…",
                        lambda: self._executar(self._f4_criar_pb))
        criar.addAction("Fatorial geral…",
                        lambda: self._executar(self._f4_criar_geral))
        criar.addAction("Superfície de resposta (CCD/Box-Behnken)…",
                        lambda: self._executar(self._f4_criar_superficie))
        criar.addAction("Taguchi (L4/L8/L9/L12/L16)…",
                        lambda: self._executar(self._f4_criar_taguchi))
        doe.addSeparator()
        doe.addAction("Analisar fatorial de 2 níveis…",
                      lambda: self._executar(self._f4_analisar_fatorial))
        doe.addAction("Analisar superfície de resposta…",
                      lambda: self._executar(self._f4_analisar_superficie))
        doe.addAction("Analisar Taguchi (razão S/N)…",
                      lambda: self._executar(self._f4_analisar_taguchi))
        doe.addSeparator()
        doe.addAction("Otimizar resposta (último modelo)…",
                      lambda: self._executar(self._f4_otimizar))
        doe.addAction("Contorno / superfície 3D (último modelo)…",
                      lambda: self._executar(self._f4_contorno))
        doe.addAction("Gráfico de cubo (último modelo, 3 fatores)",
                      lambda: self._executar(self._f4_cubo))

        series = barra.addMenu("Séries &Temporais")
        series.addAction("Análise de tendência…",
                         lambda: self._executar(self._f4_tendencia))
        series.addAction("Decomposição…",
                         lambda: self._executar(self._f4_decomposicao))
        series.addAction("Média móvel / suavização exponencial / Winters…",
                         lambda: self._executar(self._f4_suavizacao))
        series.addSeparator()
        series.addAction("ACF e PACF…", lambda: self._executar(self._f4_acf))
        series.addAction("Correlação cruzada (CCF)…",
                         lambda: self._executar(self._f4_ccf))
        series.addAction("ARIMA…", lambda: self._executar(self._f4_arima))

        mv = barra.addMenu("&Multivariada")
        mv.addAction("Componentes principais (PCA)…",
                     lambda: self._executar(self._f4_pca))
        mv.addAction("Análise fatorial…",
                     lambda: self._executar(self._f4_fatorial_mv))
        mv.addAction("Análise discriminante…",
                     lambda: self._executar(self._f4_discriminante))
        mv.addAction("Agrupamento (cluster)…",
                     lambda: self._executar(self._f4_cluster))
        mv.addAction("Análise de correspondência…",
                     lambda: self._executar(self._f4_correspondencia))
        mv.addAction("Alfa de Cronbach (análise de itens)…",
                     lambda: self._executar(self._f4_cronbach))
        mv.addSeparator()
        mv.addAction("Árvore de classificação/regressão…",
                     lambda: self._executar(self._f4_arvore))

        conf = barra.addMenu("Con&fiabilidade")
        conf.addAction("Kaplan-Meier (não paramétrica)…",
                       lambda: self._executar(self._f4_km))
        conf.addAction("Análise de distribuição (Weibull/Lognormal/"
                       "Exponencial)…",
                       lambda: self._executar(self._f4_parametrica))

        # itens acrescentados a menus existentes
        reg = self._menu_regressao
        reg.addSeparator()
        reg.addAction("Regressão não linear…",
                      lambda: self._executar(self._f4_nao_linear))
        reg.addAction("Logística ordinal…",
                      lambda: self._executar(self._f4_ordinal))
        reg.addAction("Logística nominal…",
                      lambda: self._executar(self._f4_nominal))
        reg.addAction("Mínimos quadrados parciais (PLS)…",
                      lambda: self._executar(self._f4_pls))
        reg.addAction("Regressão ortogonal (Deming)…",
                      lambda: self._executar(self._f4_ortogonal))

        self._menu_basica.addSeparator()
        self._menu_basica.addAction(
            "Teste de equivalência (TOST)…",
            lambda: self._executar(self._f4_equivalencia))

        self._menu_distribuicoes.addSeparator()
        self._menu_distribuicoes.addAction(
            "Bootstrap (IC de média/mediana/DP)…",
            lambda: self._executar(self._f4_bootstrap))
        self._menu_distribuicoes.addAction(
            "Teste de aleatorização (2 amostras)…",
            lambda: self._executar(self._f4_aleatorizacao))

    # ------------------------------------------------------------ DOE: criar
    def _f4_criar_2k(self) -> None:
        opcoes = (DialogoAnalise("Plano fatorial de 2 níveis", self)
                  .inteiro("fatores", "Número de fatores (2–9):", 3, 2, 9)
                  .inteiro("fracao", "Fração (0 = completo; 1 = ½; 2 = ¼…):",
                           0, 0, 5)
                  .inteiro("replicas", "Réplicas:", 1, 1, 10)
                  .caixa("aleatorizar", "Aleatorizar ordem", True)
                  .inteiro("semente", "Semente (0 = aleatória):", 0, 0, 10**9)
                  .executar())
        if opcoes:
            plano = mod_doe.gerar_fatorial_2k(
                opcoes["fatores"], opcoes["replicas"], opcoes["fracao"],
                opcoes["aleatorizar"], opcoes["semente"] or None)
            self._nova_planilha(Planilha("Plano fatorial", plano))
            self.statusBar().showMessage(
                f"Plano com {len(plano)} corridas criado — preencha a coluna "
                "'resposta' após executar o experimento.", 8000)

    def _f4_criar_pb(self) -> None:
        opcoes = (DialogoAnalise("Plackett-Burman", self)
                  .inteiro("fatores", "Número de fatores (2–23):", 7, 2, 23)
                  .caixa("aleatorizar", "Aleatorizar ordem", True)
                  .inteiro("semente", "Semente (0 = aleatória):", 0, 0, 10**9)
                  .executar())
        if opcoes:
            plano = mod_doe.gerar_plackett_burman(
                opcoes["fatores"], opcoes["aleatorizar"],
                opcoes["semente"] or None)
            self._nova_planilha(Planilha("Plano Plackett-Burman", plano))

    def _f4_criar_geral(self) -> None:
        opcoes = (DialogoAnalise("Fatorial geral", self)
                  .texto("niveis", "Níveis por fator (ex.: 3;2;2):", "3;2")
                  .inteiro("replicas", "Réplicas:", 1, 1, 10)
                  .caixa("aleatorizar", "Aleatorizar ordem", True)
                  .executar())
        if opcoes:
            try:
                niveis = [int(v) for v in opcoes["niveis"].split(";")]
            except ValueError:
                raise ErroAnalise("Formato inválido: use números separados por "
                                  "ponto-e-vírgula (ex.: 3;2;2).")
            plano = mod_doe.gerar_fatorial_geral(niveis, opcoes["replicas"],
                                                 opcoes["aleatorizar"])
            self._nova_planilha(Planilha("Plano fatorial geral", plano))

    def _f4_criar_superficie(self) -> None:
        opcoes = (DialogoAnalise("Plano de superfície de resposta", self)
                  .escolha("tipo", "Tipo:",
                           [("composto central (CCD)", "ccd"),
                            ("Box-Behnken", "box-behnken")])
                  .inteiro("fatores", "Número de fatores:", 3, 2, 6)
                  .caixa("aleatorizar", "Aleatorizar ordem", True)
                  .executar())
        if opcoes:
            plano = mod_doe.gerar_superficie(opcoes["fatores"], opcoes["tipo"],
                                             opcoes["aleatorizar"])
            self._nova_planilha(Planilha("Plano de superfície", plano))

    def _f4_criar_taguchi(self) -> None:
        opcoes = (DialogoAnalise("Arranjo de Taguchi", self)
                  .escolha("arranjo", "Arranjo ortogonal:",
                           [("L4 (3 fatores, 2 níveis)", "L4"),
                            ("L8 (7 fatores, 2 níveis)", "L8"),
                            ("L9 (4 fatores, 3 níveis)", "L9"),
                            ("L12 (11 fatores, 2 níveis)", "L12"),
                            ("L16 (15 fatores, 2 níveis)", "L16")])
                  .executar())
        if opcoes:
            plano = mod_doe.gerar_taguchi(opcoes["arranjo"])
            self._nova_planilha(Planilha(f"Taguchi {opcoes['arranjo']}", plano))
            self.statusBar().showMessage(
                "Preencha as colunas de resposta (réplicas) após o experimento.",
                8000)

    # --------------------------------------------------------- DOE: analisar
    def _f4_analisar_fatorial(self) -> None:
        colunas = self._colunas_numericas(3)
        opcoes = (DialogoAnalise("Análise fatorial de 2 níveis", self)
                  .combo_coluna("resposta", "Resposta:", colunas)
                  .lista_colunas("fatores", "Fatores (2 níveis cada):", colunas)
                  .escolha("ordem", "Interações até a ordem:",
                           [("2 (pares)", 2), ("3", 3)])
                  .numero("alfa", "Nível de significância (α):", 0.05, 0.001,
                          0.5, 3)
                  .caixa("pareto", "Pareto dos efeitos", True)
                  .caixa("normal", "Gráfico normal dos efeitos", False)
                  .caixa("meio_normal", "Gráfico meio-normal dos efeitos", True)
                  .caixa("residuos", "Gráficos de resíduos", True)
                  .executar())
        if not opcoes:
            return
        resultado = mod_doe.analise_fatorial(
            self._planilha_atual().df, opcoes["resposta"], opcoes["fatores"],
            opcoes["ordem"], opcoes["alfa"])
        self._modelo_doe = resultado.dados
        self.sessao.acrescentar(render_composto(resultado))
        efeitos = resultado.dados["efeitos"]
        margem = resultado.dados["margem_lenth"]
        titulo = opcoes["resposta"]
        if opcoes["pareto"]:
            self._abrir_grafico("Pareto dos efeitos",
                                lambda: f4.pareto_efeitos(efeitos, margem,
                                                          titulo))
        if opcoes["normal"]:
            self._abrir_grafico("Normal dos efeitos",
                                lambda: f4.normal_efeitos(efeitos, margem,
                                                          titulo))
        if opcoes["meio_normal"]:
            self._abrir_grafico(
                "Meio-normal dos efeitos",
                lambda: f4.normal_efeitos(efeitos, margem, titulo,
                                          meio_normal=True))
        if opcoes["residuos"] and not resultado.dados["sem_erro"]:
            self._graficos_residuos(resultado, titulo)

    def _f4_analisar_superficie(self) -> None:
        colunas = self._colunas_numericas(3)
        opcoes = (DialogoAnalise("Análise de superfície de resposta", self)
                  .combo_coluna("resposta", "Resposta:", colunas)
                  .lista_colunas("fatores", "Fatores:", colunas)
                  .caixa("residuos", "Gráficos de resíduos", True)
                  .caixa("contorno", "Contorno (2 primeiros fatores)", True)
                  .caixa("superficie", "Superfície 3D rotacionável", True)
                  .executar())
        if not opcoes:
            return
        resultado = mod_doe.analise_superficie(
            self._planilha_atual().df, opcoes["resposta"], opcoes["fatores"])
        self._modelo_doe = resultado.dados
        self.sessao.acrescentar(render_composto(resultado))
        fatores = opcoes["fatores"]
        dados_modelo = resultado.dados
        if opcoes["residuos"]:
            self._graficos_residuos(resultado, opcoes["resposta"])
        if len(fatores) >= 2:
            if opcoes["contorno"]:
                self._abrir_grafico(
                    "Contorno", lambda: f4.contorno_superficie(
                        dados_modelo, fatores[0], fatores[1]))
            if opcoes["superficie"]:
                self._abrir_grafico(
                    "Superfície 3D", lambda: f4.contorno_superficie(
                        dados_modelo, fatores[0], fatores[1], em_3d=True))

    def _f4_analisar_taguchi(self) -> None:
        todas = self._todas_colunas_com_dados()
        opcoes = (DialogoAnalise("Análise de Taguchi", self)
                  .lista_colunas("fatores", "Fatores (colunas do arranjo):",
                                 todas)
                  .lista_colunas("respostas", "Respostas (réplicas):",
                                 self._colunas_numericas())
                  .escolha("criterio", "Critério:",
                           [("maior é melhor", "maior-melhor"),
                            ("menor é melhor", "menor-melhor"),
                            ("nominal é melhor", "nominal-melhor")])
                  .executar())
        if opcoes:
            resultado = mod_doe.analise_taguchi(
                self._planilha_atual().df, opcoes["fatores"],
                opcoes["respostas"], opcoes["criterio"])
            self.sessao.acrescentar(render_composto(resultado))

    def _exigir_modelo_doe(self) -> dict:
        if not self._modelo_doe:
            raise ErroAnalise("Execute antes uma análise fatorial ou de "
                              "superfície (menu DOE) para ajustar um modelo.")
        return self._modelo_doe

    def _f4_otimizar(self) -> None:
        dados_modelo = self._exigir_modelo_doe()
        opcoes = (DialogoAnalise("Otimizar resposta", self)
                  .escolha("objetivo", "Objetivo:",
                           [("maximizar", "maximizar"),
                            ("minimizar", "minimizar"),
                            ("atingir alvo", "alvo")])
                  .numero("alvo", "Alvo (se aplicável):", 0.0)
                  .executar())
        if opcoes:
            resultado = mod_doe.otimizar_resposta(
                dados_modelo, opcoes["objetivo"],
                opcoes["alvo"] if opcoes["objetivo"] == "alvo" else None)
            self.sessao.acrescentar(render_composto(resultado))

    def _f4_contorno(self) -> None:
        dados_modelo = self._exigir_modelo_doe()
        fatores = dados_modelo["fatores"]
        if len(fatores) < 2:
            raise ErroAnalise("O modelo precisa de pelo menos 2 fatores.")
        opcoes = (DialogoAnalise("Contorno / superfície", self)
                  .combo_coluna("x", "Fator do eixo X:", fatores)
                  .combo_coluna("y", "Fator do eixo Y:", fatores)
                  .caixa("em_3d", "Superfície 3D rotacionável (senão contorno)",
                         False)
                  .executar())
        if opcoes:
            if opcoes["x"] == opcoes["y"]:
                raise ErroAnalise("Escolha dois fatores diferentes.")
            self._abrir_grafico(
                "Superfície 3D" if opcoes["em_3d"] else "Contorno",
                lambda: f4.contorno_superficie(dados_modelo, opcoes["x"],
                                               opcoes["y"], opcoes["em_3d"]))

    def _f4_cubo(self) -> None:
        dados_modelo = self._exigir_modelo_doe()
        self._abrir_grafico("Gráfico de cubo", lambda: f4.cubo(dados_modelo))

    # ------------------------------------------------------------- séries
    def _f4_tendencia(self) -> None:
        opcoes = (DialogoAnalise("Análise de tendência", self)
                  .combo_coluna("coluna", "Série (na ordem temporal):",
                                self._colunas_numericas())
                  .escolha("modelo", "Modelo de tendência:",
                           [("linear", "linear"), ("quadrático", "quadratico"),
                            ("exponencial", "exponencial")])
                  .inteiro("horizonte", "Períodos a prever:", 6, 1, 60)
                  .executar())
        if opcoes:
            resultado = mod_se.tendencia(
                self._planilha_atual().valores(opcoes["coluna"]),
                opcoes["coluna"], opcoes["modelo"], opcoes["horizonte"])
            self.sessao.acrescentar(render_composto(resultado))
            dados = resultado.dados
            self._abrir_grafico("Tendência",
                                lambda: f4.serie_previsao(dados,
                                                          resultado.titulo))

    def _f4_decomposicao(self) -> None:
        opcoes = (DialogoAnalise("Decomposição", self)
                  .combo_coluna("coluna", "Série:", self._colunas_numericas())
                  .inteiro("periodo", "Período sazonal (ex.: 12 p/ mensal):",
                           12, 2, 366)
                  .escolha("modelo", "Modelo:",
                           [("aditivo", "aditivo"),
                            ("multiplicativo", "multiplicativo")])
                  .executar())
        if opcoes:
            resultado = mod_se.decomposicao(
                self._planilha_atual().valores(opcoes["coluna"]),
                opcoes["coluna"], opcoes["periodo"], opcoes["modelo"])
            self.sessao.acrescentar(render_composto(resultado))
            dados = resultado.dados
            self._abrir_grafico("Decomposição",
                                lambda: f4.decomposicao_paineis(dados))

    def _f4_suavizacao(self) -> None:
        opcoes = (DialogoAnalise("Suavização / média móvel", self)
                  .combo_coluna("coluna", "Série:", self._colunas_numericas())
                  .escolha("metodo", "Método:",
                           [("média móvel", "media_movel"),
                            ("suavização exponencial simples", "ses"),
                            ("dupla (Holt)", "holt"),
                            ("Winters (tripla)", "winters")])
                  .numero("parametro", "Janela (média móvel) ou α (SES; "
                                       "0 = otimizar):", 3, 0, 100, 2)
                  .inteiro("periodo", "Período sazonal (Winters):", 12, 2, 366)
                  .inteiro("horizonte", "Períodos a prever:", 6, 1, 60)
                  .executar())
        if opcoes:
            resultado = mod_se.suavizacao(
                self._planilha_atual().valores(opcoes["coluna"]),
                opcoes["coluna"], opcoes["metodo"], opcoes["parametro"],
                opcoes["periodo"], opcoes["horizonte"])
            self.sessao.acrescentar(render_composto(resultado))
            dados = resultado.dados
            self._abrir_grafico("Suavização",
                                lambda: f4.serie_previsao(dados,
                                                          resultado.titulo))

    def _f4_acf(self) -> None:
        opcoes = (DialogoAnalise("ACF e PACF", self)
                  .combo_coluna("coluna", "Série:", self._colunas_numericas())
                  .inteiro("defasagens", "Defasagens:", 20, 2, 100)
                  .executar())
        if opcoes:
            resultado = mod_se.autocorrelacao(
                self._planilha_atual().valores(opcoes["coluna"]),
                opcoes["coluna"], defasagens=opcoes["defasagens"])
            self.sessao.acrescentar(render_composto(resultado))
            dados = resultado.dados
            self._abrir_grafico("ACF/PACF", lambda: f4.acf_pacf(dados))

    def _f4_ccf(self) -> None:
        colunas = self._colunas_numericas(2)
        opcoes = (DialogoAnalise("Correlação cruzada", self)
                  .combo_coluna("col1", "Série 1:", colunas)
                  .combo_coluna("col2", "Série 2:", colunas)
                  .inteiro("defasagens", "Defasagens:", 20, 2, 100)
                  .executar())
        if opcoes:
            planilha = self._planilha_atual()
            resultado = mod_se.autocorrelacao(
                planilha.valores(opcoes["col1"]), opcoes["col1"],
                planilha.valores(opcoes["col2"]), opcoes["col2"],
                opcoes["defasagens"])
            self.sessao.acrescentar(render_composto(resultado))
            dados = resultado.dados
            self._abrir_grafico("CCF", lambda: f4.acf_pacf(dados))

    def _f4_arima(self) -> None:
        opcoes = (DialogoAnalise("ARIMA", self)
                  .combo_coluna("coluna", "Série:", self._colunas_numericas())
                  .inteiro("p", "p (ordem AR):", 1, 0, 5)
                  .inteiro("d", "d (diferenciações):", 0, 0, 2)
                  .inteiro("q", "q (ordem MA):", 0, 0, 5)
                  .inteiro("horizonte", "Períodos a prever:", 6, 1, 60)
                  .executar())
        if opcoes:
            resultado = mod_se.arima(
                self._planilha_atual().valores(opcoes["coluna"]),
                opcoes["coluna"], opcoes["p"], opcoes["d"], opcoes["q"],
                opcoes["horizonte"])
            self.sessao.acrescentar(render_composto(resultado))
            dados = resultado.dados
            self._abrir_grafico("ARIMA",
                                lambda: f4.serie_previsao(dados,
                                                          resultado.titulo))

    # ---------------------------------------------------------- multivariada
    def _f4_pca(self) -> None:
        opcoes = (DialogoAnalise("Componentes principais", self)
                  .lista_colunas("colunas", "Variáveis (2+):",
                                 self._colunas_numericas(2))
                  .caixa("grafico", "Scree plot e escores", True)
                  .executar())
        if opcoes:
            resultado = mod_mv.pca(self._planilha_atual().df, opcoes["colunas"])
            self.sessao.acrescentar(render_composto(resultado))
            if opcoes["grafico"]:
                dados = resultado.dados
                self._abrir_grafico("PCA", lambda: f4.scree_escores(dados))

    def _f4_fatorial_mv(self) -> None:
        opcoes = (DialogoAnalise("Análise fatorial", self)
                  .lista_colunas("colunas", "Variáveis (3+):",
                                 self._colunas_numericas(3))
                  .inteiro("fatores", "Número de fatores:", 2, 1, 10)
                  .executar())
        if opcoes:
            resultado = mod_mv.analise_fatorial(self._planilha_atual().df,
                                                opcoes["colunas"],
                                                opcoes["fatores"])
            self.sessao.acrescentar(render_composto(resultado))

    def _f4_discriminante(self) -> None:
        opcoes = (DialogoAnalise("Análise discriminante", self)
                  .combo_coluna("grupo", "Coluna de grupos:",
                                self._todas_colunas_com_dados())
                  .lista_colunas("preditores", "Preditores (numéricos):",
                                 self._colunas_numericas())
                  .executar())
        if opcoes:
            resultado = mod_mv.discriminante(self._planilha_atual().df,
                                             opcoes["grupo"],
                                             opcoes["preditores"])
            self.sessao.acrescentar(render_composto(resultado))

    def _f4_cluster(self) -> None:
        opcoes = (DialogoAnalise("Agrupamento", self)
                  .lista_colunas("colunas", "Variáveis:",
                                 self._colunas_numericas(2))
                  .inteiro("k", "Número de grupos (k):", 3, 2, 20)
                  .escolha("metodo", "Método:",
                           [("k-médias", "kmeans"),
                            ("hierárquico (Ward)", "hierarquico")])
                  .caixa("dendro", "Dendrograma (hierárquico)", True)
                  .executar())
        if opcoes:
            resultado = mod_mv.cluster(self._planilha_atual().df,
                                       opcoes["colunas"], opcoes["k"],
                                       opcoes["metodo"])
            self.sessao.acrescentar(render_composto(resultado))
            rotulos = resultado.dados["rotulos"].to_numpy()
            self._escrever_coluna("grupo_cluster", rotulos)
            if opcoes["dendro"] and opcoes["metodo"] == "hierarquico":
                dados = resultado.dados
                self._abrir_grafico("Dendrograma",
                                    lambda: f4.dendrograma(dados))

    def _f4_correspondencia(self) -> None:
        colunas = self._todas_colunas_com_dados()
        opcoes = (DialogoAnalise("Análise de correspondência", self)
                  .combo_coluna("linhas", "Variável das linhas:", colunas)
                  .combo_coluna("colunas", "Variável das colunas:", colunas)
                  .executar())
        if opcoes:
            if opcoes["linhas"] == opcoes["colunas"]:
                raise ErroAnalise("Escolha duas colunas diferentes.")
            resultado = mod_mv.correspondencia(self._planilha_atual().df,
                                               opcoes["linhas"],
                                               opcoes["colunas"])
            self.sessao.acrescentar(render_composto(resultado))
            dados = resultado.dados
            self._abrir_grafico("Mapa de correspondência",
                                lambda: f4.mapa_correspondencia(dados))

    def _f4_cronbach(self) -> None:
        opcoes = (DialogoAnalise("Alfa de Cronbach", self)
                  .lista_colunas("colunas", "Itens da escala (3+):",
                                 self._colunas_numericas(3))
                  .executar())
        if opcoes:
            resultado = mod_mv.cronbach(self._planilha_atual().df,
                                        opcoes["colunas"])
            self.sessao.acrescentar(render_composto(resultado))

    def _f4_arvore(self) -> None:
        opcoes = (DialogoAnalise("Árvore de classificação/regressão", self)
                  .combo_coluna("resposta", "Resposta:",
                                self._todas_colunas_com_dados())
                  .lista_colunas("preditores", "Preditores (numéricos):",
                                 self._colunas_numericas())
                  .inteiro("profundidade", "Profundidade máxima:", 3, 1, 8)
                  .caixa("grafico", "Desenhar a árvore", True)
                  .executar())
        if opcoes:
            resultado = mod_rx.arvore(self._planilha_atual().df,
                                      opcoes["resposta"], opcoes["preditores"],
                                      opcoes["profundidade"])
            self.sessao.acrescentar(render_composto(resultado))
            if opcoes["grafico"]:
                dados = resultado.dados
                self._abrir_grafico("Árvore", lambda: f4.arvore_plot(dados))

    # -------------------------------------------------------- confiabilidade
    def _f4_km(self) -> None:
        colunas = self._colunas_numericas()
        opcoes = (DialogoAnalise("Kaplan-Meier", self)
                  .combo_coluna("tempo", "Coluna de tempos:", colunas)
                  .combo_coluna("censura",
                                "Coluna de censura (1 = falha, 0 = censurado):",
                                [SEM_COLUNA] + colunas)
                  .caixa("weibull_prob", "Gráfico de probabilidade Weibull",
                         True)
                  .executar())
        if opcoes:
            censura = (None if opcoes["censura"] == SEM_COLUNA
                       else opcoes["censura"])
            resultado = mod_conf.kaplan_meier(self._planilha_atual().df,
                                              opcoes["tempo"], censura)
            self.sessao.acrescentar(render_composto(resultado))
            dados = resultado.dados
            self._abrir_grafico("Sobrevivência",
                                lambda: f4.sobrevivencia(dados))
            if opcoes["weibull_prob"]:
                self._abrir_grafico("Probabilidade Weibull",
                                    lambda: f4.probabilidade_weibull(dados))

    def _f4_parametrica(self) -> None:
        colunas = self._colunas_numericas()
        opcoes = (DialogoAnalise("Análise de distribuição (paramétrica)", self)
                  .combo_coluna("tempo", "Coluna de tempos (ou início do "
                                         "intervalo):", colunas)
                  .escolha("familia", "Família:",
                           [("Weibull", "Weibull"), ("Lognormal", "Lognormal"),
                            ("Exponencial", "Exponencial")])
                  .escolha("tipo", "Censura:",
                           [("à direita", "direita"), ("à esquerda", "esquerda"),
                            ("por intervalo", "intervalo")])
                  .combo_coluna("censura",
                                "Coluna de censura (1 = falha, 0 = censurado):",
                                [SEM_COLUNA] + colunas)
                  .combo_coluna("tempo_fim", "Fim do intervalo (se aplicável):",
                                [SEM_COLUNA] + colunas)
                  .executar())
        if opcoes:
            censura = (None if opcoes["censura"] == SEM_COLUNA
                       else opcoes["censura"])
            fim = (None if opcoes["tempo_fim"] == SEM_COLUNA
                   else opcoes["tempo_fim"])
            resultado = mod_conf.analise_parametrica(
                self._planilha_atual().df, opcoes["tempo"], censura,
                opcoes["familia"], opcoes["tipo"], fim)
            self.sessao.acrescentar(render_composto(resultado))
            dados = resultado.dados
            self._abrir_grafico("Sobrevivência e risco",
                                lambda: f4.sobrevivencia(dados))

    # --------------------------------------------------- regressões extras
    def _f4_nao_linear(self) -> None:
        colunas = self._colunas_numericas(2)
        opcoes = (DialogoAnalise("Regressão não linear", self)
                  .combo_coluna("resposta", "Resposta (Y):", colunas)
                  .combo_coluna("preditor", "Preditor (X):", colunas)
                  .escolha("modelo", "Modelo:",
                           [(nome, nome)
                            for nome in mod_rx.MODELOS_NAO_LINEARES])
                  .executar())
        if opcoes:
            resultado = mod_rx.regressao_nao_linear(
                self._planilha_atual().df, opcoes["resposta"],
                opcoes["preditor"], opcoes["modelo"])
            self.sessao.acrescentar(render_composto(resultado))
            dados = resultado.dados
            self._abrir_grafico("Ajuste não linear",
                                lambda: f4.ajuste_nao_linear(dados))

    def _f4_ordinal(self) -> None:
        opcoes = (DialogoAnalise("Logística ordinal", self)
                  .combo_coluna("resposta", "Resposta (3+ categorias):",
                                self._todas_colunas_com_dados())
                  .lista_colunas("preditores", "Preditores (numéricos):",
                                 self._colunas_numericas())
                  .texto("ordem", "Ordem das categorias (menor→maior, "
                                  "separadas por ;) — vazio = alfabética:")
                  .executar())
        if opcoes:
            ordem = ([v.strip() for v in opcoes["ordem"].split(";") if v.strip()]
                     or None)
            resultado = mod_rx.logistica_ordinal(
                self._planilha_atual().df, opcoes["resposta"],
                opcoes["preditores"], ordem)
            self.sessao.acrescentar(render_composto(resultado))

    def _f4_nominal(self) -> None:
        opcoes = (DialogoAnalise("Logística nominal", self)
                  .combo_coluna("resposta", "Resposta (3+ categorias):",
                                self._todas_colunas_com_dados())
                  .lista_colunas("preditores", "Preditores (numéricos):",
                                 self._colunas_numericas())
                  .executar())
        if opcoes:
            resultado = mod_rx.logistica_nominal(
                self._planilha_atual().df, opcoes["resposta"],
                opcoes["preditores"])
            self.sessao.acrescentar(render_composto(resultado))

    def _f4_pls(self) -> None:
        colunas = self._colunas_numericas(2)
        opcoes = (DialogoAnalise("Mínimos quadrados parciais", self)
                  .combo_coluna("resposta", "Resposta (Y):", colunas)
                  .lista_colunas("preditores", "Preditores (X):", colunas)
                  .inteiro("componentes", "Componentes:", 2, 1, 15)
                  .executar())
        if opcoes:
            resultado = mod_rx.pls(self._planilha_atual().df,
                                   opcoes["resposta"], opcoes["preditores"],
                                   opcoes["componentes"])
            self.sessao.acrescentar(render_composto(resultado))

    def _f4_ortogonal(self) -> None:
        colunas = self._colunas_numericas(2)
        opcoes = (DialogoAnalise("Regressão ortogonal (Deming)", self)
                  .combo_coluna("resposta", "Resposta (Y):", colunas)
                  .combo_coluna("preditor", "Preditor (X):", colunas)
                  .numero("razao", "Razão var(erro Y)/var(erro X):", 1.0, 1e-6)
                  .executar())
        if opcoes:
            resultado = mod_rx.regressao_ortogonal(
                self._planilha_atual().df, opcoes["resposta"],
                opcoes["preditor"], opcoes["razao"])
            self.sessao.acrescentar(render_composto(resultado))

    # ------------------------------------------------ equivalência/bootstrap
    def _f4_equivalencia(self) -> None:
        colunas = self._colunas_numericas()
        opcoes = (DialogoAnalise("Teste de equivalência (TOST)", self)
                  .escolha("tipo", "Tipo:",
                           [("2 amostras", "2"), ("pareado", "par"),
                            ("1 amostra", "1")])
                  .combo_coluna("col1", "Amostra 1:", colunas)
                  .combo_coluna("col2", "Amostra 2 (se aplicável):", colunas)
                  .numero("margem_inf", "Margem inferior:", -0.5)
                  .numero("margem_sup", "Margem superior:", 0.5)
                  .numero("alfa", "Nível de significância (α):", 0.05, 0.001,
                          0.5, 3)
                  .executar())
        if opcoes:
            planilha = self._planilha_atual()
            if opcoes["tipo"] == "1":
                resultado = mod_rx.equivalencia(
                    planilha.valores(opcoes["col1"]), opcoes["col1"],
                    opcoes["margem_inf"], opcoes["margem_sup"],
                    alfa=opcoes["alfa"])
            else:
                resultado = mod_rx.equivalencia(
                    planilha.valores(opcoes["col1"]), opcoes["col1"],
                    opcoes["margem_inf"], opcoes["margem_sup"],
                    dados2=planilha.valores(opcoes["col2"]),
                    col2=opcoes["col2"], pareado=opcoes["tipo"] == "par",
                    alfa=opcoes["alfa"])
            self.sessao.acrescentar(render_teste(resultado))

    def _f4_bootstrap(self) -> None:
        opcoes = (DialogoAnalise("Bootstrap", self)
                  .combo_coluna("coluna", "Coluna:", self._colunas_numericas())
                  .escolha("estatistica", "Estatística:",
                           [("média", "média"), ("mediana", "mediana"),
                            ("desvio-padrão", "desvio-padrão")])
                  .inteiro("reamostras", "Reamostras:", 2000, 100, 20000)
                  .numero("alfa", "Nível de significância (α):", 0.05, 0.001,
                          0.5, 3)
                  .caixa("grafico", "Histograma da distribuição bootstrap",
                         True)
                  .executar())
        if opcoes:
            resultado = mod_rx.bootstrap_ic(
                self._planilha_atual().valores(opcoes["coluna"]),
                opcoes["coluna"], opcoes["estatistica"], opcoes["reamostras"],
                opcoes["alfa"])
            self.sessao.acrescentar(render_composto(resultado))
            if opcoes["grafico"]:
                dados = resultado.dados
                self._abrir_grafico("Bootstrap",
                                    lambda: f4.bootstrap_hist(dados))

    def _f4_aleatorizacao(self) -> None:
        colunas = self._colunas_numericas(2)
        opcoes = (DialogoAnalise("Teste de aleatorização", self)
                  .combo_coluna("col1", "Amostra 1:", colunas)
                  .combo_coluna("col2", "Amostra 2:", colunas)
                  .inteiro("permutacoes", "Permutações:", 5000, 500, 100000)
                  .executar())
        if opcoes:
            planilha = self._planilha_atual()
            resultado = mod_rx.teste_aleatorizacao(
                planilha.valores(opcoes["col1"]),
                planilha.valores(opcoes["col2"]),
                opcoes["col1"], opcoes["col2"], opcoes["permutacoes"])
            self.sessao.acrescentar(render_teste(resultado))

    # -------------------------------------------------------- dados exemplo
    def _dados_exemplo_fase4(self) -> None:
        rng = np.random.default_rng(31)
        # DOE: fatorial 2^3 com 2 réplicas e efeitos conhecidos
        plano = mod_doe.gerar_fatorial_2k(3, replicas=2, aleatorizar=False)
        plano["resposta"] = np.round(
            50 + 4 * plano.F1 - 2.5 * plano.F2 + 1.5 * plano.F1 * plano.F2
            + rng.normal(0, 0.8, len(plano)), 2)
        self._nova_planilha(Planilha("DOE (exemplo)", plano))

        # série mensal com tendência e sazonalidade (4 anos)
        t = np.arange(48)
        vendas = 100 + 1.2 * t + 12 * np.sin(2 * np.pi * t / 12) \
            + rng.normal(0, 3, 48)
        serie = pd.DataFrame({"mes": t + 1, "vendas": np.round(vendas, 1)})
        self._nova_planilha(Planilha("Série temporal (exemplo)", serie))

        # confiabilidade: Weibull com 30% de censura à direita
        tempos = 200 * rng.weibull(1.8, 80)
        corte = np.quantile(tempos, 0.7)
        confiab = pd.DataFrame({
            "tempo_h": np.round(np.minimum(tempos, corte), 1),
            "falha": (tempos <= corte).astype(int),
        })
        self._nova_planilha(Planilha("Confiabilidade (exemplo)", confiab))
        self.statusBar().showMessage(
            "Três planilhas criadas: DOE 2³ com réplicas, série mensal com "
            "sazonalidade e tempos de falha Weibull com censura.", 10000)
