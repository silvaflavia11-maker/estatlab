"""Menus e handlers da Fase 2 (mixin da JanelaPrincipal).

Usa a infraestrutura da janela principal: ``_executar``, ``_colunas_numericas``,
``_todas_colunas_com_dados``, ``_planilha_atual``, ``_abrir_grafico`` e
``self.sessao``.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QInputDialog

from app.core import anova as mod_anova
from app.core import distribuicoes as mod_dist
from app.core import naoparametricos as mod_np
from app.core import poder as mod_poder
from app.core import regressao as mod_reg
from app.core import tabelas as mod_tab
from app.core.resultados import ErroAnalise
from app.plots import graficos
from app.reports.formatacao import render_composto, render_tabela, render_teste
from app.ui.dialogos import DialogoAnalise


class AcoesFase2:
    # ---------------------------------------------------------------- menus
    def _montar_menus_fase2(self, barra) -> None:
        reg = barra.addMenu("&Regressão")
        self._menu_regressao = reg
        reg.addAction("Regressão linear…", lambda: self._executar(self._f2_reg_linear))
        reg.addAction("Stepwise…", lambda: self._executar(self._f2_stepwise))
        reg.addAction("Melhores subconjuntos…",
                      lambda: self._executar(self._f2_subconjuntos))
        reg.addSeparator()
        reg.addAction("Regressão logística binária…",
                      lambda: self._executar(self._f2_logistica))
        reg.addAction("Regressão de Poisson…",
                      lambda: self._executar(self._f2_reg_poisson))

        av = barra.addMenu("A&NOVA")
        av.addAction("ANOVA de 1 fator…", lambda: self._executar(self._f2_anova1))
        av.addAction("ANOVA de 2 fatores…", lambda: self._executar(self._f2_anova2))
        av.addSeparator()
        av.addAction("Igualdade de variâncias (Levene/Bartlett)…",
                     lambda: self._executar(self._f2_var_grupos))

        np_menu = barra.addMenu("&Não Paramétricos")
        np_menu.addAction("Teste do sinal…", lambda: self._executar(self._f2_sinal))
        np_menu.addAction("Wilcoxon (1 amostra)…",
                          lambda: self._executar(self._f2_wilcoxon1))
        np_menu.addAction("Wilcoxon (pareado)…",
                          lambda: self._executar(self._f2_wilcoxon_par))
        np_menu.addAction("Mann-Whitney…", lambda: self._executar(self._f2_mann))
        np_menu.addSeparator()
        np_menu.addAction("Kruskal-Wallis…", lambda: self._executar(self._f2_kruskal))
        np_menu.addAction("Mediana de Mood…", lambda: self._executar(self._f2_mood))
        np_menu.addAction("Friedman…", lambda: self._executar(self._f2_friedman))
        np_menu.addSeparator()
        np_menu.addAction("Teste de sequências (runs)…",
                          lambda: self._executar(self._f2_runs))

        tab = barra.addMenu("&Tabelas")
        tab.addAction("Tabulação cruzada e qui-quadrado…",
                      lambda: self._executar(self._f2_cruzada))
        tab.addAction("Teste exato de Fisher (2×2)…",
                      lambda: self._executar(self._f2_fisher))
        tab.addAction("Qui-quadrado de aderência…",
                      lambda: self._executar(self._f2_aderencia))

        dist = barra.addMenu("&Distribuições")
        self._menu_distribuicoes = dist
        dist.addAction("Calculadora de probabilidade…",
                       lambda: self._executar(self._f2_calculadora))
        dist.addAction("Gráfico de distribuição…",
                       lambda: self._executar(self._f2_grafico_dist))
        dist.addSeparator()
        dist.addAction("Gerar números aleatórios…",
                       lambda: self._executar(self._f2_gerar))
        dist.addAction("Amostragem aleatória de coluna…",
                       lambda: self._executar(self._f2_amostrar))

        pod = barra.addMenu("P&oder e Amostra")
        pod.addAction("Teste t (1 amostra, pareado, 2 amostras)…",
                      lambda: self._executar(self._f2_poder_t))
        pod.addAction("Teste Z de 1 amostra…",
                      lambda: self._executar(self._f2_poder_z))
        pod.addAction("Proporções (1 e 2)…",
                      lambda: self._executar(self._f2_poder_prop))
        pod.addAction("Variâncias (1 e 2)…",
                      lambda: self._executar(self._f2_poder_var))
        pod.addAction("ANOVA de 1 fator…",
                      lambda: self._executar(self._f2_poder_anova))

    # ------------------------------------------------------------ auxiliares
    def _escrever_coluna(self, nome: str, valores) -> None:
        modelo = self._modelo_atual()
        planilha = self._planilha_atual()
        base, contador = nome, 2
        while nome in planilha.colunas():
            nome = f"{base}_{contador}"
            contador += 1
        modelo.adicionar_coluna(nome)
        if len(valores) > len(planilha.df):
            modelo.adicionar_linhas(len(valores) - len(planilha.df) + 10)
        planilha.df.loc[: len(valores) - 1, nome] = np.asarray(valores, dtype=object)
        modelo.notificar_tudo()
        self.statusBar().showMessage(
            f"Coluna '{nome}' criada com {len(valores)} valores.", 6000)

    def _graficos_residuos(self, resultado, titulo: str) -> None:
        residuos = resultado.dados["residuos"]
        ajustados = resultado.dados["ajustados"]
        self._abrir_grafico(f"Resíduos — {titulo}",
                            lambda: graficos.residuos_4paineis(residuos, ajustados,
                                                               titulo))

    def _parametros_distribuicao(self, titulo: str) -> tuple | None:
        nomes = list(mod_dist.DISTRIBUICOES)
        nome, ok = QInputDialog.getItem(self, titulo, "Distribuição:", nomes,
                                        editable=False)
        if not ok:
            return None
        rotulos, padroes, _, _ = mod_dist.DISTRIBUICOES[nome]
        dialogo = DialogoAnalise(f"{titulo} — {nome}", self)
        for i, (rotulo, padrao) in enumerate(zip(rotulos, padroes)):
            dialogo.numero(f"p{i}", rotulo + ":", padrao)
        return nome, rotulos, dialogo

    @staticmethod
    def _n_ou_poder(opcoes: dict) -> tuple:
        """Converte a convenção dos diálogos (0 = calcular) para None."""
        n = opcoes["n"] or None
        poder = opcoes["poder"] or None
        return n, poder

    def _mostrar_poder(self, resultado, com_curva: bool, n_informado) -> None:
        self.sessao.acrescentar(render_composto(resultado))
        if com_curva and "curva" in resultado.dados:
            curva = resultado.dados["curva"]
            n_ref = n_informado or resultado.dados.get("n_calc") or 30
            n_max = max(int(n_ref * 2.5), 30)
            self._abrir_grafico(
                "Curva de poder",
                lambda: graficos.curva_poder(curva, 2, n_max, resultado.titulo,
                                             n_atual=n_ref))

    # ---------------------------------------------------------------- ANOVA
    def _f2_anova1(self) -> None:
        opcoes = (DialogoAnalise("ANOVA de 1 fator", self)
                  .combo_coluna("resposta", "Resposta (numérica):",
                                self._colunas_numericas())
                  .combo_coluna("fator", "Fator (grupos):",
                                self._todas_colunas_com_dados())
                  .escolha("comparacao", "Comparações múltiplas:",
                           [("nenhuma", None), ("Tukey", "tukey"),
                            ("Fisher (LSD)", "fisher"),
                            ("Dunnett (vs controle)", "dunnett")])
                  .numero("alfa", "Nível de significância (α):", 0.05, 0.001, 0.5, 3)
                  .caixa("residuos", "Gráficos de resíduos", True)
                  .caixa("efeitos", "Gráfico de efeitos principais", True)
                  .executar())
        if not opcoes:
            return
        controle = None
        if opcoes["comparacao"] == "dunnett":
            planilha = self._planilha_atual()
            niveis = sorted(planilha.valores(opcoes["fator"]).dropna()
                            .astype(str).unique())
            controle, ok = QInputDialog.getItem(self, "Dunnett",
                                                "Nível de controle:", niveis,
                                                editable=False)
            if not ok:
                return
        resultado = mod_anova.anova_1via(self._planilha_atual().df,
                                         opcoes["resposta"], opcoes["fator"],
                                         opcoes["alfa"], opcoes["comparacao"],
                                         controle)
        self.sessao.acrescentar(render_composto(resultado))
        titulo = f"{opcoes['resposta']} × {opcoes['fator']}"
        if opcoes["residuos"]:
            self._graficos_residuos(resultado, titulo)
        if opcoes["efeitos"]:
            grupos = resultado.dados["grupos"]
            self._abrir_grafico(
                f"Efeitos principais — {titulo}",
                lambda: graficos.efeitos_principais(grupos, opcoes["fator"],
                                                    opcoes["resposta"]))

    def _f2_anova2(self) -> None:
        fatores = self._todas_colunas_com_dados()
        opcoes = (DialogoAnalise("ANOVA de 2 fatores", self)
                  .combo_coluna("resposta", "Resposta (numérica):",
                                self._colunas_numericas())
                  .combo_coluna("fator_a", "Fator A:", fatores)
                  .combo_coluna("fator_b", "Fator B:", fatores)
                  .caixa("interacao", "Incluir interação A × B", True)
                  .numero("alfa", "Nível de significância (α):", 0.05, 0.001, 0.5, 3)
                  .caixa("residuos", "Gráficos de resíduos", True)
                  .caixa("grafico_int", "Gráfico de interação", True)
                  .executar())
        if not opcoes:
            return
        if opcoes["fator_a"] == opcoes["fator_b"]:
            raise ErroAnalise("Escolha dois fatores diferentes.")
        resultado = mod_anova.anova_2vias(self._planilha_atual().df,
                                          opcoes["resposta"], opcoes["fator_a"],
                                          opcoes["fator_b"], opcoes["interacao"],
                                          opcoes["alfa"])
        self.sessao.acrescentar(render_composto(resultado))
        if opcoes["residuos"]:
            self._graficos_residuos(resultado,
                                    f"{opcoes['resposta']} × 2 fatores")
        if opcoes["grafico_int"]:
            medias = resultado.dados["medias_interacao"]
            self._abrir_grafico(
                f"Interação — {opcoes['fator_a']} × {opcoes['fator_b']}",
                lambda: graficos.interacao(medias, opcoes["fator_a"],
                                           opcoes["fator_b"], opcoes["resposta"]))

    def _f2_var_grupos(self) -> None:
        opcoes = (DialogoAnalise("Igualdade de variâncias", self)
                  .combo_coluna("resposta", "Resposta (numérica):",
                                self._colunas_numericas())
                  .combo_coluna("fator", "Fator (grupos):",
                                self._todas_colunas_com_dados())
                  .escolha("metodo", "Método:",
                           [("Levene (robusto)", "levene"), ("Bartlett", "bartlett")])
                  .numero("alfa", "Nível de significância (α):", 0.05, 0.001, 0.5, 3)
                  .executar())
        if opcoes:
            resultado = mod_anova.variancias_grupos(
                self._planilha_atual().df, opcoes["resposta"], opcoes["fator"],
                opcoes["metodo"], opcoes["alfa"])
            self.sessao.acrescentar(render_teste(resultado))

    # ------------------------------------------------------------- Regressão
    def _f2_reg_linear(self) -> None:
        colunas = self._colunas_numericas(2)
        opcoes = (DialogoAnalise("Regressão linear", self)
                  .combo_coluna("resposta", "Resposta (Y):", colunas)
                  .lista_colunas("preditores", "Preditores (X):", colunas)
                  .numero("alfa", "Nível de significância (α):", 0.05, 0.001, 0.5, 3)
                  .caixa("residuos", "Gráficos de resíduos", True)
                  .caixa("prever", "Fazer predição para valores específicos", False)
                  .executar())
        if not opcoes:
            return
        predicao = None
        if opcoes["prever"]:
            dialogo = DialogoAnalise("Valores para predição", self)
            for preditor in opcoes["preditores"]:
                dialogo.numero(preditor, f"{preditor} =", 0.0)
            predicao = dialogo.executar()
            if predicao is None:
                return
        resultado = mod_reg.regressao_linear(self._planilha_atual().df,
                                             opcoes["resposta"],
                                             opcoes["preditores"], opcoes["alfa"],
                                             predicao)
        self.sessao.acrescentar(render_composto(resultado))
        if opcoes["residuos"]:
            self._graficos_residuos(resultado, f"regressão de {opcoes['resposta']}")

    def _f2_stepwise(self) -> None:
        colunas = self._colunas_numericas(2)
        opcoes = (DialogoAnalise("Regressão stepwise", self)
                  .combo_coluna("resposta", "Resposta (Y):", colunas)
                  .lista_colunas("preditores", "Preditores candidatos:", colunas)
                  .escolha("criterio", "Critério de seleção:",
                           [("p-valor", "p"), ("AICc", "aicc"), ("BIC", "bic")])
                  .numero("alfa_entrada", "α de entrada (critério p-valor):",
                          0.15, 0.01, 0.5, 3)
                  .executar())
        if opcoes:
            resultado = mod_reg.stepwise(self._planilha_atual().df,
                                         opcoes["resposta"], opcoes["preditores"],
                                         opcoes["criterio"], opcoes["alfa_entrada"])
            self.sessao.acrescentar(render_composto(resultado))

    def _f2_subconjuntos(self) -> None:
        colunas = self._colunas_numericas(2)
        opcoes = (DialogoAnalise("Melhores subconjuntos", self)
                  .combo_coluna("resposta", "Resposta (Y):", colunas)
                  .lista_colunas("preditores", "Preditores candidatos:", colunas)
                  .executar())
        if opcoes:
            resultado = mod_reg.melhores_subconjuntos(
                self._planilha_atual().df, opcoes["resposta"], opcoes["preditores"])
            self.sessao.acrescentar(render_composto(resultado))

    def _f2_logistica(self) -> None:
        opcoes = (DialogoAnalise("Regressão logística binária", self)
                  .combo_coluna("resposta", "Resposta (2 categorias):",
                                self._todas_colunas_com_dados())
                  .lista_colunas("preditores", "Preditores (numéricos):",
                                 self._colunas_numericas())
                  .numero("alfa", "Nível de significância (α):", 0.05, 0.001, 0.5, 3)
                  .executar())
        if opcoes:
            resultado = mod_reg.regressao_logistica(
                self._planilha_atual().df, opcoes["resposta"],
                opcoes["preditores"], opcoes["alfa"])
            self.sessao.acrescentar(render_composto(resultado))

    def _f2_reg_poisson(self) -> None:
        colunas = self._colunas_numericas(2)
        opcoes = (DialogoAnalise("Regressão de Poisson", self)
                  .combo_coluna("resposta", "Resposta (contagens):", colunas)
                  .lista_colunas("preditores", "Preditores:", colunas)
                  .numero("alfa", "Nível de significância (α):", 0.05, 0.001, 0.5, 3)
                  .executar())
        if opcoes:
            resultado = mod_reg.regressao_poisson(
                self._planilha_atual().df, opcoes["resposta"],
                opcoes["preditores"], opcoes["alfa"])
            self.sessao.acrescentar(render_composto(resultado))

    # ------------------------------------------------------- Não paramétricos
    def _f2_sinal(self) -> None:
        opcoes = (DialogoAnalise("Teste do sinal", self)
                  .combo_coluna("coluna", "Coluna:", self._colunas_numericas())
                  .numero("mediana0", "Mediana hipotética:", 0.0)
                  .alternativa_e_alfa().executar())
        if opcoes:
            resultado = mod_np.teste_sinal(
                self._planilha_atual().valores(opcoes["coluna"]), opcoes["coluna"],
                opcoes["mediana0"], opcoes["alternativa"], opcoes["alfa"])
            self.sessao.acrescentar(render_teste(resultado))

    def _f2_wilcoxon1(self) -> None:
        opcoes = (DialogoAnalise("Wilcoxon (1 amostra)", self)
                  .combo_coluna("coluna", "Coluna:", self._colunas_numericas())
                  .numero("mediana0", "Mediana hipotética:", 0.0)
                  .alternativa_e_alfa().executar())
        if opcoes:
            resultado = mod_np.teste_wilcoxon(
                self._planilha_atual().valores(opcoes["coluna"]), opcoes["coluna"],
                opcoes["mediana0"], alternativa=opcoes["alternativa"],
                alfa=opcoes["alfa"])
            self.sessao.acrescentar(render_teste(resultado))

    def _f2_wilcoxon_par(self) -> None:
        colunas = self._colunas_numericas(2)
        opcoes = (DialogoAnalise("Wilcoxon (pareado)", self)
                  .combo_coluna("col1", "Primeira medição:", colunas)
                  .combo_coluna("col2", "Segunda medição:", colunas)
                  .alternativa_e_alfa().executar())
        if opcoes:
            planilha = self._planilha_atual()
            resultado = mod_np.teste_wilcoxon(
                planilha.valores(opcoes["col1"]), opcoes["col1"],
                dados2=planilha.valores(opcoes["col2"]), coluna2=opcoes["col2"],
                alternativa=opcoes["alternativa"], alfa=opcoes["alfa"])
            self.sessao.acrescentar(render_teste(resultado))

    def _f2_mann(self) -> None:
        colunas = self._colunas_numericas(2)
        opcoes = (DialogoAnalise("Mann-Whitney", self)
                  .combo_coluna("col1", "Amostra 1:", colunas)
                  .combo_coluna("col2", "Amostra 2:", colunas)
                  .alternativa_e_alfa().executar())
        if opcoes:
            planilha = self._planilha_atual()
            resultado = mod_np.teste_mann_whitney(
                planilha.valores(opcoes["col1"]), planilha.valores(opcoes["col2"]),
                opcoes["col1"], opcoes["col2"], opcoes["alternativa"], opcoes["alfa"])
            self.sessao.acrescentar(render_teste(resultado))

    def _f2_kruskal(self) -> None:
        opcoes = (DialogoAnalise("Kruskal-Wallis", self)
                  .combo_coluna("resposta", "Resposta (numérica):",
                                self._colunas_numericas())
                  .combo_coluna("fator", "Fator (grupos):",
                                self._todas_colunas_com_dados())
                  .numero("alfa", "Nível de significância (α):", 0.05, 0.001, 0.5, 3)
                  .executar())
        if opcoes:
            resultado = mod_np.teste_kruskal(self._planilha_atual().df,
                                             opcoes["resposta"], opcoes["fator"],
                                             opcoes["alfa"])
            self.sessao.acrescentar(render_teste(resultado))

    def _f2_mood(self) -> None:
        opcoes = (DialogoAnalise("Mediana de Mood", self)
                  .combo_coluna("resposta", "Resposta (numérica):",
                                self._colunas_numericas())
                  .combo_coluna("fator", "Fator (grupos):",
                                self._todas_colunas_com_dados())
                  .numero("alfa", "Nível de significância (α):", 0.05, 0.001, 0.5, 3)
                  .executar())
        if opcoes:
            resultado = mod_np.teste_mood(self._planilha_atual().df,
                                          opcoes["resposta"], opcoes["fator"],
                                          opcoes["alfa"])
            self.sessao.acrescentar(render_teste(resultado))

    def _f2_friedman(self) -> None:
        opcoes = (DialogoAnalise("Teste de Friedman", self)
                  .lista_colunas("colunas",
                                 "Tratamentos (3+ colunas; linhas = blocos):",
                                 self._colunas_numericas(3))
                  .numero("alfa", "Nível de significância (α):", 0.05, 0.001, 0.5, 3)
                  .executar())
        if opcoes:
            resultado = mod_np.teste_friedman(self._planilha_atual().df,
                                              opcoes["colunas"], opcoes["alfa"])
            self.sessao.acrescentar(render_teste(resultado))

    def _f2_runs(self) -> None:
        opcoes = (DialogoAnalise("Teste de sequências (runs)", self)
                  .combo_coluna("coluna", "Coluna (na ordem de coleta):",
                                self._colunas_numericas())
                  .numero("alfa", "Nível de significância (α):", 0.05, 0.001, 0.5, 3)
                  .executar())
        if opcoes:
            resultado = mod_np.teste_runs(
                self._planilha_atual().valores(opcoes["coluna"]), opcoes["coluna"],
                opcoes["alfa"])
            self.sessao.acrescentar(render_teste(resultado))

    # ---------------------------------------------------------------- Tabelas
    def _f2_cruzada(self) -> None:
        colunas = self._todas_colunas_com_dados()
        opcoes = (DialogoAnalise("Tabulação cruzada e qui-quadrado", self)
                  .combo_coluna("linhas", "Variável das linhas:", colunas)
                  .combo_coluna("colunas", "Variável das colunas:", colunas)
                  .numero("alfa", "Nível de significância (α):", 0.05, 0.001, 0.5, 3)
                  .executar())
        if opcoes:
            if opcoes["linhas"] == opcoes["colunas"]:
                raise ErroAnalise("Escolha duas colunas diferentes.")
            resultado = mod_tab.tabulacao_cruzada(
                self._planilha_atual().df, opcoes["linhas"], opcoes["colunas"],
                opcoes["alfa"])
            self.sessao.acrescentar(render_composto(resultado))

    def _f2_fisher(self) -> None:
        colunas = self._todas_colunas_com_dados()
        opcoes = (DialogoAnalise("Teste exato de Fisher", self)
                  .combo_coluna("linhas", "Variável das linhas (2 categorias):",
                                colunas)
                  .combo_coluna("colunas", "Variável das colunas (2 categorias):",
                                colunas)
                  .numero("alfa", "Nível de significância (α):", 0.05, 0.001, 0.5, 3)
                  .executar())
        if opcoes:
            if opcoes["linhas"] == opcoes["colunas"]:
                raise ErroAnalise("Escolha duas colunas diferentes.")
            resultado = mod_tab.fisher_exato(
                self._planilha_atual().df, opcoes["linhas"], opcoes["colunas"],
                opcoes["alfa"])
            self.sessao.acrescentar(render_composto(resultado))

    def _f2_aderencia(self) -> None:
        opcoes = (DialogoAnalise("Qui-quadrado de aderência", self)
                  .combo_coluna("coluna", "Coluna (categorias):",
                                self._todas_colunas_com_dados())
                  .caixa("iguais", "Proporções esperadas iguais", True)
                  .numero("alfa", "Nível de significância (α):", 0.05, 0.001, 0.5, 3)
                  .executar())
        if not opcoes:
            return
        proporcoes = None
        if not opcoes["iguais"]:
            texto, ok = QInputDialog.getText(
                self, "Proporções esperadas",
                "Informe categoria=proporção separadas por ponto-e-vírgula\n"
                "(ex.: azul=0,5; vermelho=0,3; verde=0,2):")
            if not ok:
                return
            proporcoes = {}
            try:
                for parte in texto.split(";"):
                    chave, valor = parte.split("=")
                    proporcoes[chave.strip()] = float(
                        valor.strip().replace(",", "."))
            except ValueError:
                raise ErroAnalise("Formato inválido. Use categoria=proporção "
                                  "separadas por ponto-e-vírgula.")
        resultado = mod_tab.aderencia(self._planilha_atual().df, opcoes["coluna"],
                                      proporcoes, opcoes["alfa"])
        self.sessao.acrescentar(render_composto(resultado))

    # ------------------------------------------------------------ Distribuições
    def _f2_calculadora(self) -> None:
        escolha = self._parametros_distribuicao("Calculadora de probabilidade")
        if escolha is None:
            return
        nome, rotulos, dialogo = escolha
        dialogo.escolha("tipo", "Cálculo:",
                        [("acumulada — P(X ≤ x)", "acumulada"),
                         ("densidade/probabilidade em x", "densidade"),
                         ("inversa — quantil de uma probabilidade", "inversa")])
        dialogo.numero("valor", "Valor (x ou probabilidade):", 0.0)
        dialogo.caixa("grafico", "Mostrar gráfico da distribuição", True)
        opcoes = dialogo.executar()
        if not opcoes:
            return
        parametros = [opcoes[f"p{i}"] for i in range(len(rotulos))]
        resultado = mod_dist.calcular(nome, parametros, opcoes["tipo"],
                                      opcoes["valor"])
        self.sessao.acrescentar(render_tabela(resultado))
        if opcoes["grafico"]:
            sombra = opcoes["valor"] if opcoes["tipo"] == "acumulada" else None
            curva = mod_dist.dados_curva(nome, parametros, sombra_ate=sombra)
            self._abrir_grafico(f"Distribuição {nome}",
                                lambda: graficos.distribuicao(curva,
                                                              resultado.titulo))

    def _f2_grafico_dist(self) -> None:
        escolha = self._parametros_distribuicao("Gráfico de distribuição")
        if escolha is None:
            return
        nome, rotulos, dialogo = escolha
        opcoes = dialogo.executar()
        if not opcoes:
            return
        parametros = [opcoes[f"p{i}"] for i in range(len(rotulos))]
        curva = mod_dist.dados_curva(nome, parametros)
        self._abrir_grafico(f"Distribuição {nome}",
                            lambda: graficos.distribuicao(curva,
                                                          f"Distribuição {nome}"))

    def _f2_gerar(self) -> None:
        escolha = self._parametros_distribuicao("Gerar números aleatórios")
        if escolha is None:
            return
        nome, rotulos, dialogo = escolha
        dialogo.inteiro("quantidade", "Quantidade de valores:", 100, 1, 100_000)
        dialogo.inteiro("semente", "Semente (0 = aleatória):", 0, 0, 10**9)
        opcoes = dialogo.executar()
        if not opcoes:
            return
        parametros = [opcoes[f"p{i}"] for i in range(len(rotulos))]
        valores = mod_dist.gerar_aleatorios(nome, parametros, opcoes["quantidade"],
                                            opcoes["semente"] or None)
        self._escrever_coluna(f"aleatorio_{nome.lower().split()[0]}", valores)

    def _f2_amostrar(self) -> None:
        opcoes = (DialogoAnalise("Amostragem aleatória", self)
                  .combo_coluna("coluna", "Coluna de origem:",
                                self._todas_colunas_com_dados())
                  .inteiro("quantidade", "Tamanho da amostra:", 10, 1, 100_000)
                  .caixa("reposicao", "Com reposição", False)
                  .inteiro("semente", "Semente (0 = aleatória):", 0, 0, 10**9)
                  .executar())
        if opcoes:
            valores = mod_dist.amostrar_coluna(
                self._planilha_atual().valores(opcoes["coluna"]), opcoes["coluna"],
                opcoes["quantidade"], opcoes["reposicao"],
                opcoes["semente"] or None)
            self._escrever_coluna(f"amostra_{opcoes['coluna']}", valores)

    # ------------------------------------------------------------------ Poder
    def _f2_poder_t(self) -> None:
        opcoes = (DialogoAnalise("Poder — teste t", self)
                  .escolha("tipo", "Tipo de teste:",
                           [("2 amostras", "2amostras"), ("1 amostra", "1amostra"),
                            ("pareado", "pareado")])
                  .numero("diferenca", "Diferença de interesse:", 1.0)
                  .numero("desvio", "Desvio-padrão suposto:", 1.0, 1e-12)
                  .inteiro("n", "n por amostra (0 = calcular):", 0, 0)
                  .numero("poder", "Poder desejado (0 = calcular):", 0.8, 0.0, 0.999, 3)
                  .escolha("alternativa", "Hipótese alternativa:",
                           [("bilateral (≠)", "bilateral"),
                            ("unilateral (<)", "menor"),
                            ("unilateral (>)", "maior")])
                  .numero("alfa", "Nível de significância (α):", 0.05, 0.001, 0.5, 3)
                  .caixa("curva", "Mostrar curva de poder", True)
                  .executar())
        if opcoes:
            n, poder = self._n_ou_poder(opcoes)
            resultado = mod_poder.poder_t(opcoes["tipo"], opcoes["diferenca"],
                                          opcoes["desvio"], opcoes["alfa"],
                                          n, poder, opcoes["alternativa"])
            self._mostrar_poder(resultado, opcoes["curva"], n)

    def _f2_poder_z(self) -> None:
        opcoes = (DialogoAnalise("Poder — teste Z de 1 amostra", self)
                  .numero("diferenca", "Diferença de interesse:", 1.0)
                  .numero("sigma", "σ (conhecido):", 1.0, 1e-12)
                  .inteiro("n", "n (0 = calcular):", 0, 0)
                  .numero("poder", "Poder desejado (0 = calcular):", 0.8, 0.0, 0.999, 3)
                  .escolha("alternativa", "Hipótese alternativa:",
                           [("bilateral (≠)", "bilateral"),
                            ("unilateral (<)", "menor"),
                            ("unilateral (>)", "maior")])
                  .numero("alfa", "Nível de significância (α):", 0.05, 0.001, 0.5, 3)
                  .caixa("curva", "Mostrar curva de poder", True)
                  .executar())
        if opcoes:
            n, poder = self._n_ou_poder(opcoes)
            resultado = mod_poder.poder_z_1amostra(opcoes["diferenca"],
                                                   opcoes["sigma"], opcoes["alfa"],
                                                   n, poder, opcoes["alternativa"])
            self._mostrar_poder(resultado, opcoes["curva"], n)

    def _f2_poder_prop(self) -> None:
        opcoes = (DialogoAnalise("Poder — proporções", self)
                  .escolha("tipo", "Tipo de teste:",
                           [("2 proporções", "2proporcoes"),
                            ("1 proporção", "1proporcao")])
                  .numero("p0", "p₀ (hipotética / grupo 2):", 0.5, 0.001, 0.999, 3)
                  .numero("p1", "p₁ (real suposta / grupo 1):", 0.6, 0.001, 0.999, 3)
                  .inteiro("n", "n por grupo (0 = calcular):", 0, 0)
                  .numero("poder", "Poder desejado (0 = calcular):", 0.8, 0.0, 0.999, 3)
                  .escolha("alternativa", "Hipótese alternativa:",
                           [("bilateral (≠)", "bilateral"),
                            ("unilateral (<)", "menor"),
                            ("unilateral (>)", "maior")])
                  .numero("alfa", "Nível de significância (α):", 0.05, 0.001, 0.5, 3)
                  .caixa("curva", "Mostrar curva de poder", True)
                  .executar())
        if opcoes:
            n, poder = self._n_ou_poder(opcoes)
            resultado = mod_poder.poder_proporcoes(
                opcoes["tipo"], opcoes["p0"], opcoes["p1"], opcoes["alfa"],
                n, poder, opcoes["alternativa"])
            self._mostrar_poder(resultado, opcoes["curva"], n)

    def _f2_poder_var(self) -> None:
        opcoes = (DialogoAnalise("Poder — variâncias (bilateral)", self)
                  .escolha("tipo", "Tipo de teste:",
                           [("1 variância", "1variancia"),
                            ("2 variâncias", "2variancias")])
                  .numero("razao", "Razão de desvios-padrão (σ/σ₀ ou σ₁/σ₂):",
                          1.5, 1e-6)
                  .inteiro("n", "n por amostra (0 = calcular):", 0, 0)
                  .numero("poder", "Poder desejado (0 = calcular):", 0.8, 0.0, 0.999, 3)
                  .numero("alfa", "Nível de significância (α):", 0.05, 0.001, 0.5, 3)
                  .caixa("curva", "Mostrar curva de poder", True)
                  .executar())
        if opcoes:
            n, poder = self._n_ou_poder(opcoes)
            resultado = mod_poder.poder_variancias(opcoes["tipo"], opcoes["razao"],
                                                   opcoes["alfa"], n, poder)
            self._mostrar_poder(resultado, opcoes["curva"], n)

    def _f2_poder_anova(self) -> None:
        opcoes = (DialogoAnalise("Poder — ANOVA de 1 fator", self)
                  .inteiro("k", "Número de grupos (k):", 3, 2, 50)
                  .numero("efeito", "Tamanho de efeito f (σ_entre/σ_dentro):",
                          0.25, 1e-6, 10, 3)
                  .inteiro("n", "n por grupo (0 = calcular):", 0, 0)
                  .numero("poder", "Poder desejado (0 = calcular):", 0.8, 0.0, 0.999, 3)
                  .numero("alfa", "Nível de significância (α):", 0.05, 0.001, 0.5, 3)
                  .caixa("curva", "Mostrar curva de poder", True)
                  .executar())
        if opcoes:
            n, poder = self._n_ou_poder(opcoes)
            resultado = mod_poder.poder_anova(opcoes["k"], opcoes["efeito"],
                                              opcoes["alfa"], n, poder)
            self._mostrar_poder(resultado, opcoes["curva"], n)
