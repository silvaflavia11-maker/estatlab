"""Menus e handlers da Fase 3 (mixin da JanelaPrincipal): cartas de controle,
qualidade/capabilidade e MSA."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.core import capabilidade as mod_cap
from app.core import cep as mod_cep
from app.core import msa as mod_msa
from app.core import qualidade as mod_qual
from app.core.resultados import ErroAnalise
from app.plots import qualidade_plots as qp
from app.reports.formatacao import render_composto
from app.ui.dialogos import DialogoAnalise
from app.worksheet.model import Planilha

OPCOES_TESTES = [("testes 1–4 (padrão)", frozenset({1, 2, 3, 4})),
                 ("somente teste 1 (além de 3σ)", frozenset({1})),
                 ("todos os testes (1–8)", frozenset(range(1, 9)))]
SEM_ESTAGIO = "— nenhum —"


class AcoesFase3:
    # ---------------------------------------------------------------- menus
    def _montar_menus_fase3(self, barra) -> None:
        cep = barra.addMenu("&Cartas de Controle")
        cep.addAction("I-MR (individuais)…", lambda: self._executar(self._f3_imr))
        cep.addAction("Xbarra-R…", lambda: self._executar(self._f3_xbar_r))
        cep.addAction("Xbarra-S…", lambda: self._executar(self._f3_xbar_s))
        cep.addSeparator()
        cep.addAction("P (proporção defeituosa)…",
                      lambda: self._executar(self._f3_p))
        cep.addAction("NP (nº de defeituosos)…",
                      lambda: self._executar(self._f3_np))
        cep.addAction("C (nº de defeitos)…", lambda: self._executar(self._f3_c))
        cep.addAction("U (defeitos por unidade)…",
                      lambda: self._executar(self._f3_u))
        cep.addSeparator()
        cep.addAction("MA (média móvel)…", lambda: self._executar(self._f3_ma))
        cep.addAction("EWMA…", lambda: self._executar(self._f3_ewma))
        cep.addAction("CUSUM…", lambda: self._executar(self._f3_cusum))

        qual = barra.addMenu("&Qualidade")
        qual.addAction("Run chart…", lambda: self._executar(self._f3_run))
        qual.addAction("Diagrama de Pareto…",
                       lambda: self._executar(self._f3_pareto))
        qual.addAction("Diagrama de causa e efeito (Ishikawa)…",
                       lambda: self._executar(self._f3_ishikawa))
        qual.addAction("Carta Multi-Vari…",
                       lambda: self._executar(self._f3_multivari))
        qual.addSeparator()
        qual.addAction("Capabilidade de processo (normal/transformada)…",
                       lambda: self._executar(self._f3_capabilidade))
        qual.addAction("Relatório de Capabilidade Completo…",
                       lambda: self._executar(self._f3_relatorio_cap))
        qual.addAction("Identificação de distribuição individual…",
                       lambda: self._executar(self._f3_identificar))
        qual.addAction("Capabilidade por atributos (binomial)…",
                       lambda: self._executar(self._f3_cap_atributos))
        qual.addSeparator()
        qual.addAction("Intervalo de tolerância…",
                       lambda: self._executar(self._f3_tolerancia))

        msa = barra.addMenu("&MSA")
        msa.addAction("Gage R&R cruzado (ANOVA)…",
                      lambda: self._executar(self._f3_grr_cruzado))
        msa.addAction("Gage R&R aninhado (ANOVA)…",
                      lambda: self._executar(self._f3_grr_aninhado))
        msa.addAction("Gage run chart…", lambda: self._executar(self._f3_gage_run))
        msa.addSeparator()
        msa.addAction("Linearidade e viés…",
                      lambda: self._executar(self._f3_linearidade))
        msa.addAction("Estudo Tipo 1…", lambda: self._executar(self._f3_tipo1))
        msa.addAction("Concordância por atributos…",
                      lambda: self._executar(self._f3_concordancia))
        msa.addSeparator()
        msa.addAction("Gerar planilha de coleta (Gage R&R)…",
                      lambda: self._executar(self._f3_planilha_coleta))

    # ------------------------------------------------------------ auxiliares
    def _dialogo_carta(self, titulo: str, com_estagios: bool = True,
                       zonas: bool = True) -> DialogoAnalise:
        dialogo = (DialogoAnalise(titulo, self)
                   .combo_coluna("coluna", "Coluna de dados:",
                                 self._colunas_numericas()))
        if com_estagios:
            dialogo.combo_coluna("estagios", "Coluna de estágios (opcional):",
                                 [SEM_ESTAGIO] + self._todas_colunas_com_dados())
        if zonas:
            dialogo.escolha("testes", "Testes de causas especiais:",
                            OPCOES_TESTES)
        return dialogo

    def _mostrar_cartas(self, cartas, resumo, gerador) -> None:
        self.sessao.acrescentar(render_composto(resumo))
        self._abrir_grafico(resumo.titulo, gerador)

    def _estagios_de(self, opcoes) -> object:
        nome = opcoes.get("estagios")
        if not nome or nome == SEM_ESTAGIO:
            return None
        return self._planilha_atual().valores(nome)

    # -------------------------------------------------------------- cartas
    def _f3_imr(self) -> None:
        opcoes = self._dialogo_carta("Carta I-MR").executar()
        if opcoes:
            planilha, coluna = self._planilha_atual(), opcoes["coluna"]
            estagios = self._estagios_de(opcoes)
            testes = set(opcoes["testes"])

            def gerar():
                cartas, _ = mod_cep.carta_i_mr(planilha.valores(coluna), coluna,
                                               estagios, testes)
                return qp.carta_controle(cartas, f"Carta I-MR — {coluna}")

            cartas, resumo = mod_cep.carta_i_mr(planilha.valores(coluna), coluna,
                                                estagios, testes)
            self._mostrar_cartas(cartas, resumo, gerar)

    def _carta_xbar_generica(self, titulo: str, funcao) -> None:
        dialogo = self._dialogo_carta(titulo)
        dialogo.inteiro("tamanho", "Tamanho do subgrupo (linhas consecutivas):",
                        5, 2, 25)
        opcoes = dialogo.executar()
        if opcoes:
            planilha, coluna = self._planilha_atual(), opcoes["coluna"]
            estagios = self._estagios_de(opcoes)
            testes = set(opcoes["testes"])
            tamanho = opcoes["tamanho"]

            def gerar():
                cartas, _ = funcao(planilha.valores(coluna), coluna, tamanho,
                                   estagios, testes)
                return qp.carta_controle(cartas, f"{titulo} — {coluna}")

            cartas, resumo = funcao(planilha.valores(coluna), coluna, tamanho,
                                    estagios, testes)
            self._mostrar_cartas(cartas, resumo, gerar)

    def _f3_xbar_r(self) -> None:
        self._carta_xbar_generica("Carta Xbarra-R", mod_cep.carta_xbar_r)

    def _f3_xbar_s(self) -> None:
        self._carta_xbar_generica("Carta Xbarra-S", mod_cep.carta_xbar_s)

    def _carta_atributo(self, titulo: str, np_chart: bool) -> None:
        colunas = self._colunas_numericas()
        dialogo = (DialogoAnalise(titulo, self)
                   .combo_coluna("coluna", "Coluna de defeituosos:", colunas)
                   .escolha("origem_n", "Tamanho da amostra:",
                            [("constante (informar abaixo)", "constante"),
                             ("em uma coluna", "coluna")])
                   .inteiro("n_constante", "Tamanho constante:", 100, 1)
                   .combo_coluna("n_coluna", "Coluna de tamanhos (se aplicável):",
                                 colunas)
                   .escolha("testes", "Testes de causas especiais:",
                            OPCOES_TESTES))
        opcoes = dialogo.executar()
        if opcoes:
            planilha = self._planilha_atual()
            tamanhos = (opcoes["n_constante"] if opcoes["origem_n"] == "constante"
                        else planilha.valores(opcoes["n_coluna"]))
            coluna = opcoes["coluna"]
            testes = set(opcoes["testes"])

            def gerar():
                cartas, _ = mod_cep.carta_p(planilha.valores(coluna), coluna,
                                            tamanhos, testes, np_chart=np_chart)
                return qp.carta_controle(cartas, f"{titulo} — {coluna}")

            cartas, resumo = mod_cep.carta_p(planilha.valores(coluna), coluna,
                                             tamanhos, testes, np_chart=np_chart)
            self._mostrar_cartas(cartas, resumo, gerar)

    def _f3_p(self) -> None:
        self._carta_atributo("Carta P", np_chart=False)

    def _f3_np(self) -> None:
        self._carta_atributo("Carta NP", np_chart=True)

    def _f3_c(self) -> None:
        opcoes = (DialogoAnalise("Carta C", self)
                  .combo_coluna("coluna", "Coluna de contagens de defeitos:",
                                self._colunas_numericas())
                  .escolha("testes", "Testes de causas especiais:", OPCOES_TESTES)
                  .executar())
        if opcoes:
            planilha, coluna = self._planilha_atual(), opcoes["coluna"]
            testes = set(opcoes["testes"])

            def gerar():
                cartas, _ = mod_cep.carta_c(planilha.valores(coluna), coluna,
                                            testes=testes)
                return qp.carta_controle(cartas, f"Carta C — {coluna}")

            cartas, resumo = mod_cep.carta_c(planilha.valores(coluna), coluna,
                                             testes=testes)
            self._mostrar_cartas(cartas, resumo, gerar)

    def _f3_u(self) -> None:
        colunas = self._colunas_numericas(2)
        opcoes = (DialogoAnalise("Carta U", self)
                  .combo_coluna("coluna", "Coluna de contagens de defeitos:",
                                colunas)
                  .combo_coluna("tamanhos", "Coluna de nº de unidades:", colunas)
                  .escolha("testes", "Testes de causas especiais:", OPCOES_TESTES)
                  .executar())
        if opcoes:
            planilha, coluna = self._planilha_atual(), opcoes["coluna"]
            tamanhos = planilha.valores(opcoes["tamanhos"])
            testes = set(opcoes["testes"])

            def gerar():
                cartas, _ = mod_cep.carta_c(planilha.valores(coluna), coluna,
                                            tamanhos=tamanhos, testes=testes)
                return qp.carta_controle(cartas, f"Carta U — {coluna}")

            cartas, resumo = mod_cep.carta_c(planilha.valores(coluna), coluna,
                                             tamanhos=tamanhos, testes=testes)
            self._mostrar_cartas(cartas, resumo, gerar)

    def _f3_ma(self) -> None:
        opcoes = (self._dialogo_carta("Carta MA", com_estagios=False, zonas=False)
                  .inteiro("janela", "Janela da média móvel:", 3, 2, 20)
                  .executar())
        if opcoes:
            planilha, coluna = self._planilha_atual(), opcoes["coluna"]
            janela = opcoes["janela"]

            def gerar():
                cartas, _ = mod_cep.carta_ma(planilha.valores(coluna), coluna,
                                             janela)
                return qp.carta_controle(cartas, f"Carta MA — {coluna}")

            cartas, resumo = mod_cep.carta_ma(planilha.valores(coluna), coluna,
                                              janela)
            self._mostrar_cartas(cartas, resumo, gerar)

    def _f3_ewma(self) -> None:
        opcoes = (self._dialogo_carta("Carta EWMA", com_estagios=False,
                                      zonas=False)
                  .numero("lam", "λ (peso da observação atual):", 0.2, 0.01, 1, 2)
                  .executar())
        if opcoes:
            planilha, coluna = self._planilha_atual(), opcoes["coluna"]
            lam = opcoes["lam"]

            def gerar():
                cartas, _ = mod_cep.carta_ewma(planilha.valores(coluna), coluna,
                                               lam)
                return qp.carta_controle(cartas, f"Carta EWMA — {coluna}")

            cartas, resumo = mod_cep.carta_ewma(planilha.valores(coluna), coluna,
                                                lam)
            self._mostrar_cartas(cartas, resumo, gerar)

    def _f3_cusum(self) -> None:
        opcoes = (self._dialogo_carta("Carta CUSUM", com_estagios=False,
                                      zonas=False)
                  .caixa("usar_alvo", "Informar alvo (senão usa a média)", False)
                  .numero("alvo", "Alvo do processo:", 0.0)
                  .numero("k", "k (folga, em σ):", 0.5, 0.01, 5, 2)
                  .numero("h", "h (limite de decisão, em σ):", 4.0, 0.5, 10, 2)
                  .executar())
        if opcoes:
            planilha, coluna = self._planilha_atual(), opcoes["coluna"]
            alvo = opcoes["alvo"] if opcoes["usar_alvo"] else None
            k, h = opcoes["k"], opcoes["h"]

            def gerar():
                cartas, _ = mod_cep.carta_cusum(planilha.valores(coluna), coluna,
                                                alvo, k, h)
                return qp.carta_controle(cartas, f"Carta CUSUM — {coluna}")

            cartas, resumo = mod_cep.carta_cusum(planilha.valores(coluna),
                                                 coluna, alvo, k, h)
            self._mostrar_cartas(cartas, resumo, gerar)

    # ------------------------------------------------------------- qualidade
    def _f3_run(self) -> None:
        opcoes = (DialogoAnalise("Run chart", self)
                  .combo_coluna("coluna", "Coluna (na ordem de coleta):",
                                self._colunas_numericas())
                  .executar())
        if opcoes:
            from app.core.naoparametricos import teste_runs
            from app.reports.formatacao import render_teste

            planilha, coluna = self._planilha_atual(), opcoes["coluna"]
            self.sessao.acrescentar(render_teste(
                teste_runs(planilha.valores(coluna), coluna)))
            self._abrir_grafico(f"Run chart — {coluna}",
                                lambda: qp.run_chart(planilha.valores(coluna),
                                                     coluna))

    def _f3_pareto(self) -> None:
        opcoes = (DialogoAnalise("Diagrama de Pareto", self)
                  .combo_coluna("coluna", "Coluna de categorias/defeitos:",
                                self._todas_colunas_com_dados())
                  .executar())
        if opcoes:
            planilha, coluna = self._planilha_atual(), opcoes["coluna"]
            self.sessao.acrescentar(render_composto(
                mod_qual.pareto_resumo(planilha.valores(coluna), coluna)))
            self._abrir_grafico(
                f"Pareto — {coluna}",
                lambda: qp.pareto(mod_qual.pareto_contagens(
                    planilha.valores(coluna), coluna), coluna))

    def _f3_ishikawa(self) -> None:
        dialogo = (DialogoAnalise("Diagrama de causa e efeito", self)
                   .texto("problema", "Problema (efeito):",
                          "Defeito no produto"))
        for categoria in ("Máquina", "Método", "Material", "Mão de obra",
                          "Medição", "Meio ambiente"):
            dialogo.texto(categoria, f"{categoria} (causas separadas por ;):")
        opcoes = dialogo.executar()
        if opcoes:
            problema = opcoes.pop("problema").strip() or "Problema"
            causas = {categoria: [c.strip() for c in texto.split(";") if c.strip()]
                      for categoria, texto in opcoes.items()}
            self._abrir_grafico(f"Ishikawa — {problema}",
                                lambda: qp.ishikawa(problema, causas))

    def _f3_multivari(self) -> None:
        fatores_disponiveis = self._todas_colunas_com_dados()
        opcoes = (DialogoAnalise("Carta Multi-Vari", self)
                  .combo_coluna("resposta", "Resposta (numérica):",
                                self._colunas_numericas())
                  .combo_coluna("f1", "Fator 1 (eixo X):", fatores_disponiveis)
                  .combo_coluna("f2", "Fator 2 (séries):", fatores_disponiveis)
                  .combo_coluna("f3", "Fator 3 (painéis, opcional):",
                                [SEM_ESTAGIO] + fatores_disponiveis)
                  .executar())
        if opcoes:
            fatores = [opcoes["f1"], opcoes["f2"]]
            if opcoes["f3"] != SEM_ESTAGIO:
                fatores.append(opcoes["f3"])
            if len(set(fatores)) != len(fatores):
                raise ErroAnalise("Escolha fatores diferentes entre si.")
            planilha, resposta = self._planilha_atual(), opcoes["resposta"]
            dados = mod_qual.multivari_dados(planilha.df, resposta, fatores)
            self._abrir_grafico("Carta Multi-Vari",
                                lambda: qp.multivari(dados, resposta, fatores))

    def _opcoes_capabilidade(self, titulo: str):
        return (DialogoAnalise(titulo, self)
                .combo_coluna("coluna", "Coluna de dados:",
                              self._colunas_numericas())
                .caixa("tem_lie", "Há limite inferior (LIE)", True)
                .numero("lie", "LIE:", 0.0)
                .caixa("tem_lse", "Há limite superior (LSE)", True)
                .numero("lse", "LSE:", 1.0)
                .caixa("tem_alvo", "Há alvo", False)
                .numero("alvo", "Alvo:", 0.5)
                .inteiro("subgrupo", "Tamanho do subgrupo (1 = individuais):",
                         1, 1, 10))

    def _f3_capabilidade(self) -> None:
        opcoes = (self._opcoes_capabilidade("Capabilidade de processo")
                  .escolha("transformacao", "Transformação (dados não normais):",
                           [("nenhuma", None), ("Box-Cox", "boxcox"),
                            ("Johnson (SU)", "johnson")])
                  .caixa("histograma", "Histograma com especificações", True)
                  .executar())
        if opcoes:
            planilha, coluna = self._planilha_atual(), opcoes["coluna"]
            resultado = mod_cap.capabilidade_normal(
                planilha.valores(coluna), coluna,
                opcoes["lie"] if opcoes["tem_lie"] else None,
                opcoes["lse"] if opcoes["tem_lse"] else None,
                opcoes["alvo"] if opcoes["tem_alvo"] else None,
                opcoes["subgrupo"], opcoes["transformacao"])
            self.sessao.acrescentar(render_composto(resultado))
            if opcoes["histograma"]:
                dados_cap = resultado.dados
                self._abrir_grafico(f"Capabilidade — {coluna}",
                                    lambda: qp.capabilidade_histograma(dados_cap))

    def _f3_relatorio_cap(self) -> None:
        opcoes = self._opcoes_capabilidade(
            "Relatório de Capabilidade Completo").executar()
        if opcoes:
            planilha, coluna = self._planilha_atual(), opcoes["coluna"]
            resultado = mod_cap.capabilidade_normal(
                planilha.valores(coluna), coluna,
                opcoes["lie"] if opcoes["tem_lie"] else None,
                opcoes["lse"] if opcoes["tem_lse"] else None,
                opcoes["alvo"] if opcoes["tem_alvo"] else None,
                opcoes["subgrupo"])
            self.sessao.acrescentar(render_composto(resultado))
            dados_cap = resultado.dados
            self._abrir_grafico(f"Relatório de Capabilidade — {coluna}",
                                lambda: qp.relatorio_capabilidade(dados_cap))

    def _f3_identificar(self) -> None:
        opcoes = (DialogoAnalise("Identificação de distribuição", self)
                  .combo_coluna("coluna", "Coluna de dados:",
                                self._colunas_numericas())
                  .executar())
        if opcoes:
            resultado = mod_cap.identificar_distribuicao(
                self._planilha_atual().valores(opcoes["coluna"]),
                opcoes["coluna"])
            self.sessao.acrescentar(render_composto(resultado))

    def _f3_cap_atributos(self) -> None:
        opcoes = (DialogoAnalise("Capabilidade por atributos", self)
                  .inteiro("defeituosos", "Total de defeituosos:", 10)
                  .inteiro("total", "Total inspecionado:", 1000, 1)
                  .numero("confianca", "Nível de confiança:", 0.95, 0.5, 0.999, 3)
                  .executar())
        if opcoes:
            resultado = mod_cap.capabilidade_atributos(
                opcoes["defeituosos"], opcoes["total"], opcoes["confianca"])
            self.sessao.acrescentar(render_composto(resultado))

    def _f3_tolerancia(self) -> None:
        opcoes = (DialogoAnalise("Intervalo de tolerância", self)
                  .combo_coluna("coluna", "Coluna de dados:",
                                self._colunas_numericas())
                  .numero("cobertura", "Cobertura da população:", 0.95, 0.5,
                          0.999, 3)
                  .numero("confianca", "Nível de confiança:", 0.95, 0.5, 0.999, 3)
                  .escolha("lado", "Tipo de intervalo:",
                           [("bilateral", "bilateral"),
                            ("unilateral superior", "superior"),
                            ("unilateral inferior", "inferior")])
                  .executar())
        if opcoes:
            resultado = mod_qual.intervalo_tolerancia(
                self._planilha_atual().valores(opcoes["coluna"]),
                opcoes["coluna"], opcoes["cobertura"], opcoes["confianca"],
                opcoes["lado"])
            self.sessao.acrescentar(render_composto(resultado))

    # ------------------------------------------------------------------- MSA
    def _dialogo_grr(self, titulo: str) -> DialogoAnalise:
        return (DialogoAnalise(titulo, self)
                .combo_coluna("medicao", "Coluna de medições:",
                              self._colunas_numericas())
                .combo_coluna("peca", "Coluna de peças:",
                              self._todas_colunas_com_dados())
                .combo_coluna("operador", "Coluna de operadores:",
                              self._todas_colunas_com_dados()))

    def _f3_grr_cruzado(self) -> None:
        opcoes = (self._dialogo_grr("Gage R&R cruzado")
                  .caixa("grafico", "Gage run chart", True)
                  .executar())
        if opcoes:
            planilha = self._planilha_atual()
            resultado = mod_msa.gage_rr_cruzado(planilha.df, opcoes["medicao"],
                                                opcoes["peca"], opcoes["operador"])
            self.sessao.acrescentar(render_composto(resultado))
            if opcoes["grafico"]:
                dados = resultado.dados["dados"]
                self._abrir_grafico("Gage run chart",
                                    lambda: qp.gage_run(dados, opcoes["medicao"]))

    def _f3_grr_aninhado(self) -> None:
        opcoes = self._dialogo_grr("Gage R&R aninhado").executar()
        if opcoes:
            resultado = mod_msa.gage_rr_aninhado(
                self._planilha_atual().df, opcoes["medicao"], opcoes["peca"],
                opcoes["operador"])
            self.sessao.acrescentar(render_composto(resultado))

    def _f3_gage_run(self) -> None:
        opcoes = self._dialogo_grr("Gage run chart").executar()
        if opcoes:
            dados = mod_msa._dados_msa(self._planilha_atual().df,
                                       opcoes["medicao"], opcoes["peca"],
                                       opcoes["operador"])
            self._abrir_grafico("Gage run chart",
                                lambda: qp.gage_run(dados, opcoes["medicao"]))

    def _f3_linearidade(self) -> None:
        colunas = self._colunas_numericas(2)
        opcoes = (DialogoAnalise("Linearidade e viés", self)
                  .combo_coluna("medicao", "Coluna de medições:", colunas)
                  .combo_coluna("referencia", "Coluna de valores de referência:",
                                colunas)
                  .caixa("tem_vp", "Informar variação do processo (6σ)", False)
                  .numero("vp", "Variação do processo:", 1.0, 1e-12)
                  .executar())
        if opcoes:
            resultado = mod_msa.linearidade_vies(
                self._planilha_atual().df, opcoes["medicao"],
                opcoes["referencia"],
                opcoes["vp"] if opcoes["tem_vp"] else None)
            self.sessao.acrescentar(render_composto(resultado))

    def _f3_tipo1(self) -> None:
        opcoes = (DialogoAnalise("Estudo Tipo 1", self)
                  .combo_coluna("coluna",
                                "Coluna de medições (mesma peça repetida):",
                                self._colunas_numericas())
                  .numero("referencia", "Valor de referência da peça:", 0.0)
                  .numero("tolerancia", "Tolerância (LSE − LIE):", 1.0, 1e-12)
                  .numero("k", "% da tolerância para o critério:", 20.0, 1, 100, 0)
                  .executar())
        if opcoes:
            resultado = mod_msa.estudo_tipo1(
                self._planilha_atual().valores(opcoes["coluna"]),
                opcoes["coluna"], opcoes["referencia"], opcoes["tolerancia"],
                opcoes["k"])
            self.sessao.acrescentar(render_composto(resultado))

    def _f3_concordancia(self) -> None:
        colunas = self._todas_colunas_com_dados()
        opcoes = (DialogoAnalise("Concordância por atributos", self)
                  .combo_coluna("peca", "Coluna de peças/itens:", colunas)
                  .combo_coluna("avaliador", "Coluna de avaliadores:", colunas)
                  .combo_coluna("resultado", "Coluna de resultados:", colunas)
                  .combo_coluna("padrao", "Coluna do padrão (opcional):",
                                [SEM_ESTAGIO] + colunas)
                  .executar())
        if opcoes:
            padrao = None if opcoes["padrao"] == SEM_ESTAGIO else opcoes["padrao"]
            resultado = mod_msa.concordancia_atributos(
                self._planilha_atual().df, opcoes["peca"], opcoes["avaliador"],
                opcoes["resultado"], padrao)
            self.sessao.acrescentar(render_composto(resultado))

    def _f3_planilha_coleta(self) -> None:
        opcoes = (DialogoAnalise("Planilha de coleta — Gage R&R", self)
                  .inteiro("pecas", "Número de peças:", 10, 2, 50)
                  .inteiro("operadores", "Número de operadores:", 3, 2, 10)
                  .inteiro("replicas", "Número de repetições:", 3, 2, 10)
                  .inteiro("semente", "Semente (0 = aleatória):", 0, 0, 10**9)
                  .executar())
        if opcoes:
            tabela = mod_msa.planilha_coleta_grr(
                opcoes["pecas"], opcoes["operadores"], opcoes["replicas"],
                opcoes["semente"] or None)
            self._nova_planilha(Planilha("Coleta Gage R&R", tabela))
            self.statusBar().showMessage(
                "Planilha de coleta criada em ordem aleatorizada — preencha a "
                "coluna 'medição' durante o estudo.", 8000)

    # -------------------------------------------------------- dados exemplo
    def _dados_exemplo_qualidade(self) -> None:
        rng = np.random.default_rng(21)
        # processo com deslocamento de média no meio (para cartas/capabilidade)
        medida = np.concatenate([rng.normal(50, 1.5, 40),
                                 rng.normal(51.5, 1.5, 20)])
        processo = pd.DataFrame({
            "medida": np.round(medida, 2),
            "estagio": ["antes"] * 40 + ["depois"] * 20,
            "defeituosos": rng.binomial(200, 0.03, 60),
            "tamanho_amostra": np.full(60, 200),
            "defeitos": rng.poisson(4.0, 60),
        })
        self._nova_planilha(Planilha("Processo (exemplo)", processo))

        # estudo Gage R&R balanceado: 10 peças × 3 operadores × 2 repetições
        pecas = rng.normal(0, 2.0, 10)
        vies_oper = {"Ana": 0.0, "Bruno": 0.3, "Carla": -0.2}
        linhas = []
        for i, vp in enumerate(pecas):
            for oper, vo in vies_oper.items():
                for _ in range(2):
                    linhas.append({
                        "peça": f"P{i + 1:02d}", "operador": oper,
                        "medição": round(50 + vp + vo + rng.normal(0, 0.25), 3)})
        self._nova_planilha(Planilha("MSA (exemplo)", pd.DataFrame(linhas)))
        self.statusBar().showMessage(
            "Duas planilhas criadas: 'Processo (exemplo)' (cartas/capabilidade, "
            "com deslocamento no estágio 'depois') e 'MSA (exemplo)' (Gage R&R).",
            10000)
