# EstatLab

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21435514.svg)](https://doi.org/10.5281/zenodo.21435514)
[![CI](https://github.com/silvaflavia11-maker/estatlab/actions/workflows/ci.yml/badge.svg)](https://github.com/silvaflavia11-maker/estatlab/actions/workflows/ci.yml)

Aplicativo desktop de análise estatística em português, para ensino e uso
acadêmico. Multiplataforma (Windows, macOS e Linux). Distribuído sob licença
MIT. *English summary [below](#english).*

Toda análise produz, além dos números, uma **interpretação didática**:
hipóteses por extenso, p-valor, decisão ao nível de significância escolhido
e avisos de pressupostos. Dúvidas, bugs e sugestões: abra uma
[issue](https://github.com/silvaflavia11-maker/estatlab/issues) — veja o
[guia de contribuição](CONTRIBUTING.md).

## Como citar

Se você usar o EstatLab em trabalho acadêmico, cite o software (metadados em
`CITATION.cff`; DOI via Zenodo após o primeiro release) **e** as bibliotecas
de cálculo que realizam as análises — SciPy, statsmodels, NumPy, pandas,
scikit-learn e lifelines —, conforme as recomendações de citação de cada
projeto. Exemplo (APA):

> Silva, F. (2026). *EstatLab: aplicativo estatístico educacional*
> (Versão 1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.21435514

**Fase 1 implementada:** planilha de dados, estatística descritiva, testes de
hipóteses (Z, t, proporções, variâncias, taxas de Poisson), correlação e
covariância, testes de normalidade e de outliers, e gráficos (histograma,
boxplot, dotplot, dispersão, matriz de dispersão, barras, pizza, série
temporal, probabilidade normal) — sempre com interpretação didática na
janela de sessão.

**Fase 2 implementada:**
- **ANOVA** de 1 fator (com Tukey, Fisher LSD e Dunnett) e de 2 fatores
  (com interação), igualdade de variâncias (Levene/Bartlett), gráficos de
  resíduos (4 painéis), efeitos principais e interação;
- **Regressão** linear simples/múltipla (IC dos coeficientes, VIF, predição
  com IC/IP), stepwise (p-valor, AICc, BIC), melhores subconjuntos,
  logística binária (razões de chances) e Poisson (razões de taxas, alerta
  de superdispersão);
- **Não paramétricos**: sinal, Wilcoxon (1 amostra e pareado), Mann-Whitney,
  Kruskal-Wallis, mediana de Mood, Friedman e sequências (runs);
- **Tabelas**: tabulação cruzada com qui-quadrado e V de Cramér, exato de
  Fisher, qui-quadrado de aderência;
- **Distribuições**: calculadora (densidade/acumulada/inversa) para 9
  distribuições, gráficos, geração de números aleatórios e amostragem;
- **Poder e tamanho de amostra**: Z, t (1/2 amostras, pareado), proporções,
  variâncias e ANOVA, com curvas de poder.

**Fase 3 implementada:**
- **Cartas de controle**: I-MR, Xbarra-R, Xbarra-S (com estágios históricos),
  P, NP, C, U, MA, EWMA e CUSUM, com testes de causas especiais de Nelson
  (1–8) configuráveis e pontos sinalizados no gráfico;
- **Capabilidade de processo**: normal (Cp/Cpk, Pp/Ppk, Cpm, PPM), não normal
  via transformações Box-Cox e Johnson, identificação de distribuição
  individual (ranking por Anderson-Darling), capabilidade por atributos e
  Relatório de Capabilidade Completo (painel de 6 gráficos);
- **Ferramentas da qualidade**: run chart (com teste de aleatoriedade),
  Pareto, diagrama de Ishikawa, carta Multi-Vari (2–3 fatores) e intervalos
  de tolerância (normal e não paramétrico);
- **MSA**: Gage R&R cruzado e aninhado (método ANOVA, com %contribuição,
  %variação do estudo e ndc), gage run chart, linearidade e viés, estudo
  Tipo 1 (Cg/Cgk), concordância por atributos (kappa de Fleiss e Cohen) e
  geração de planilha de coleta aleatorizada.

**Fase 4 implementada (escopo completo):**
- **DOE**: geração de planos (fatoriais 2^k completos e fracionados,
  Plackett-Burman, fatorial geral, CCD/Box-Behnken, Taguchi L4–L16) e
  análise (efeitos com método de Lenth para planos saturados, corridas
  perdidas via regressão, superfície de resposta, análise Taguchi por razão
  S/N, otimização de resposta, gráficos de Pareto/normal/meio-normal dos
  efeitos, cubo, contorno e superfície 3D rotacionável);
- **Séries temporais**: tendência (linear/quadrática/exponencial),
  decomposição, média móvel, suavização exponencial simples/dupla (Holt),
  Winters, ACF/PACF/CCF e ARIMA com previsões e diagnóstico de Ljung-Box;
- **Multivariada**: PCA (scree/escores), análise fatorial, discriminante
  linear, cluster k-médias e hierárquico (dendrograma; grupos gravados na
  planilha), correspondência (mapa) e alfa de Cronbach;
- **Confiabilidade**: Kaplan-Meier, análise paramétrica Weibull/Lognormal/
  Exponencial com censura à direita, à esquerda e por intervalo, percentis
  (B-lives), gráficos de sobrevivência/risco e probabilidade Weibull;
- **Regressões adicionais**: não linear (curve fit), logística ordinal e
  nominal, PLS, ortogonal (Deming); testes de equivalência (TOST);
  bootstrap (BCa) e testes de aleatorização; árvores de classificação e
  regressão.

## Como executar (desenvolvimento)

```bash
# 1. criar o ambiente (uma única vez)
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt        # macOS/Linux
# .venv\Scripts\pip install -r requirements.txt    # Windows

# 2. rodar o app
./.venv/bin/python -m app.main                     # macOS/Linux
# .venv\Scripts\python -m app.main                 # Windows
```

Para experimentar rapidamente: menu **Ajuda → Carregar dados de exemplo**.

## Testes

```bash
./.venv/bin/python -m pytest
```

Todos os métodos estatísticos têm testes de validação numérica comparando com
scipy/statsmodels, fórmulas recomputadas de forma independente e valores
clássicos publicados (ver `tests/test_core.py`).

## Empacotamento (executável)

O PyInstaller não faz cross-compilation: cada executável é gerado no próprio
sistema.

```bash
# macOS (local)
./build/build_macos.sh

# Windows (local, numa máquina Windows)
build\build_windows.bat
```

**Windows sem máquina Windows:** o workflow **Build Windows** (aba *Actions*
do GitHub) gera o `EstatLab-Windows.zip` num runner Windows — roda
automaticamente a cada release ou manualmente por *Run workflow*; baixe o
artefato na página da execução.

O executável fica em `dist/`. Na primeira execução, sistemas exibem um aviso
de segurança para apps não assinados (Gatekeeper/SmartScreen): no macOS,
clique com o botão direito → Abrir; no Windows, "Mais informações" →
"Executar assim mesmo".

## Estrutura

```
app/core/       motor estatístico (funções puras, sem Qt, 100% testável)
app/worksheet/  planilha de dados e modelo Qt
app/plots/      geração de gráficos (matplotlib)
app/reports/    formatação PT-BR e interpretação didática (HTML da sessão)
app/ui/         janelas e diálogos (PySide6)
app/i18n/       strings compartilhadas
tests/          validação numérica + testes de fumaça da interface
paper/          artigo para o JOSS (paper.md + paper.bib)
.github/        CI (testes em 3 SOs), build Windows e formulários de issue
build/          scripts de empacotamento local (macOS e Windows)
dist/           executável gerado (EstatLab.app) — recriado pelo build
docs/           documentos do projeto (local, não versionado)
distribuicao/   pacotes prontos para envio à equipe (local, não versionado)
```

## Documentos do projeto (pasta `docs/`, local — não versionada)

- `prompt-app-estatistica.md` — especificação para desenvolvimento assistido
- `Detalhamento-Tecnico-App-Estatistico-v0.2.docx` — detalhamento técnico
  aprovado pela equipe
- `Guia-Rapido-EstatLab.docx` (e .pdf) — guia do usuário para a equipe de
  validação
- `Guia-Issues-GitHub-EstatLab.docx` — como a equipe registra bugs e
  sugestões via GitHub
- `arquivo/` — documentos concluídos ou supersedidos (detalhamento v0.1,
  instruções de build manual no Windows, checklist da publicação com DOI)

## Distribuição (pasta `distribuicao/`, local — não versionada)

- `EstatLab-macOS.zip` — executável do macOS pronto para envio à equipe
- Executável do Windows: gerar pelo workflow **Build Windows** na aba
  Actions do GitHub (artefato `EstatLab-Windows.zip`)
- Código-fonte: o repositório público https://github.com/silvaflavia11-maker/estatlab
  substituiu o antigo zip de código

---

## English

**EstatLab** is an open-source desktop application for statistical analysis
with a GUI entirely in Brazilian Portuguese, built for teaching and academic
use. Every analysis outputs a *didactic interpretation* alongside the
numbers: hypotheses in plain language, p-value, the decision at the chosen
significance level, and assumption warnings. Coverage spans basic statistics,
hypothesis tests, ANOVA, regression, nonparametrics, contingency tables,
distributions, power and sample size, control charts, process capability,
measurement systems analysis (Gage R&R), design of experiments, time series,
multivariate methods and reliability/survival — with all numerical
procedures delegated to SciPy, statsmodels, scikit-learn and lifelines, and
validated by an automated suite of 147 tests.

**Install & run (from source):**

```bash
git clone https://github.com/silvaflavia11-maker/estatlab.git
cd estatlab
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip
./.venv/bin/python -m app.main                # Windows: .venv\Scripts\python
```

Try it: menu **Ajuda → Carregar dados de exemplo** loads sample datasets.

**Tests:** `python -m pytest -q` (147 tests; set `QT_QPA_PLATFORM=offscreen`
on headless systems).

**How to cite:** see `CITATION.cff` or the DOI badge above — and please also
cite the underlying scientific libraries.

**Contributing / support:** see [CONTRIBUTING.md](CONTRIBUTING.md); open an
[issue](https://github.com/silvaflavia11-maker/estatlab/issues) in Portuguese
or English.
