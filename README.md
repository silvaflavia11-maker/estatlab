# EstatLab

Aplicativo desktop de análise estatística em português, para ensino e uso
acadêmico. Multiplataforma (macOS e Windows). Distribuído sob licença MIT,
sem garantia de suporte externo (projeto acadêmico).

## Como citar

Se você usar o EstatLab em trabalho acadêmico, cite o software (metadados em
`CITATION.cff`; DOI via Zenodo após o primeiro release) **e** as bibliotecas
de cálculo que realizam as análises — SciPy, statsmodels, NumPy, pandas,
scikit-learn e lifelines —, conforme as recomendações de citação de cada
projeto. Exemplo (APA):

> Silva, F. (2026). *EstatLab: aplicativo estatístico educacional*
> (Versão 1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX
> *(DOI definitivo após o primeiro release)*

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

O PyInstaller não faz cross-compilation: gere o executável de cada sistema no
próprio sistema.

```bash
# macOS
./build/build_macos.sh

# Windows (prompt de comando na pasta do projeto)
build\build_windows.bat
```

O executável fica em `dist/`. Na primeira execução, sistemas exibem um aviso
de segurança para apps não assinados (Gatekeeper/SmartScreen) — como o uso é
interno, basta autorizar: no macOS, clique com o botão direito → Abrir; no
Windows, "Mais informações" → "Executar assim mesmo".

## Estrutura

```
app/core/       motor estatístico (funções puras, sem Qt, 100% testável)
app/worksheet/  planilha de dados e modelo Qt
app/plots/      geração de gráficos (matplotlib)
app/reports/    formatação PT-BR e interpretação didática (HTML da sessão)
app/ui/         janelas e diálogos (PySide6)
app/i18n/       strings compartilhadas
tests/          validação numérica + testes de fumaça da interface
build/          scripts de empacotamento (macOS e Windows)
dist/           executável gerado (EstatLab.app) — recriado pelo build
docs/           documentos do projeto (especificação, guias, detalhamento)
distribuicao/   pacotes prontos para envio à equipe (zips)
```

## Documentos do projeto (pasta `docs/`, local — não versionada)

- `prompt-app-estatistica.md` — especificação para desenvolvimento assistido
- `Detalhamento-Tecnico-App-Estatistico-v0.2.docx` — detalhamento técnico
  revisado pela equipe (escopo aprovado; requisito multiplataforma);
  v0.1 arquivada na mesma pasta
- `Guia-Rapido-EstatLab.docx` (e .pdf) — guia do usuário para a equipe de
  validação
- `Instrucoes-Build-Windows-EstatLab.docx` — passo a passo para gerar o
  executável do Windows (pendência em aberto)

## Distribuição (pasta `distribuicao/`)

- `EstatLab-macOS.zip` — executável do macOS pronto para envio
- `EstatLab-codigo-fonte.zip` — código mínimo para gerar o executável do
  Windows (regenerar após mudanças no código: ver comando no histórico ou
  recriar com zip de app/, tests/, build/*.sh|bat, requirements.txt,
  pytest.ini e README.md)
