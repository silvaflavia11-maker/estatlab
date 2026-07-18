# Histórico de versões — EstatLab

Formato baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/);
versionamento [SemVer](https://semver.org/lang/pt-BR/).

## [1.0.0] — 2026-07 (pendente de release)

Primeira versão completa, com as quatro fases do escopo aprovado:

- **Fase 1 — Fundação e estatística básica**: planilha de dados (importação
  CSV/XLSX, projeto salvável .estat), estatística descritiva, testes Z e t,
  proporções, variâncias, taxas de Poisson, correlação/covariância,
  normalidade, outliers (Grubbs/Dixon), gráficos básicos com exportação e
  janela de sessão com interpretação didática em PT-BR.
- **Fase 2 — Inferência ampliada**: ANOVA de 1 e 2 fatores com comparações
  múltiplas (Tukey, Fisher, Dunnett), Levene/Bartlett, regressão linear com
  stepwise e melhores subconjuntos, logística binária, Poisson, testes não
  paramétricos, qui-quadrado/Fisher exato, calculadora de distribuições,
  geração de amostras e poder/tamanho de amostra com curvas.
- **Fase 3 — Qualidade industrial**: cartas de controle (I-MR, Xbarra-R/S com
  estágios, P, NP, C, U, MA, EWMA, CUSUM) com testes de Nelson, capabilidade
  (normal, Box-Cox/Johnson, identificação de distribuição, atributos,
  relatório completo), run chart, Pareto, Ishikawa, Multi-Vari, intervalos de
  tolerância e MSA (Gage R&R cruzado/aninhado, linearidade e viés, Tipo 1,
  concordância por atributos, planilha de coleta).
- **Fase 4 — Métodos avançados**: DOE (fatoriais 2^k e frações,
  Plackett-Burman, fatorial geral, CCD/Box-Behnken, Taguchi; método de Lenth;
  otimização de resposta; contorno/superfície 3D), séries temporais
  (tendência, decomposição, suavizações, Winters, ACF/PACF/CCF, ARIMA),
  multivariada (PCA, fatorial, discriminante, cluster, correspondência,
  alfa de Cronbach), confiabilidade (Kaplan-Meier, Weibull/Lognormal/
  Exponencial com censura), regressões não linear/ordinal/nominal/PLS/
  ortogonal, equivalência (TOST), bootstrap, aleatorização e árvores.

Validação: 147 testes automatizados comparando com bibliotecas de referência,
fórmulas recomputadas de forma independente e valores clássicos publicados.
