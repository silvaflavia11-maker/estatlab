# Contributing to EstatLab / Contribuindo com o EstatLab

*(English below — o EstatLab é um projeto em português brasileiro; issues e
PRs são bem-vindos em português ou inglês.)*

## Como buscar suporte ou relatar problemas

- **Dúvidas de uso e pedidos de suporte**: abra uma
  [issue](https://github.com/silvaflavia11-maker/estatlab/issues) com o
  rótulo `pergunta`.
- **Bugs**: abra uma issue com (1) os passos para reproduzir, (2) o
  resultado obtido × esperado, (3) o sistema operacional e a versão do
  Python, e — se possível — (4) o arquivo `.estat` do projeto ou os dados
  mínimos que reproduzem o problema. Em erros numéricos, indique a
  referência de comparação (livro, R, outro software).
- **Sugestões de melhoria**: abra uma issue descrevendo o caso de uso
  didático ou de análise que a motivou.

## Como contribuir com código

1. Faça um *fork* e crie um branch a partir de `main`.
2. Instale o ambiente de desenvolvimento:

   ```bash
   python3 -m venv .venv
   ./.venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip
   ```

3. Faça sua alteração seguindo as convenções do projeto:
   - **Métodos estatísticos nunca são reimplementados** quando existem em
     scipy/statsmodels/scikit-learn/lifelines — o `app/core/` só orquestra.
     Implementações próprias (quando inevitáveis) exigem fonte bibliográfica
     citada no docstring e teste com valor de referência publicado.
   - Todo método novo em `app/core/` precisa de teste de validação numérica
     em `tests/` (compare com a biblioteca de referência ou com valores
     publicados, 4 casas decimais).
   - Interface e mensagens em PT-BR; números exibidos com vírgula decimal.
   - Funções de `app/core/` não importam Qt.
4. Rode a suíte completa e confirme que está verde:

   ```bash
   ./.venv/bin/python -m pytest -q
   ```

5. Abra o *pull request* descrevendo o que muda e por quê.

## Conduta

Este projeto adota uma regra simples: seja respeitoso e construtivo.
Críticas dirigem-se a código e ideias, nunca a pessoas. Comportamento
abusivo leva a bloqueio.

---

# English

EstatLab is an educational statistics application in Brazilian Portuguese.
Issues and pull requests are welcome in Portuguese or English.

- **Support / questions**: open an
  [issue](https://github.com/silvaflavia11-maker/estatlab/issues) labelled
  `pergunta`.
- **Bug reports**: include steps to reproduce, observed × expected results,
  OS and Python version, and minimal data (or a `.estat` project file).
  For numerical discrepancies, name the reference (textbook, R, other
  software).
- **Code contributions**: fork, branch from `main`, install with
  `pip install -r requirements.txt`, and follow the project rules:
  statistical methods are never reimplemented when available in
  scipy/statsmodels/scikit-learn/lifelines; every new `app/core/` function
  requires a numerical validation test in `tests/`; UI strings are in
  Brazilian Portuguese; `app/core/` must stay Qt-free. Run
  `python -m pytest -q` (147 tests) before opening the PR.

Be respectful and constructive; abusive behaviour leads to a block.
