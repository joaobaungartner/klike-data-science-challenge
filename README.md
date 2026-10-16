# Klike Data Science Challenge

Análise exploratória, modelagem preditiva do **Klike Score** e protótipo interativo de recomendações para criativos. Dataset simulado do desafio Millai / Klike.

## Teste o Recommendations Engine

**[Abrir notebook interativo](notebooks/parte-3-recommendations-engine.ipynb)**: informe uma campanha nova e clique em **Gerar recomendações**. Não é preciso alterar código. O formulário exige execução em JupyterLab ou no VS Code; o visualizador do GitHub não executa widgets.

### Instalação

Recomendado: Python 3.13. Abra um terminal **na raiz do repositório**:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab
```

Abra `notebooks/parte-3-recommendations-engine.ipynb`, selecione o kernel do ambiente virtual, execute as três células Python em ordem e preencha o formulário. Funciona também no VS Code com extensões Python e Jupyter.

> O Ridge serializado foi treinado com **scikit-learn 1.9.1** (fixado nas dependências). Carregue arquivos joblib apenas de fontes confiáveis.

## Organização

```text
data/klike_challenge_dataset.csv
models/ridge_klike.joblib
models/features_klike.joblib
notebooks/parte-1.ipynb
notebooks/parte-2.ipynb
notebooks/parte-3.ipynb                          # análise, evidências e testes
notebooks/parte-3-recommendations-engine.ipynb   # interface interativa
src/recommendations_engine.py                    # lógica reutilizável
tests/test_recommendations_engine.py
requirements.txt
```

Os caminhos no notebook da Parte 3 funcionam tanto com a raiz quanto com a pasta `notebooks/` como diretório de trabalho.

## Metodologia

- O Ridge reduzido prevê o Klike Score a partir de dez features, preservando as demais na simulação de uma alteração.
- As comparações observacionais usam apenas 400 campanhas do treinamento (`random_state=42`); as métricas de CTR inconsistentes são desconsideradas somente na análise correspondente.
- O suporte amostral mínimo é de 10 campanhas por grupo, com bootstrap exploratório de 2.000 reamostragens; os candidatos precisam ter delta previsto de ao menos um ponto e associação observada favorável.
- Os resultados são ordenados por diferença **prevista**, sem somar métricas de escalas diferentes. Campanhas podem ter menos de três recomendações elegíveis.

Na campanha hipotética de demonstração Meta / E-commerce, o protótipo encontrou três sugestões: hook, rosto humano e legendas.

**Limitações:** diferenças entre previsões e associações não são efeitos causais; o bootstrap não quantifica incerteza preditiva do Ridge; não foram controlados confundimento nem múltiplas comparações. Os limiares utilizados são heurísticos.

## Testes

```bash
python -m pytest -q
```

Para conferir a análise integral, abra `notebooks/parte-3.ipynb` e execute **Restart Kernel / Run All**.

## Notebooks

- [Parte 1 — EDA](notebooks/parte-1.ipynb)
- [Parte 2 — Modelagem](notebooks/parte-2.ipynb)
- [Parte 3 — Desenvolvimento e validação](notebooks/parte-3.ipynb)
- [Parte 3 — Interface interativa](notebooks/parte-3-recommendations-engine.ipynb)
