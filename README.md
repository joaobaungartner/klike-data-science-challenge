# Klike Data Science Challenge

Análise exploratória, modelagem preditiva do **Klike Score** e recomendações para criativos, com dados simulados do desafio.

## Teste sem modificar os notebooks (recomendado)

A interface Streamlit funciona no navegador e **substitui os resultados em cada execução**. Ao testar uma campanha, ela não altera arquivos `.ipynb` e não cria mudanças no Git.

Na raiz do repositório:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

A página abrirá no navegador. Preencha as dez características do criativo e clique em **Gerar recomendações**. Cada novo envio substitui o resultado anterior.

## Notebooks de análise

Os notebooks documentam a metodologia e os testes:
- [Parte 1 — EDA](notebooks/parte-1.ipynb)
- [Parte 2 — Modelagem](notebooks/parte-2.ipynb)
- [Parte 3 — Metodologia](notebooks/parte-3.ipynb)
- [Parte 3 — Interface ipywidgets](notebooks/parte-3-recommendations-engine.ipynb)

A interface ipywidgets foi mantida como opção, e seu painel agora **substitui**, em vez de acumular, as respostas.

**Importante:** ao executar notebooks versionados em Jupyter ou VS Code, o `.ipynb` pode ser marcado como modificado porque outputs e metadados são gravados dentro dele. Isso é comportamento normal e não implica mudança no código. Se você não alterou intencionalmente o notebook, use `git restore notebooks/parte-3-recommendations-engine.ipynb` para descartar apenas os outputs locais, após conferir que não há mudanças que queira guardar.

## Organização

```text
data/klike_challenge_dataset.csv
models/ridge_klike.joblib
models/features_klike.joblib
notebooks/parte-1.ipynb
notebooks/parte-2.ipynb
notebooks/parte-3.ipynb                          # análise, evidências e testes
notebooks/parte-3-recommendations-engine.ipynb   # interface interativa opcional
app.py                                           # interface Streamlit recomendada
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
